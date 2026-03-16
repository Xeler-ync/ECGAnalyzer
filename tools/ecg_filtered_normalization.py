import os.path
import numpy as np
import concurrent.futures
from tqdm import tqdm
from ecg_heartbeat_normalization_pipeline import (
    process_ecg_signal,
    process_lead_with_r_peaks,
)
from utils._config import (
    SAMPLING_RATE,
    RESULTS_PATH,
    MAX_WORKERS,
    PATH,
    LEAD_NAMES,
    SCP_CODES,
    MIN_VALUES,
)
from utils._data import load_raw_data, Y


def process_and_save_filtered_heartbeats(scp_code_key, min_value, output_dir=None):
    """
    Process ECG signals that meet specific criteria and save normalized heartbeats to npy files.

    Parameters:
    - scp_code_key: String key or list of keys to filter Y.scp_codes dictionary
    - min_value: Minimum value or list of minimum values for the scp_code_key to include in processing
    - output_dir: Directory to save the npy files. If None, uses RESULTS_PATH

    Returns:
    - processed_count: Number of signals processed
    """
    # Set output directory
    if output_dir is None:
        output_dir = RESULTS_PATH
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Convert single values to lists for uniform processing
    scp_code_keys = [scp_code_key] if isinstance(scp_code_key, str) else scp_code_key
    min_values = [min_value] if isinstance(min_value, (int, float)) else min_value

    # Ensure scp_code_keys and min_values have the same length
    if len(scp_code_keys) != len(min_values):
        raise ValueError("Length of scp_code_keys and min_values must be the same")

    # Filter signals based on scp_codes
    filtered_indices = []
    for idx in range(Y.patient_id.count()):
        scp_codes = Y.scp_codes.iloc[idx]
        # Check if all criteria are met (AND logic)
        criteria_met = True
        for key, value in zip(scp_code_keys, min_values):
            if key not in scp_codes or scp_codes[key] < value:
                criteria_met = False
                break

        if criteria_met:
            filtered_indices.append(idx)

    print(
        f"Found {len(filtered_indices)} signals matching criteria: {', '.join([f'{k}>={v}' for k, v in zip(scp_code_keys, min_values)])}"
    )

    # Process each filtered signal using thread pool
    processed_count = 0
    error_count = 0

    def process_single_signal(signal_index):
        """Process a single ECG signal and save normalized heartbeats"""
        try:
            ecg_id = Y.index[signal_index]

            # Load and process the signal
            X = load_raw_data(Y, SAMPLING_RATE, PATH, signal_index)
            all_leads_normalized = {}

            # Process Lead II first to get R-peaks
            lead_II_idx = 1
            results_II = process_ecg_signal(X[0, :, lead_II_idx], lead_II_idx)
            all_leads_normalized[lead_II_idx] = results_II["normalized_heartbeats"]
            r_peaks_II = results_II["r_peaks"]

            # Process all other leads using Lead II R-peaks
            for lead_idx in range(12):
                if lead_idx == lead_II_idx:
                    continue
                all_leads_normalized[lead_idx] = process_lead_with_r_peaks(
                    X[0, :, lead_idx], r_peaks_II, lead_idx
                )

            for lead_idx, lead_name in enumerate(LEAD_NAMES):
                if lead_idx in all_leads_normalized and all_leads_normalized[lead_idx]:
                    # Extract signals from normalized heartbeats
                    normalized_heartbeats = all_leads_normalized[lead_idx]
                    signals = np.array(
                        [beat["signal"] for beat in normalized_heartbeats]
                    )

                    # Create filename with ecg_id and save
                    filename = os.path.join(
                        output_dir, f"{ecg_id}_{lead_name}_normalized_heartbeats.npy"
                    )
                    np.save(filename, signals)

            return True
        except Exception as e:
            print(f"Error processing signal {signal_index}: {e}")
            return False

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_signal, idx): idx for idx in filtered_indices
        }

        # Process results as they complete
        for future in tqdm(
            concurrent.futures.as_completed(future_to_index),
            total=len(filtered_indices),
            desc="Processing filtered signals",
        ):
            if future.result():
                processed_count += 1
            else:
                error_count += 1

    print(
        f"Successfully processed {processed_count} out of {len(filtered_indices)} filtered signals"
    )
    if error_count > 0:
        print(f"Failed to process {error_count} signals")
    return processed_count


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Process ECG signals with specific SCP codes"
    )
    parser.add_argument(
        "--scp_codes",
        type=str,
        nargs="+",
        default=SCP_CODES,
        help="SCP code key(s) to filter signals (space-separated list)",
    )
    parser.add_argument(
        "--min_value",
        type=float,
        nargs="+",
        default=MIN_VALUES,
        help="Minimum value(s) for the SCP code(s) (space-separated list)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory for npy files"
    )

    args = parser.parse_args()

    # Ensure scp_codes and min_value have the same length
    if len(args.scp_codes) != len(args.min_value):
        raise ValueError(
            "Number of scp_codes must match number of min_value parameters"
        )

    print("=" * 80)
    print("ECG Signal Processing with SCP Code Filtering")
    print("=" * 80)
    print(
        f"Filter criteria: {', '.join([f'{k}>={v}' for k, v in zip(args.scp_codes, args.min_value)])}"
    )

    # Create output directory name based on filter criteria
    criteria_str = "_".join(
        [f"{k}_{v}" for k, v in zip(args.scp_codes, args.min_value)]
    )
    output_dir = args.output_dir or f"./results/{criteria_str}"

    process_and_save_filtered_heartbeats(
        scp_code_key=args.scp_codes, min_value=args.min_value, output_dir=output_dir
    )

    print("=" * 80)
    print("ECG Signal Processing Completed")
    print("=" * 80)


if __name__ == "__main__":
    main()
