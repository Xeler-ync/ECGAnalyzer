import os.path
import numpy as np
from wfdb import processing as wfdb_processing
import matplotlib.pyplot as plt
from utils._baseline import (
    remove_baseline_wander_hp_filter,
    remove_baseline_wander_savgol,
    remove_baseline_wander_morphological,
    evaluate_baseline_removal,
)
from utils._r_peaks import (
    detect_r_peaks_neurokit_NeuroKit2,
    evaluate_r_peak_detection,
)
from utils._config import SAMPLING_RATE, RESULTS_PATH, PATH
from utils._data import load_raw_data, Y
from utils._heartbeats import (
    extract_heartbeats,
    split_and_resample_heartbeats,
    plot_heartbeats_overlay_normalized,
)
from utils._helpers import _round_and_clip_indices


def process_ecg_signal(ecg_signal, lead_index, sampling_rate=SAMPLING_RATE):
    """
    Process a single ECG lead: baseline removal -> R-peak detection -> heartbeat extraction and normalization

    Args:
        ecg_signal: Raw ECG signal array
        lead_index: Index of the lead being processed
        sampling_rate: Sampling rate of the signal in Hz

    Returns:
        Dictionary containing processed signals and evaluation metrics
    """
    # Step 1: Baseline removal
    filtered_signal = remove_baseline_wander_hp_filter(
        ecg_signal, sampling_rate, cutoff=0.5
    )

    # Evaluate baseline removal effectiveness
    baseline_eval = evaluate_baseline_removal(ecg_signal, filtered_signal)
    print(f"\nBaseline Removal Evaluation (Lead {lead_index}):")
    print(f"  Variance Reduction: {baseline_eval['variance_reduction']:.2f}%")
    print(f"  RMS Error: {baseline_eval['rms_error']:.4f}")

    # Step 2: R-peak detection
    r_peaks = detect_r_peaks_neurokit_NeuroKit2(filtered_signal, sampling_rate)
    r_peaks = _round_and_clip_indices(
        r_peaks, len(filtered_signal), filtered_signal, "NeuroKit2"
    )

    # Evaluate R-peak detection
    peak_eval = evaluate_r_peak_detection(r_peaks, filtered_signal, sampling_rate)
    print(f"\nR-Peak Detection Evaluation (Lead {lead_index}):")
    print(f"  Detected R-Peaks: {peak_eval['peak_count']}")
    print(f"  Estimated Heart Rate: {peak_eval['heart_rate_bpm']:.2f} BPM")
    print(f"  Detection Reliability: {peak_eval['detection_reliability']:.4f}")

    # Step 3: Extract heartbeats
    heartbeats = extract_heartbeats(filtered_signal, r_peaks, sampling_rate)
    print(f"\nHeartbeat Extraction (Lead {lead_index}):")
    print(f"  Extracted Heartbeats: {len(heartbeats)}")

    # Step 4: Normalize heartbeats
    normalized_heartbeats, pre_r_samp, post_r_samp, total_samp = (
        split_and_resample_heartbeats(heartbeats, sampling_rate)
    )
    print(f"\nHeartbeat Normalization (Lead {lead_index}):")
    print(f"  Normalized Heartbeats: {len(normalized_heartbeats)}")
    print(f"  Pre-R Samples: {pre_r_samp}")
    print(f"  Post-R Samples: {post_r_samp}")
    print(f"  Total Samples per Heartbeat: {total_samp}")

    return {
        "filtered_signal": filtered_signal,
        "r_peaks": r_peaks,
        "heartbeats": heartbeats,
        "normalized_heartbeats": normalized_heartbeats,
        "baseline_eval": baseline_eval,
        "peak_eval": peak_eval,
    }


def plot_all_leads_normalized_heartbeats(
    all_leads_normalized, sampling_rate, signal_index, max_beats=15
):
    """
    Display normalized heartbeats for all 12 leads in a single figure,
    maintaining the same visualization style as individual lead plots.

    Args:
        all_leads_normalized: Dictionary with lead indices as keys and normalized heartbeat lists as values
        sampling_rate: Sampling rate in Hz
        signal_index: Index of the signal being processed
        max_beats: Maximum number of heartbeats to display per lead
    """
    # Standard ECG lead names
    lead_names = [
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]

    # Create figure with 3 rows and 4 columns for 12 leads
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()  # Flatten for easier indexing

    # Process each lead
    for lead_idx, lead_name in enumerate(lead_names):
        ax = axes[lead_idx]

        # Check if this lead has normalized heartbeats
        if (
            lead_idx not in all_leads_normalized
            or len(all_leads_normalized[lead_idx]) == 0
        ):
            ax.text(
                0.5,
                0.5,
                f"No data for lead {lead_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Lead {lead_name}")
            continue

        # Get normalized heartbeats for this lead
        normalized_heartbeats = all_leads_normalized[lead_idx]
        display_beats = normalized_heartbeats[:max_beats]

        # Extract signals from heartbeat dictionaries
        signals = [beat["signal"] for beat in display_beats]
        signals_array = np.array(signals)

        # Calculate statistics
        mean_beat = np.mean(signals_array, axis=0)
        std_beat = np.std(signals_array, axis=0)

        # Calculate additional statistics
        max_std = np.max(std_beat)
        avg_std = np.mean(std_beat)
        std_of_std = np.std(std_beat)

        # Plot individual heartbeats with transparency
        for signal in signals:
            ax.plot(signal, alpha=0.2, linewidth=0.8, color="gray")

        # Plot mean heartbeat with confidence interval
        x = np.arange(len(mean_beat))

        # Plot standard deviation bands
        ax.fill_between(
            x,
            mean_beat - std_beat,
            mean_beat + std_beat,
            alpha=0.2,
            color="blue",
            label="±1 SD",
        )

        ax.fill_between(
            x,
            mean_beat - 2 * std_beat,
            mean_beat + 2 * std_beat,
            alpha=0.2,
            color="blue",
            label="±2 SD",
        )

        # Plot mean heartbeat
        ax.plot(x, mean_beat, linewidth=2, color="blue", label="Average")

        # Set figure properties for each subplot
        ax.set_title(
            f"Lead {lead_name} (n={len(display_beats)})\n"
            f"Max Std: {max_std:.4f} | Avg Std: {avg_std:.4f}\nStd of Std: {std_of_std:.4f}"
        )
        ax.set_xlabel("Sample Points")
        ax.set_ylabel("Amplitude (mV)")
        ax.grid(True, alpha=0.3)

        # Only add legend to first subplot to avoid clutter
        if lead_idx == 0:
            ax.legend(loc="upper right", fontsize="small")

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(
        os.path.join(
            RESULTS_PATH, f"{signal_index}_all_leads_normalized_heartbeats.png"
        )
    )


def main():
    """
    Main workflow: From file reading to displaying normalized heartbeats
    """
    print("=" * 80)
    print("ECG Signal Processing Workflow")
    print("=" * 80)

    # Process all 12 leads
    leads_to_process = list(range(12))  # 0-11 for 12 standard leads
    all_leads_normalized = {}

    # signal_index = 0
    for signal_index in range(0, Y.patient_id.count()):
        X = load_raw_data(Y, SAMPLING_RATE, PATH, signal_index)

        # Process each lead
        for lead_idx in leads_to_process:
            # Extract signal for this lead
            ecg_signal = X[0, :, lead_idx]

            # Process signal
            results = process_ecg_signal(ecg_signal, lead_idx)

            # Store normalized heartbeats
            all_leads_normalized[lead_idx] = results["normalized_heartbeats"]

        # Display normalized heartbeats for all leads
        print("\nDisplaying normalized heartbeats for all leads...")
        plot_all_leads_normalized_heartbeats(
            all_leads_normalized,
            SAMPLING_RATE,
            signal_index,
            max_beats=15,
        )

        # Clear the dictionary for next signal
        all_leads_normalized.clear()


if __name__ == "__main__":
    main()
