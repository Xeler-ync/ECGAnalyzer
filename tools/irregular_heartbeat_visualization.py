import sys
import os.path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures

from utils._baseline import remove_baseline_wander_hp_filter, evaluate_baseline_removal
from utils._r_peaks import detect_r_peaks_envelope, evaluate_r_peak_detection
from utils._config import SAMPLING_RATE, RESULTS_PATH, PATH, MAX_WORKERS, LEAD_NAMES
from utils._data import load_raw_data, Y
from utils._heartbeats import extract_heartbeats, split_and_resample_heartbeats
from utils._helpers import _round_and_clip_indices
import json

# Ensure output directory exists
output_dir = os.path.join(os.path.dirname(RESULTS_PATH), "irregular")
os.makedirs(output_dir, exist_ok=True)

# List of irregular ECG signal indices
with open("./data/irregular_indices.json", "r") as f:
    irregular = json.load(f)["irregular"]


def calculate_total_std_with_r_peaks(X, r_peaks, sampling_rate=SAMPLING_RATE):
    """
    Calculate total standard deviation of all leads when using specific R-peak positions

    Parameters:
        X: ECG data with shape (1, samples, leads)
        r_peaks: R-peak positions to use
        sampling_rate: Sampling rate of the ECG

    Returns:
        total_std: Total standard deviation across all leads
    """
    total_std = 0.0
    for lead_idx in range(12):
        lead_signal = X[0, :, lead_idx]
        filtered_signal = remove_baseline_wander_hp_filter(
            lead_signal, sampling_rate, cutoff=0.5
        )
        heartbeats = extract_heartbeats(filtered_signal, r_peaks, sampling_rate)
        normalized_heartbeats, _, _, _ = split_and_resample_heartbeats(
            heartbeats, sampling_rate
        )

        if len(normalized_heartbeats) > 0:
            signals = np.array([beat["signal"] for beat in normalized_heartbeats])
            lead_std = np.mean(np.std(signals, axis=0))
            total_std += lead_std

    return total_std


def process_ecg_signal(ecg_signal, lead_index, sampling_rate=SAMPLING_RATE):
    """Process single ECG lead: baseline removal -> R-peak detection -> heartbeat extraction and normalization"""
    filtered_signal = remove_baseline_wander_hp_filter(
        ecg_signal, sampling_rate, cutoff=0.5
    )
    baseline_eval = evaluate_baseline_removal(ecg_signal, filtered_signal)

    # R-peak detection
    r_peaks = _round_and_clip_indices(
        detect_r_peaks_envelope(filtered_signal, sampling_rate),
        len(filtered_signal),
        filtered_signal,
        "NeuroKit2",
    )
    peak_eval = evaluate_r_peak_detection(r_peaks, filtered_signal, sampling_rate)

    # Extract and normalize heartbeats
    heartbeats = extract_heartbeats(filtered_signal, r_peaks, sampling_rate)
    normalized_heartbeats, pre_r, post_r, total = split_and_resample_heartbeats(
        heartbeats, sampling_rate
    )

    return {
        "filtered_signal": filtered_signal,
        "r_peaks": r_peaks,
        "heartbeats": heartbeats,
        "normalized_heartbeats": normalized_heartbeats,
        "baseline_eval": baseline_eval,
        "peak_eval": peak_eval,
    }


def process_lead_with_r_peaks(
    ecg_signal, r_peaks, lead_index, sampling_rate=SAMPLING_RATE
):
    """Process ECG lead using provided R-peak positions"""
    filtered_signal = remove_baseline_wander_hp_filter(
        ecg_signal, sampling_rate, cutoff=0.5
    )
    heartbeats = extract_heartbeats(filtered_signal, r_peaks, sampling_rate)
    normalized_heartbeats, pre_r, post_r, total = split_and_resample_heartbeats(
        heartbeats, sampling_rate
    )
    return normalized_heartbeats


def plot_all_leads_normalized_heartbeats(
    all_leads_normalized,
    sampling_rate,
    signal_index,
    max_beats=15,
    used_r_peak_leads=None,
):
    """Display normalized heartbeats for all 12 leads in a single figure"""
    if used_r_peak_leads is None:
        used_r_peak_leads = []

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for lead_idx, lead_name in enumerate(LEAD_NAMES):
        ax = axes[lead_idx]
        if lead_idx not in all_leads_normalized or not all_leads_normalized[lead_idx]:
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

        normalized_heartbeats = all_leads_normalized[lead_idx][:max_beats]
        signals = np.array([beat["signal"] for beat in normalized_heartbeats])
        mean_beat = np.mean(signals, axis=0)
        std_beat = np.std(signals, axis=0)

        for signal in signals:
            ax.plot(signal, alpha=0.2, linewidth=0.8, color="gray")

        x = np.arange(len(mean_beat))
        ax.fill_between(
            x,
            mean_beat - std_beat,
            mean_beat + std_beat,
            alpha=0.2,
            color="blue",
            label="+/-1 SD",
        )
        ax.fill_between(
            x,
            mean_beat - 2 * std_beat,
            mean_beat + 2 * std_beat,
            alpha=0.2,
            color="blue",
        )
        ax.plot(x, mean_beat, linewidth=2, color="blue", label="Average")

        # Set title color to red if this lead's R-peaks were used
        title_color = "red" if lead_idx in used_r_peak_leads else "black"
        ax.set_title(
            f"Lead {lead_name} (n={len(normalized_heartbeats)})\nMax Std: {np.max(std_beat):.4f} | Avg Std: {np.mean(std_beat):.4f}",
            color=title_color,
        )
        ax.set_xlabel("Sample Points")
        ax.set_ylabel("Amplitude (mV)")
        ax.grid(True, alpha=0.3)
        if lead_idx == 0:
            ax.legend(loc="upper right", fontsize="small")

    plt.subplots_adjust(
        left=0.052, bottom=0.052, right=0.971, top=0.914, wspace=0.22, hspace=0.488
    )
    plt.savefig(
        os.path.join(output_dir, f"{signal_index}_all_leads_normalized_heartbeats.png")
    )
    plt.close()


def plot_baseline_removed_signal(X, signal_index, used_r_peak_leads=None):
    """
    Plot the baseline removed ECG signal for all 12 leads with R-peak markers

    Parameters:
        X: Original ECG data
        signal_index: Signal index
        used_r_peak_leads: List of lead indices whose R-peaks were used
    """
    if used_r_peak_leads is None:
        used_r_peak_leads = []

    # Create 3x4 subplot layout
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    # Get signal data
    signal = X[0, :, :]

    # Create time axis
    time_axis = np.arange(len(signal)) / SAMPLING_RATE

    # Process Lead II to get R-peak positions
    lead_II_idx = 1
    lead_II_signal = signal[:, lead_II_idx]
    filtered_II = remove_baseline_wander_hp_filter(
        lead_II_signal, SAMPLING_RATE, cutoff=0.5
    )
    r_peaks_II = _round_and_clip_indices(
        detect_r_peaks_envelope(filtered_II, SAMPLING_RATE),
        len(filtered_II),
        filtered_II,
        "NeuroKit2",
    )

    # Convert R-peak indices to time points
    r_peaks_time = r_peaks_II / SAMPLING_RATE

    # Process and plot each lead
    for lead_idx, lead_name in enumerate(LEAD_NAMES):
        ax = axes[lead_idx]

        # Get original signal for current lead
        original_signal = signal[:, lead_idx]

        # Process lead to get baseline removed signal
        filtered_signal = remove_baseline_wander_hp_filter(
            original_signal, SAMPLING_RATE, cutoff=0.5
        )

        # Plot baseline removed signal
        ax.plot(
            time_axis,
            filtered_signal,
            "b-",
            linewidth=0.8,
            label="Baseline Removed",
        )

        # Add markers at R-peak positions
        ax.plot(
            r_peaks_time,
            filtered_signal[r_peaks_II],
            "ro",
            markersize=4,
            alpha=0.7,
            label="R-peaks",
        )

        # Set title color to red if this lead's R-peaks were used
        title_color = "red" if lead_idx in used_r_peak_leads else "black"

        # Set title and labels
        ax.set_title(
            f"Lead {lead_name} (Index: {signal_index})",
            fontsize=12,
            fontweight="bold",
            color=title_color,
        )
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude (mV)")
        ax.grid(True, alpha=0.3)

        # Show legend only in first subplot
        if lead_idx == 0:
            ax.legend(loc="upper right", fontsize="small")

        # Set x-axis range to show first 10 seconds or entire signal if shorter
        ax.set_xlim(0, min(10, time_axis[-1]))

    # Adjust subplot spacing
    plt.subplots_adjust(
        left=0.052, bottom=0.052, right=0.971, top=0.914, wspace=0.22, hspace=0.488
    )

    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{signal_index}_baseline_removed_signal.png"))
    plt.close()


def plot_baseline_removal_overlay(X, signal_index, used_r_peak_leads=None):
    """
    Plot the overlay of original signal, estimated baseline, and baseline removal for all 12 leads

    Parameters:
        X: Original ECG data
        signal_index: Signal index
        used_r_peak_leads: List of lead indices whose R-peaks were used
    """
    if used_r_peak_leads is None:
        used_r_peak_leads = []

    # Create 3x4 subplot layout
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    # Get signal data
    signal = X[0, :, :]

    # Create time axis
    time_axis = np.arange(len(signal)) / SAMPLING_RATE

    # Process and plot each lead
    for lead_idx, lead_name in enumerate(LEAD_NAMES):
        ax = axes[lead_idx]

        # Get original signal for current lead
        original_signal = signal[:, lead_idx]

        # Process lead to get baseline and baseline removed signal
        filtered_signal = remove_baseline_wander_hp_filter(
            original_signal, SAMPLING_RATE, cutoff=0.5
        )

        # Estimate baseline
        baseline = original_signal - filtered_signal

        # Plot original signal
        ax.plot(
            time_axis,
            original_signal,
            "b-",
            linewidth=1,
            alpha=0.4,
            label="Original Signal",
        )

        # Plot estimated baseline
        ax.plot(
            time_axis,
            baseline,
            "r-",
            linewidth=2.5,
            alpha=0.4,
            label="Estimated Baseline",
        )

        # Plot baseline removed signal
        ax.plot(
            time_axis,
            filtered_signal,
            "g-",
            linewidth=0.4,
            alpha=0.7,
            label="Baseline Removed",
        )

        # Set title color to red if this lead's R-peaks were used
        title_color = "red" if lead_idx in used_r_peak_leads else "black"

        # Set title and labels
        ax.set_title(
            f"Lead {lead_name} (Index: {signal_index})",
            fontsize=12,
            fontweight="bold",
            color=title_color,
        )
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude (mV)")
        ax.grid(True, alpha=0.3)

        # Show legend only in first subplot
        if lead_idx == 0:
            ax.legend(loc="upper right", fontsize="small")

        # Set x-axis range to show first 10 seconds or entire signal if shorter
        ax.set_xlim(0, min(10, time_axis[-1]))

    # Adjust subplot spacing
    plt.subplots_adjust(
        left=0.052, bottom=0.052, right=0.971, top=0.914, wspace=0.22, hspace=0.488
    )

    # Save figure
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{signal_index}_baseline_removal_overlay.png")
    )
    plt.close()


def process_single_signal(signal_index):
    """Process a single ECG signal - designed for parallel execution"""
    X = load_raw_data(Y, SAMPLING_RATE, PATH, signal_index)

    # Detect R-peaks for all leads
    all_r_peaks = {}
    for lead_idx in range(12):
        lead_signal = X[0, :, lead_idx]
        filtered_signal = remove_baseline_wander_hp_filter(
            lead_signal, SAMPLING_RATE, cutoff=0.5
        )
        r_peaks = _round_and_clip_indices(
            detect_r_peaks_envelope(filtered_signal, SAMPLING_RATE),
            len(filtered_signal),
            filtered_signal,
            "Envelope Filter",
        )
        all_r_peaks[lead_idx] = r_peaks

    # Find unique R-peak sets
    unique_r_peaks_sets = []
    for lead_idx, r_peaks in all_r_peaks.items():
        # Convert to tuple for comparison
        r_peaks_tuple = tuple(r_peaks)
        # Check if this set already exists
        is_duplicate = False
        for existing_set in unique_r_peaks_sets:
            if r_peaks_tuple == existing_set[0]:
                is_duplicate = True
                existing_set[1].append(lead_idx)
                break

        if not is_duplicate:
            unique_r_peaks_sets.append((r_peaks_tuple, [lead_idx]))

    # Calculate total standard deviation for each unique R-peak set
    r_peaks_with_std = []
    for r_peaks_tuple, lead_indices in unique_r_peaks_sets:
        r_peaks = np.array(r_peaks_tuple)
        total_std = calculate_total_std_with_r_peaks(X, r_peaks)
        r_peaks_with_std.append((r_peaks, total_std, lead_indices))

    # Select the R-peak set with minimum total standard deviation
    best_r_peaks, min_std, used_r_peak_leads = min(r_peaks_with_std, key=lambda x: x[1])

    # Plot the baseline removed signal
    plot_baseline_removed_signal(X, signal_index, used_r_peak_leads=used_r_peak_leads)

    # Plot the baseline removal overlay
    plot_baseline_removal_overlay(X, signal_index, used_r_peak_leads=used_r_peak_leads)

    all_leads_normalized = {}

    # Process all leads using the best R-peaks
    for lead_idx in range(12):
        all_leads_normalized[lead_idx] = process_lead_with_r_peaks(
            X[0, :, lead_idx], best_r_peaks, lead_idx
        )

    # Pass the leads whose R-peaks were used to the plotting function
    plot_all_leads_normalized_heartbeats(
        all_leads_normalized,
        SAMPLING_RATE,
        signal_index,
        used_r_peak_leads=used_r_peak_leads,
    )

    return signal_index


def main():
    print("=" * 80)
    print("ECG Signal Processing Workflow for Irregular Heartbeats")
    print("=" * 80)

    current_backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    max_workers = MAX_WORKERS
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for signal_index in irregular:
            futures.append(executor.submit(process_single_signal, signal_index))

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing irregular signals",
            unit="signal",
        ):
            try:
                result = future.result()
            except Exception as e:
                print(f"Error processing signal: {e}")
    matplotlib.use(current_backend)

    print("=" * 80)
    print("ECG Signal Processing Workflow for Irregular Heartbeats Completed")
    print("=" * 80)


if __name__ == "__main__":
    main()
