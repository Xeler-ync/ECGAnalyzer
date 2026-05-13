import sys
import os.path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures

from utils._baseline import remove_baseline_wander_hp_filter, evaluate_baseline_removal
from utils._r_peaks import detect_r_peaks_neurokit_NeuroKit2, evaluate_r_peak_detection
from utils._bpm import calc_bpm_by_fft, calculate_bpm_from_r_peaks
from utils._config import SAMPLING_RATE, RESULTS_PATH, PATH, MAX_WORKERS, LEAD_NAMES
from utils._data import load_raw_data, Y
from utils._heartbeats import extract_heartbeats, split_and_resample_heartbeats
from utils._helpers import _round_and_clip_indices


def process_ecg_signal(ecg_signal, lead_index, sampling_rate=SAMPLING_RATE):
    """Process single ECG lead: baseline removal -> R-peak detection -> heartbeat extraction and normalization"""
    filtered_signal = remove_baseline_wander_hp_filter(
        ecg_signal, sampling_rate, cutoff=0.5
    )
    baseline_eval = evaluate_baseline_removal(ecg_signal, filtered_signal)

    # R-peak detection
    r_peaks = _round_and_clip_indices(
        detect_r_peaks_neurokit_NeuroKit2(filtered_signal, sampling_rate),
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
    bpm_from_r_peaks=None,
    bpm_from_fft=None,
    is_bpm_diff_significant=False,
):
    """Display normalized heartbeats for all 12 leads in a single figure"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    # Determine title color based on BPM difference
    title_color = "red" if is_bpm_diff_significant else "black"

    # Add BPM comparison text to the figure
    bpm_text = f"BPM (R-peaks): {bpm_from_r_peaks:.1f} | BPM (FFT): {bpm_from_fft:.1f} | Difference: {abs(bpm_from_r_peaks - bpm_from_fft):.1f}"

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

        ax.set_title(
            f"Lead {lead_name} (n={len(normalized_heartbeats)})\nMax Std: {np.max(std_beat):.4f} | Avg Std: {np.mean(std_beat):.4f}"
        )
        ax.set_xlabel("Sample Points")
        ax.set_ylabel("Amplitude (mV)")
        ax.grid(True, alpha=0.3)
        if lead_idx == 0:
            ax.legend(loc="upper right", fontsize="small")

    # Add BPM comparison as figure title
    fig.suptitle(bpm_text, fontsize=14, fontweight="bold", color=title_color)

    plt.subplots_adjust(
        left=0.052, bottom=0.052, right=0.971, top=0.914, wspace=0.22, hspace=0.488
    )
    plt.savefig(
        os.path.join(
            RESULTS_PATH, f"{signal_index}_all_leads_normalized_heartbeats.png"
        )
    )
    plt.close()


def process_single_signal(signal_index):
    """Process a single ECG signal - designed for parallel execution"""
    X = load_raw_data(Y, SAMPLING_RATE, PATH, signal_index)
    # Process Lead II first to get R-peaks
    lead_II_idx = 1
    results_II = process_ecg_signal(X[0, :, lead_II_idx], lead_II_idx)
    all_leads_normalized = {lead_II_idx: results_II["normalized_heartbeats"]}
    r_peaks_II = results_II["r_peaks"]

    # Calculate BPM from R-peaks
    bpm_from_r_peaks = calculate_bpm_from_r_peaks(r_peaks_II)

    # Calculate BPM using FFT method
    # Prepare filtered signal for all leads
    filtered_ecg_signal = np.zeros((X.shape[1], 12))
    for lead_idx in range(12):
        lead_signal = X[0, :, lead_idx]
        filtered_signal = remove_baseline_wander_hp_filter(
            lead_signal, SAMPLING_RATE, cutoff=0.5
        )
        filtered_ecg_signal[:, lead_idx] = filtered_signal

    # Calculate BPM using FFT
    bpm_from_fft = calc_bpm_by_fft(filtered_ecg_signal)

    # Determine if BPM difference is significant (threshold: 0.1 * bpm_from_fft)
    bpm_difference = abs(bpm_from_r_peaks - bpm_from_fft)
    is_bpm_diff_significant = bpm_difference > bpm_from_fft * 0.1

    # Process all other leads using Lead II R-peaks
    for lead_idx in range(12):
        if lead_idx == lead_II_idx:
            continue
        all_leads_normalized[lead_idx] = process_lead_with_r_peaks(
            X[0, :, lead_idx], r_peaks_II, lead_idx
        )

    plot_all_leads_normalized_heartbeats(
        all_leads_normalized,
        SAMPLING_RATE,
        signal_index,
        bpm_from_r_peaks=bpm_from_r_peaks,
        bpm_from_fft=bpm_from_fft,
        is_bpm_diff_significant=is_bpm_diff_significant,
    )

    return signal_index


def main():
    print(f"{"=" * 80}\nECG Signal Processing Workflow\n{"=" * 80}")

    current_backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    max_workers = MAX_WORKERS
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        # for signal_index in range(Y.patient_id.count()):
        futures.extend(
            executor.submit(process_single_signal, signal_index)
            for signal_index in range(20)
        )
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing signals",
            unit="signal",
        ):
            try:
                result = future.result()
            except Exception as e:
                print(f"Error processing signal: {e}")
    matplotlib.use(current_backend)

    print(f"{"=" * 80}\nECG Signal Processing Workflow Completed\n{"=" * 80}")


if __name__ == "__main__":
    main()
