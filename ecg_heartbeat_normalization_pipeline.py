import os.path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils._baseline import remove_baseline_wander_hp_filter, evaluate_baseline_removal
from utils._r_peaks import detect_r_peaks_neurokit_NeuroKit2, evaluate_r_peak_detection
from utils._config import SAMPLING_RATE, RESULTS_PATH, PATH
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
    all_leads_normalized, sampling_rate, signal_index, max_beats=15
):
    """Display normalized heartbeats for all 12 leads in a single figure"""
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
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for lead_idx, lead_name in enumerate(lead_names):
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
            label="±1 SD",
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

    plt.subplots_adjust(
        left=0.052, bottom=0.052, right=0.971, top=0.914, wspace=0.22, hspace=0.488
    )
    plt.savefig(
        os.path.join(
            RESULTS_PATH, f"{signal_index}_all_leads_normalized_heartbeats.png"
        )
    )


def main():
    """Main workflow: Process all signals using Lead II R-peaks for all leads"""
    print("=" * 80)
    print("ECG Signal Processing Workflow")
    print("=" * 80)

    for signal_index in tqdm(
        range(Y.patient_id.count()), desc="Processing signals", unit="signal"
    ):
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

        plot_all_leads_normalized_heartbeats(
            all_leads_normalized, SAMPLING_RATE, signal_index
        )

    print("=" * 80)
    print("ECG Signal Processing Workflow Completed")
    print("=" * 80)


if __name__ == "__main__":
    main()
