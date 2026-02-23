import numpy as np
from wfdb import processing as wfdb_processing
import matplotlib.pyplot as plt
from utils._baseline import (
    remove_baseline_wander_hp_filter,
    remove_baseline_wander_savgol,
    remove_baseline_wander_morphological,
    plot_fft_and_baseline_analysis,
    evaluate_baseline_removal,
    plot_baseline_removal_comparison,
    plot_fft_and_baseline_analysis,
)
from utils._r_peaks import (
    detect_r_peaks_basic,
    detect_r_peaks_adaptive,
    detect_r_peaks_derivative,
    detect_r_peaks_hamilton_ECG_Detectors,
    detect_r_peaks_christov_ECG_Detectors,
    detect_r_peaks_engelese_kulp_ECG_Detectors,
    detect_r_peaks_pan_tompkins_ECG_Detectors,
    detect_r_peaks_swt_ECG_Detectors,
    detect_r_peaks_matched_filter_ECG_Detectors,
    detect_r_peaks_wqrs_ECG_Detectors,
    detect_r_peaks_two_moving_average_ECG_Detectors,
    detect_r_peaks_pantompkins1985_NeuroKit2,
    detect_r_peaks_hamilton2002_NeuroKit2,
    detect_r_peaks_christov2004_NeuroKit2,
    detect_r_peaks_engzeemod2012_NeuroKit2,
    detect_r_peaks_elgendi2010_NeuroKit2,
    detect_r_peaks_zong2003_NeuroKit2,
    detect_r_peaks_martinez2004_NeuroKit2,
    detect_r_peaks_kalidas2017_NeuroKit2,
    detect_r_peaks_khamis2016_NeuroKit2,
    detect_r_peaks_manikandan2012_NeuroKit2,
    detect_r_peaks_nabian2018_NeuroKit2,
    detect_r_peaks_rodrigues2020_NeuroKit2,
    detect_r_peaks_emrich2023_NeuroKit2,
    detect_r_peaks_neurokit_NeuroKit2,
    detect_r_peaks_gamboa2008_NeuroKit2,
    detect_r_peaks_promac_NeuroKit2,
    detect_r_peaks_asi_NeuroKit2,
    evaluate_r_peak_detection,
    plot_r_peak_detection_comparison,
)
from utils._config import (
    SAMPLING_RATE,
    TGT_SAMPLING_RATE,
    PLOT_CONFIG,
)
from utils._data import X
from utils._rr_intervals import calculate_rr_intervals, plot_rr_intervals
from utils._heartbeats import (
    extract_heartbeats,
    split_and_resample_heartbeats,
    plot_heartbeats_overlay_normalized,
    plot_heartbeats_overlay_original,
    plot_single_heartbeat_normalized,
)
from utils._leads import (
    plot_multiple_leads_normalized,
    plot_original_vs_normalized_multiple_leads,
)
from utils._helpers import (
    _safe_scale_and_clip,
    _round_and_clip_indices,
)


def plot_evaluation_comparison(baseline_evals, peak_evals):
    """Plot evaluation results comparison"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    methods_baseline = list(baseline_evals.keys())

    variance_reductions = [
        baseline_evals[m]["variance_reduction"] for m in methods_baseline
    ]
    axes[0, 0].bar(
        methods_baseline, variance_reductions, color=["green", "red", "purple"]
    )
    axes[0, 0].set_title("Variance Reduction (%)", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Variance Reduction (%)")
    axes[0, 0].tick_params(axis="x", rotation=90)
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    rms_errors = [baseline_evals[m]["rms_error"] for m in methods_baseline]
    axes[0, 1].bar(methods_baseline, rms_errors, color=["green", "red", "purple"])
    axes[0, 1].set_title("RMS Error", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("RMS Error")
    axes[0, 1].tick_params(axis="x", rotation=90)
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    smoothness_vals = [baseline_evals[m]["smoothness"] for m in methods_baseline]
    axes[0, 2].bar(methods_baseline, smoothness_vals, color=["green", "red", "purple"])
    axes[0, 2].set_title("Smoothness (Lower is Better)", fontsize=12, fontweight="bold")
    axes[0, 2].set_ylabel("Average Derivative Magnitude")
    axes[0, 2].tick_params(axis="x", rotation=90)
    axes[0, 2].grid(True, alpha=0.3, axis="y")

    methods_peaks = list(peak_evals.keys())

    peak_counts = [peak_evals[m]["peak_count"] for m in methods_peaks]
    axes[1, 0].bar(methods_peaks, peak_counts, color=["red", "magenta", "cyan"])
    axes[1, 0].set_title("R Peak Count", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Number of R Peaks")
    axes[1, 0].tick_params(axis="x", rotation=90)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    heart_rates = [peak_evals[m]["heart_rate_bpm"] for m in methods_peaks]
    axes[1, 1].bar(methods_peaks, heart_rates, color=["red", "magenta", "cyan"])
    axes[1, 1].set_title("Estimated Heart Rate (BPM)", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Heart Rate (BPM)")
    axes[1, 1].tick_params(axis="x", rotation=90)
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    reliabilities = [peak_evals[m]["detection_reliability"] for m in methods_peaks]
    axes[1, 2].bar(methods_peaks, reliabilities, color=["red", "magenta", "cyan"])
    axes[1, 2].set_title("Detection Reliability", fontsize=12, fontweight="bold")
    axes[1, 2].set_ylabel("Reliability Ratio")
    axes[1, 2].tick_params(axis="x", rotation=90)
    axes[1, 2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()


# Extract II lead (index 1 for II lead, 0 is I)
lead_index = 1
ecg_signal = X[0, :, lead_index]


# Perform FFT analysis before any filtering
print(f"\n{'='*80}")
print(f"FFT-BASED BASELINE ANALYSIS")
print(f"{'='*80}")

if PLOT_CONFIG["fft_and_baseline_analysis"]:
    fft_analysis, fft_detrended = plot_fft_and_baseline_analysis(
        ecg_signal, ecg_signal, SAMPLING_RATE, lead_index + 1
    )

# Baseline removal using different methods
filtered_hp = remove_baseline_wander_hp_filter(ecg_signal, SAMPLING_RATE, cutoff=0.5)
filtered_savgol = remove_baseline_wander_savgol(
    ecg_signal, window_length=201, polyorder=3
)
filtered_morpho = remove_baseline_wander_morphological(ecg_signal, kernel_size=51)

# Plot baseline removal comparison
if PLOT_CONFIG["baseline_removal_comparison"]:
    plot_baseline_removal_comparison(
        ecg_signal,
        filtered_hp,
        filtered_savgol,
        filtered_morpho,
        SAMPLING_RATE,
        lead_index + 1,
    )

# Evaluate baseline removal methods
baseline_evals = {
    "High-Pass Filter": evaluate_baseline_removal(ecg_signal, filtered_hp),
    "Savitzky-Golay": evaluate_baseline_removal(ecg_signal, filtered_savgol),
    "Morphological": evaluate_baseline_removal(ecg_signal, filtered_morpho),
}

# resample the signal to tgt_sample_rate Hz by using wfdb.processing.resample_sig
ecg_signal_resampled, _ = wfdb_processing.resample_sig(
    filtered_hp, SAMPLING_RATE, TGT_SAMPLING_RATE
)

# Detect R peaks using different methods on high-pass filtered signal (sanitize all outputs)
r_peaks_raw = {
    "Basic": detect_r_peaks_basic(filtered_hp, SAMPLING_RATE),
    "Adaptive": detect_r_peaks_adaptive(filtered_hp, SAMPLING_RATE),
    "Derivative": detect_r_peaks_derivative(filtered_hp, SAMPLING_RATE),
    # ecgdetectors run on resampled signal then scale back to original sampling_rate
    "Hamilton (ECG-Detectors)": detect_r_peaks_hamilton_ECG_Detectors(
        ecg_signal_resampled, TGT_SAMPLING_RATE
    ),
    "Christov (ECG-Detectors)": detect_r_peaks_christov_ECG_Detectors(
        ecg_signal_resampled, TGT_SAMPLING_RATE
    ),
    "Engelse-Kulp (ECG-Detectors)": detect_r_peaks_engelese_kulp_ECG_Detectors(
        ecg_signal_resampled, TGT_SAMPLING_RATE
    ),
    "Pan-Tompkins (ECG-Detectors)": detect_r_peaks_pan_tompkins_ECG_Detectors(
        ecg_signal_resampled, TGT_SAMPLING_RATE
    ),
    "SWT (ECG-Detectors)": detect_r_peaks_swt_ECG_Detectors(
        ecg_signal_resampled, TGT_SAMPLING_RATE
    ),
    "Matched Filter (ECG-Detectors)": detect_r_peaks_matched_filter_ECG_Detectors(
        ecg_signal_resampled, TGT_SAMPLING_RATE
    ),
    "WQRS (ECG-Detectors)": detect_r_peaks_wqrs_ECG_Detectors(
        ecg_signal_resampled, TGT_SAMPLING_RATE
    ),
    "Two Moving Average (ECG-Detectors)": detect_r_peaks_two_moving_average_ECG_Detectors(
        ecg_signal_resampled, TGT_SAMPLING_RATE
    ),
    # NeuroKit2 variants (run on original filtered_hp, at sampling_rate)
    "Pan-Tompkins (NeuroKit2)": detect_r_peaks_pantompkins1985_NeuroKit2(
        filtered_hp, SAMPLING_RATE
    ),
    "Hamilton (NeuroKit2)": detect_r_peaks_hamilton2002_NeuroKit2(
        filtered_hp, SAMPLING_RATE
    ),
    "Christov (NeuroKit2)": detect_r_peaks_christov2004_NeuroKit2(
        filtered_hp, SAMPLING_RATE
    ),
    "Engelse-Kulp (NeuroKit2)": detect_r_peaks_engzeemod2012_NeuroKit2(
        ecg_signal_resampled, TGT_SAMPLING_RATE
    ),
    "Elgendi (NeuroKit2)": detect_r_peaks_elgendi2010_NeuroKit2(
        filtered_hp, SAMPLING_RATE
    ),
    "Zong (NeuroKit2)": detect_r_peaks_zong2003_NeuroKit2(filtered_hp, SAMPLING_RATE),
    "Martinez (NeuroKit2)": detect_r_peaks_martinez2004_NeuroKit2(
        filtered_hp, SAMPLING_RATE
    ),
    "Kalidas (NeuroKit2)": detect_r_peaks_kalidas2017_NeuroKit2(
        filtered_hp, SAMPLING_RATE
    ),
    "Khamis (NeuroKit2)": detect_r_peaks_khamis2016_NeuroKit2(
        filtered_hp, SAMPLING_RATE
    ),
    "Manikandan (NeuroKit2)": detect_r_peaks_manikandan2012_NeuroKit2(
        filtered_hp, SAMPLING_RATE
    ),
    "Nabian (NeuroKit2)": detect_r_peaks_nabian2018_NeuroKit2(
        filtered_hp, SAMPLING_RATE
    ),
    "Rodrigues (NeuroKit2)": detect_r_peaks_rodrigues2020_NeuroKit2(
        filtered_hp, SAMPLING_RATE
    ),
    "Emrich (NeuroKit2)": detect_r_peaks_emrich2023_NeuroKit2(
        filtered_hp, SAMPLING_RATE
    ),
    "Default (NeuroKit2)": detect_r_peaks_neurokit_NeuroKit2(filtered_hp, SAMPLING_RATE),
    "Gamboa (NeuroKit2)": detect_r_peaks_gamboa2008_NeuroKit2(
        filtered_hp, SAMPLING_RATE
    ),
    "Promac (NeuroKit2)": detect_r_peaks_promac_NeuroKit2(filtered_hp, SAMPLING_RATE),
    "ASI (NeuroKit2)": detect_r_peaks_asi_NeuroKit2(filtered_hp, SAMPLING_RATE),
}

# sanitize and, where needed, scale ecgdetectors outputs back to original signal length
r_peaks = {}
orig_len = len(filtered_hp)
for name, out in r_peaks_raw.items():
    if "ECG-Detectors" in name:
        # out produced on resampled signal length -> scale back to original sampling_rate domain
        r_peaks[name] = _safe_scale_and_clip(
            out, from_sr=TGT_SAMPLING_RATE, to_sr=SAMPLING_RATE, L=orig_len
        )
    else:
        r_peaks[name] = _round_and_clip_indices(
            out, orig_len, sig=filtered_hp, sig_name=name
        )

# Plot R peak detection comparison
if PLOT_CONFIG["r_peak_detection_comparison"]:
    plot_r_peak_detection_comparison(
        filtered_hp,
        r_peaks,
        SAMPLING_RATE,
        lead_index + 1,
    )

# Evaluate R peak detection methods
peak_evals = {
    method: evaluate_r_peak_detection(peaks, filtered_hp, SAMPLING_RATE)
    for method, peaks in r_peaks.items()
    if len(peaks) > 0
}

# Plot evaluation comparison
if PLOT_CONFIG["evaluation_comparison"]:
    plot_evaluation_comparison(baseline_evals, peak_evals)

# Use the best detection method (Basic Method as example)
selected_r_peaks = r_peaks["Default (NeuroKit2)"]

# Calculate RR intervals
rr_analysis = calculate_rr_intervals(selected_r_peaks, SAMPLING_RATE)
if PLOT_CONFIG["rr_intervals"]:
    plot_rr_intervals(selected_r_peaks, SAMPLING_RATE)

# Extract heartbeats from the single patient
heartbeats_II = extract_heartbeats(filtered_hp, selected_r_peaks, SAMPLING_RATE)

# Display heartbeat information
print(f"\n{'='*80}")
print(f"ORIGINAL HEARTBEAT EXTRACTION RESULTS")
print(f"{'='*80}")

if len(heartbeats_II) > 0:
    print(f"\nLead II: {len(heartbeats_II)} heartbeats extracted")
    print(f"  First heartbeat duration: {heartbeats_II[0]['duration_ms']:.2f} ms")
    print(f"  Last heartbeat duration: {heartbeats_II[-1]['duration_ms']:.2f} ms")

    avg_duration = np.mean([hb["duration_ms"] for hb in heartbeats_II])
    std_duration = np.std([hb["duration_ms"] for hb in heartbeats_II])
    print(f"  Mean duration: {avg_duration:.2f} ± {std_duration:.2f} ms")

# Plot original overlaid heartbeats
if PLOT_CONFIG["heartbeats_overlay_original"]:
    plot_heartbeats_overlay_original(heartbeats_II, SAMPLING_RATE, "II", max_beats=15)

print(f"\n{'='*80}")
print(f"NORMALIZED HEARTBEAT EXTRACTION")
print(f"{'='*80}")

# Normalize heartbeats
normalized_II, pre_r_samp, post_r_samp, total_samp = split_and_resample_heartbeats(
    heartbeats_II,
    SAMPLING_RATE,
)

if len(normalized_II) > 0:
    print(f"\nLead II:")
    print(f"  Normalized heartbeats: {len(normalized_II)}")
    print(f"  Pre-R samples: {pre_r_samp}")
    print(f"  Post-R samples: {post_r_samp}")
    print(f"  Total samples per heartbeat: {total_samp}")
    print(f"  Normalized duration: {total_samp / SAMPLING_RATE * 1000:.2f} ms")

# Plot normalized overlaid heartbeats
if PLOT_CONFIG["heartbeats_overlay_normalized"]:
    plot_heartbeats_overlay_normalized(normalized_II, SAMPLING_RATE, "II", max_beats=15)

# Plot individual normalized heartbeats
if PLOT_CONFIG["single_heartbeat_normalized"]:
    if len(normalized_II) > 5:
        plot_single_heartbeat_normalized(normalized_II[5], SAMPLING_RATE, "II", 6)
    if len(normalized_II) > 10:
        plot_single_heartbeat_normalized(normalized_II[10], SAMPLING_RATE, "II", 11)

# Print comprehensive evaluation results
print(f"\n{'='*80}")
print(f"ECG SIGNAL PROCESSING EVALUATION RESULTS (Lead V{lead_index + 1})")
print(f"{'='*80}")

print(f"\n--- BASELINE REMOVAL EVALUATION ---")
for method, metrics in baseline_evals.items():
    print(f"\n{method}:")
    print(f"  Original noise std:      {metrics['noise_std_original']:.4f}")
    print(f"  Filtered noise std:      {metrics['noise_std_filtered']:.4f}")
    print(f"  RMS Error:               {metrics['rms_error']:.4f}")
    print(f"  Smoothness:              {metrics['smoothness']:.4f}")
    print(f"  Variance Reduction:      {metrics['variance_reduction']:.2f}%")

print(f"\n--- R PEAK DETECTION EVALUATION ---")
for method, metrics in peak_evals.items():
    print(f"\n{method}:")
    print(f"  Peak Count:              {metrics['peak_count']}")
    print(f"  Avg Peak Distance:       {metrics['avg_peak_distance_sec']:.4f} sec")
    print(f"  Estimated Heart Rate:    {metrics['heart_rate_bpm']:.2f} BPM")
    print(f"  Peak Regularity:         {metrics['peak_regularity_percent']:.2f}%")
    print(f"  Detection Reliability:   {metrics['detection_reliability']:.4f}")
    print(f"  Avg Peak Amplitude:      {metrics['avg_peak_amplitude']:.4f} mV")

print(f"\n--- RR INTERVAL ANALYSIS ---")
print(f"Mean RR Interval:       {rr_analysis['mean_rr_ms']:.2f} ms")
print(
    f"RR Interval Range:      {rr_analysis['rr_min_ms']:.2f} - {rr_analysis['rr_max_ms']:.2f} ms"
)
print(f"Mean Heart Rate:        {rr_analysis['heart_rate_bpm']:.2f} BPM")
print(f"Heart Rate Std Dev:     {rr_analysis['heart_rate_std']:.2f} BPM")

print(f"\n{'='*80}")

# Use R peaks from II lead to extract heartbeats from all leads
print(f"\n{'='*80}")
print(f"EXTRACTING HEARTBEATS FROM ALL LEADS USING II R PEAK DETECTION")
print(f"{'='*80}")

# Define which leads to process (0-11 for all 12 leads)
# Lead indices: 0=I, 1=II, 2=III, 3=aVR, 4=aVL, 5=aVF, 6=V1, 7=II, 8=V3, 9=V4, 10=V5, 11=V6
lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "II", "V3", "V4", "V5", "V6"]
leads_to_process = [0, 6, 7, 8, 9, 10, 11]  # I, V1, II, V3, V4, V5, V6

# Dictionary to store results for all leads
all_leads_data = {}
all_leads_normalized = {}

# Filter baseline for all leads first
filtered_signals = {}
for lead_idx in leads_to_process:
    filtered_signals[lead_idx] = remove_baseline_wander_hp_filter(
        X[0, :, lead_idx], SAMPLING_RATE, cutoff=0.5
    )

# Extract and normalize heartbeats for all leads using II's R peaks
for lead_idx in leads_to_process:
    lead_name = lead_names[lead_idx]

    # Extract heartbeats using II's detected R peaks
    heartbeats = extract_heartbeats(
        filtered_signals[lead_idx], selected_r_peaks, SAMPLING_RATE
    )

    # Normalize heartbeats
    if len(heartbeats) > 0:
        normalized_hbs, _, _, _ = split_and_resample_heartbeats(
            heartbeats,
            SAMPLING_RATE,
        )

        all_leads_data[lead_idx] = heartbeats
        all_leads_normalized[lead_idx] = normalized_hbs

        print(
            f"\nLead {lead_name}: {len(normalized_hbs)} normalized heartbeats extracted"
        )
    else:
        print(f"\nLead {lead_name}: No heartbeats extracted")


# Plot V1, II, V3 together
if PLOT_CONFIG["multiple_leads_normalized(V1,II,V3)"]:
    plot_multiple_leads_normalized(
        all_leads_normalized,
        [6, 7, 8],  # V1, II, V3
        lead_names,
        SAMPLING_RATE,
        max_beats=15,
    )

# Plot all standard 12 leads (if available)
available_leads = [idx for idx in leads_to_process if idx in all_leads_normalized]
if PLOT_CONFIG["multiple_leads_normalized(ALL)"]:
    plot_multiple_leads_normalized(
        all_leads_normalized, available_leads, lead_names, SAMPLING_RATE, max_beats=10
    )


# Plot original vs normalized for precordial leads
if PLOT_CONFIG["original_vs_normalized_multiple_leads"]:
    plot_original_vs_normalized_multiple_leads(
        all_leads_data,
        all_leads_normalized,
        [6, 7, 8, 9],  # V1, II, V3, V4
        lead_names,
        SAMPLING_RATE,
        max_beats=8,
    )

print(f"\n{'='*80}")
print(f"MULTI-LEAD HEARTBEAT EXTRACTION COMPLETE")
print(f"{'='*80}")
