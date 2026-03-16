import numpy as np
from wfdb import processing as wfdb_processing
import matplotlib.pyplot as plt
from utils._baseline import (
    remove_baseline_wander_hp_filter,
    evaluate_baseline_removal,
    plot_baseline_removal_comparison,
)
from utils._r_peaks import (
    detect_r_peaks_neurokit_NeuroKit2,
    evaluate_r_peak_detection,
)
from utils._config import (
    SAMPLING_RATE,
    TGT_SAMPLING_RATE,
    PLOT_CONFIG,
    LEAD_NAMES,
)
from utils._data import X
from utils._rr_intervals import calculate_rr_intervals
from utils._heartbeats import (
    extract_heartbeats,
    split_and_resample_heartbeats,
    plot_heartbeats_overlay_normalized,
    plot_heartbeats_overlay_original,
    plot_average_heartbeat_with_variance,
    plot_heartbeat_evaluation_all,
)
from utils._helpers import (
    _safe_scale_and_clip,
    _round_and_clip_indices,
)

# Extract II lead (index 1 for II lead, 0 is I)
lead_index = 1
ecg_signal = X[0, :, lead_index]

# Baseline removal using high-pass filter
filtered_hp = remove_baseline_wander_hp_filter(ecg_signal, SAMPLING_RATE, cutoff=0.5)

# Evaluate baseline removal method
baseline_evals = {
    "High-Pass Filter": evaluate_baseline_removal(ecg_signal, filtered_hp),
}

# Resample the signal to tgt_sample_rate Hz
ecg_signal_resampled, _ = wfdb_processing.resample_sig(
    filtered_hp, SAMPLING_RATE, TGT_SAMPLING_RATE
)

# Detect R peaks using basic method
r_peaks_raw = {
    "Default (NeuroKit2)": detect_r_peaks_neurokit_NeuroKit2(
        filtered_hp, SAMPLING_RATE
    ),
}

# Sanitize R peaks
r_peaks = {}
orig_len = len(filtered_hp)
for name, out in r_peaks_raw.items():
    r_peaks[name] = _round_and_clip_indices(
        out, orig_len, sig=filtered_hp, sig_name=name
    )

# Evaluate R peak detection method
peak_evals = {
    method: evaluate_r_peak_detection(peaks, filtered_hp, SAMPLING_RATE)
    for method, peaks in r_peaks.items()
    if len(peaks) > 0
}

# Use the detected R peaks
selected_r_peaks = r_peaks["Default (NeuroKit2)"]

# Calculate intervals between consecutive R-peaks (in samples)
rr_intervals_samples = np.diff(selected_r_peaks)

# Shift each R-peak forward by 0.1a (where a is the interval to the next R-peak)
# For the last R-peak, use the previous interval
shifted_r_peaks = np.zeros_like(selected_r_peaks)
for i in range(len(selected_r_peaks)):
    if i < len(selected_r_peaks) - 1:
        # Use interval to next R-peak
        a = rr_intervals_samples[i]
    else:
        # For last peak, use the previous interval
        a = rr_intervals_samples[-1]

    # Shift forward by 0.1a
    shifted_r_peaks[i] = selected_r_peaks[i] + 0.1 * a

# Ensure shifted peaks are within signal bounds
shifted_r_peaks = np.clip(shifted_r_peaks, 0, len(filtered_hp) - 1)

# Calculate RR intervals using shifted R-peaks
rr_analysis = calculate_rr_intervals(shifted_r_peaks, SAMPLING_RATE)

# Extract heartbeats using shifted R-peaks
heartbeats_II = extract_heartbeats(filtered_hp, shifted_r_peaks, SAMPLING_RATE)

# Plot original overlaid heartbeats
if PLOT_CONFIG["heartbeats_overlay_original"]:
    plot_heartbeats_overlay_original(heartbeats_II, SAMPLING_RATE, "II", max_beats=15)

# Normalize heartbeats
normalized_II, pre_r_samp, post_r_samp, total_samp = split_and_resample_heartbeats(
    heartbeats_II,
    SAMPLING_RATE,
)

# Plot normalized overlaid heartbeats
if PLOT_CONFIG["heartbeats_overlay_normalized"]:
    plot_heartbeats_overlay_normalized(normalized_II, SAMPLING_RATE, "II", max_beats=15)

# Plot individual normalized heartbeats
if PLOT_CONFIG["single_heartbeat_normalized"]:
    plot_average_heartbeat_with_variance(normalized_II, SAMPLING_RATE, "II")

from ecg_heartbeat_normalization_pipeline import (
    process_ecg_signal,
    process_lead_with_r_peaks,
    plot_all_leads_normalized_heartbeats,
)

all_leads_normalized = {}

lead_II_idx = 1
all_leads_normalized[lead_II_idx] = normalized_II

# Process all other leads using Lead II R-peaks
for lead_idx in range(12):
    if lead_idx == lead_II_idx:
        continue
    """Process ECG lead using provided R-peak positions"""
    filtered_hp_ = remove_baseline_wander_hp_filter(
        X[0, :, lead_idx], SAMPLING_RATE, cutoff=0.5
    )
    heartbeats = extract_heartbeats(filtered_hp_, shifted_r_peaks, SAMPLING_RATE)
    all_leads_normalized[lead_idx], pre_r, post_r, total = (
        split_and_resample_heartbeats(heartbeats, SAMPLING_RATE)
    )

plot_all_leads_normalized_heartbeats(all_leads_normalized, SAMPLING_RATE, "_")
