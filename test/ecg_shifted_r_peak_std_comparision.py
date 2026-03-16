import numpy as np
from utils._baseline import (
    remove_baseline_wander_hp_filter,
)
from utils._r_peaks import (
    detect_r_peaks_neurokit_NeuroKit2,
)
from utils._config import (
    SAMPLING_RATE,
    PATH,
    LEAD_NAMES,
)
from utils._data import Y, load_raw_data
from utils._rr_intervals import calculate_rr_intervals, plot_rr_intervals
from utils._heartbeats import (
    extract_heartbeats,
)
from utils._helpers import (
    _round_and_clip_indices,
)

# Extract II lead (index 1 for II lead, 0 is I)
lead_index = 1

min_avg_std_value = 114514
min_avg_std_index = 0

for i in Y.ecg_id[:100]:
    ecg_signal = load_raw_data(Y, SAMPLING_RATE, PATH, i)

    # Baseline removal using high-pass filter
    filtered_hp = remove_baseline_wander_hp_filter(
        ecg_signal[0, :, lead_index], SAMPLING_RATE, cutoff=0.5
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

    avg_std = np.mean(np.std(heartbeats_II[0]['signal'], axis=0))

    if avg_std < min_avg_std_value:
        min_avg_std_value = avg_std
        min_avg_std_index = i

print(f"min_avg_std_index = {min_avg_std_index}")
print(f"min_avg_std_value = {min_avg_std_value}")