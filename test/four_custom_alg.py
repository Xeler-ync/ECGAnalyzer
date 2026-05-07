import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils._baseline import (
    remove_baseline_wander_hp_filter,
)
from utils._r_peaks import (
    detect_r_peaks_basic,
    detect_r_peaks_adaptive,
    detect_r_peaks_derivative,
    detect_r_peaks_envelope,
    plot_r_peak_detection_comparison,
)
from utils._config import (
    SAMPLING_RATE,
    TGT_SAMPLING_RATE,
    PLOT_CONFIG,
)
from utils._data import X
from utils._helpers import (
    _safe_scale_and_clip,
    _round_and_clip_indices,
)

# Extract II lead (index 1 for II lead, 0 is I)
lead_index = 1
ecg_signal = X[0, :, lead_index]

filtered_hp = remove_baseline_wander_hp_filter(ecg_signal, SAMPLING_RATE, cutoff=0.5)

r_peaks_raw = {
    "Basic": detect_r_peaks_basic(filtered_hp, SAMPLING_RATE),
    "Adaptive": detect_r_peaks_adaptive(filtered_hp, SAMPLING_RATE),
    "Derivative": detect_r_peaks_derivative(filtered_hp, SAMPLING_RATE),
    "Envelope": detect_r_peaks_envelope(filtered_hp, SAMPLING_RATE),
}

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
        2,
    )
