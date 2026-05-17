import sys
import os.path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import joblib
import numpy as np

from tools.irregular_heartbeat_visualization import calculate_total_std_with_r_peaks
from tools.prepare_data import extract_dynamic_heartbeats
from utils._baseline import remove_baseline_wander_hp_filter
from utils._data import load_raw_data, Y, SAMPLING_RATE, PATH, ECG_INDEX
from utils._r_peaks import detect_r_peaks_neurokit_NeuroKit2, detect_r_peaks_envelope

parser = argparse.ArgumentParser(description="ECG Prediction Script")
parser.add_argument(
    "--signal_index",
    type=int,
    default=ECG_INDEX,
    help="Index of the ECG signal to load",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="./results/model/lvh_norm_rf/model.joblib",
    help="Path to the trained model file",
)
args = parser.parse_args()

signal_index = args.signal_index
MODEL_PATH = args.model_path

rf_model = joblib.load(MODEL_PATH)
print(f"[INFO] Model loaded from {MODEL_PATH}")

X = load_raw_data(Y, SAMPLING_RATE, PATH, signal_index)  # shape: (1, time, leads)
print(
    f"[INFO] Raw ECG data shape: {X.shape} (1 record, {X.shape[1]} samples, {X.shape[2]} leads)"
)


filtered_signals = []
for lead_idx in range(12):
    raw_lead = X[0, :, lead_idx]
    filtered = remove_baseline_wander_hp_filter(raw_lead, SAMPLING_RATE, cutoff=0.5)
    filtered_signals.append(filtered)
print("[INFO] Baseline wander removed from all 12 leads (HP filter cutoff = 0.5 Hz)")

r_peak_candidates = []  # each element is a list of peak indices
method_names = []

for lead_idx, lead_signal in enumerate(filtered_signals):
    # Method 1: NeuroKit2
    peaks_nk = detect_r_peaks_neurokit_NeuroKit2(lead_signal, SAMPLING_RATE)
    r_peak_candidates.append(peaks_nk)
    method_names.append(f"Lead{lead_idx+1}_NeuroKit")
    # Method 2: Envelope
    peaks_env = detect_r_peaks_envelope(lead_signal, SAMPLING_RATE)
    r_peak_candidates.append(peaks_env)
    method_names.append(f"Lead{lead_idx+1}_Envelope")

print(
    f"[INFO] Generated {len(r_peak_candidates)} R‑peak candidate sets (2 methods × 12 leads)"
)

best_r_peaks = None
best_method = None
min_total_std = float("inf")

for i, r_peaks in enumerate(r_peak_candidates):
    if len(r_peaks) == 0:
        print(
            f"[DEBUG] Candidate {i} ({method_names[i]}) – no peaks detected, skipping"
        )
        continue
    total_std = calculate_total_std_with_r_peaks(X, r_peaks, SAMPLING_RATE)
    print(
        f"[DEBUG] Candidate {i} ({method_names[i]}) – peaks: {len(r_peaks)}, total STD: {total_std:.4f}"
    )
    if total_std < min_total_std:
        min_total_std = total_std
        best_r_peaks = r_peaks
        best_method = method_names[i]

if best_r_peaks is None:
    print("[ERROR] No valid R‑peaks detected in any candidate set. Exiting.")
    exit()

print(
    f"[INFO] Selected best R‑peak set from '{best_method}' – peaks: {len(best_r_peaks)}, total STD: {min_total_std:.4f}"
)

lead_templates = []
target_len = 65

for lead_idx, lead_signal in enumerate(filtered_signals):
    heartbeats = extract_dynamic_heartbeats(
        lead_signal, best_r_peaks, target_len=target_len
    )
    if heartbeats.size == 0:
        print(
            f"[WARNING] Lead {lead_idx+1}: no valid heartbeats extracted (check R‑peak alignment)"
        )
        break
    mean_beat = np.mean(heartbeats, axis=0)
    lead_templates.append(mean_beat)
    print(
        f"[INFO] Lead {lead_idx+1}: extracted {heartbeats.shape[0]} beats, mean template shape {mean_beat.shape}"
    )

if len(lead_templates) == 12:
    # Stack leads as columns: shape (target_len, 12), then flatten to 1D vector
    X_sample = np.stack(lead_templates, axis=1).reshape(1, -1)
    print(
        f"[INFO] Feature vector ready: {X_sample.shape[1]} features (65 samples × 12 leads)"
    )

    prediction = rf_model.predict(X_sample)
    print(f"[RESULT] Prediction: {prediction}")

    # Optional: show prediction probability if model supports it
    if hasattr(rf_model, "predict_proba"):
        proba = rf_model.predict_proba(X_sample)
        print(f"[RESULT] Prediction probabilities: {proba}")
else:
    print(
        f"[ERROR] Incomplete lead templates – only {len(lead_templates)} out of 12 leads available. Cannot predict."
    )
