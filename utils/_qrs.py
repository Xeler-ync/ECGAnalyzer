import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

def highpass_filter(signal, fs=100, cutoff=0.5, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal)

def detect_r_peaks(ecg, fs=100):
    # Simple preprocessing: de-mean + absolute value
    x = ecg - np.mean(ecg)
    x = np.abs(x)

    # Adaptive threshold: take the higher quantile
    height = np.percentile(x, 90)
    distance = int(0.3 * fs)  # At least 300ms interval

    peaks, _ = find_peaks(x, height=height, distance=distance)
    return peaks

def split_into_beats(ecg, r_peaks, beat_length=256):
    beats = []
    half = beat_length // 2
    N = len(ecg)

    for r in r_peaks:
        start = r - half
        end = r + half
        if start < 0 or end > N:
            continue
        beats.append(ecg[start:end])
    return np.array(beats)

def detect_pqrst(beat):
    N = len(beat)
    if N < 10:
        return None

    # R: Maximum value
    R = int(np.argmax(beat))

    # Q: The minimum value on the left side of R
    q_start = max(0, R - 40)
    q_end = R
    if q_end - q_start >= 2:
        Q = q_start + int(np.argmin(beat[q_start:q_end]))
    else:
        Q = max(0, R - 1)

    # S: The minimum value on the right side of R
    s_start = R
    s_end = min(N, R + 60)
    if s_end - s_start >= 2:
        S = s_start + int(np.argmin(beat[s_start:s_end]))
    else:
        S = min(N - 1, R + 1)

    # P: Maximum value on the left side of Q (window smaller a bit more stable)
    p_start = max(0, Q - 40)
    p_end = Q
    if p_end - p_start >= 2:
        P = p_start + int(np.argmax(beat[p_start:p_end]))
    else:
        P = max(0, Q - 1)

    # T: The maximum value on the right side of S
    t_start = S
    t_end = min(N, S + 120)
    if t_end - t_start >= 2:
        T = t_start + int(np.argmax(beat[t_start:t_end]))
    else:
        T = min(N - 1, S + 1)

    return P, Q, R, S, T

def extract_features_from_beat(beat, P, Q, R, S, T):
    return {
        # ---Time interval---
        "PR_interval": Q - P,
        "QRS_width": S - Q,
        "QT_interval": T - Q,
        # --- amplitude ---
        "P_amp": beat[P],
        "Q_amp": beat[Q],
        "R_amp": beat[R],
        "S_amp": beat[S],
        "T_amp": beat[T],
        # --- Slope  ---
        "QRS_rise": beat[R] - beat[Q],
        "QRS_fall": beat[R] - beat[S],
        # --- Waveform energy ---
        "QRS_energy": np.sum(beat[Q:S] ** 2),
    }

# Read the PTB-XL database index
df = pd.read_csv("PTB-XL/ptbxl_database.csv")

df['diagnostic_superclass'] = df['scp_codes'].apply(
    lambda x: list(eval(x).keys())[0]
)

# Select a patient
pid = df['patient_id'].iloc[5]
df_patient = df[df['patient_id'] == pid].reset_index(drop=True)

# Take the patient's first record (100 Hz)
row = df_patient.iloc[0]
path = row['filename_lr']

signal, meta = wfdb.rdsamp('PTB-XL/'+path)

# Select Lead II (Index 1)
ecg_raw = signal[:, 1]

fs = 100  # The filename_lr corresponds to a frequency of 100 Hz.

# high-pass filtering
ecg = highpass_filter(ecg_raw, fs=fs)

# R-peak detection
r_peaks = detect_r_peaks(ecg, fs=fs)

# Segmentation of heartbeats
beats = split_into_beats(ecg, r_peaks, beat_length=256)

print("Total number of detected heartbeats:", len(beats))

# Select the first heartbeat
beat = beats[0]

# Detect P/Q/R/S/T
P, Q, R, S, T = detect_pqrst(beat)

feat = extract_features_from_beat(beat, P, Q, R, S, T)
X = [list(feat.values())]
y = [row['diagnostic_superclass']]
# Figure
plt.figure(figsize=(12,4))
plt.plot(beat, label='ECG Beat', linewidth=2)

plt.scatter([P, Q, R, S, T],
            [beat[P], beat[Q], beat[R], beat[S], beat[T]],
            c=['purple', 'blue', 'red', 'blue', 'green'],
            s=80)

plt.text(P, beat[P], 'P', color='purple', fontsize=12)
plt.text(Q, beat[Q], 'Q', color='blue', fontsize=12)
plt.text(R, beat[R], 'R', color='red', fontsize=12)
plt.text(S, beat[S], 'S', color='blue', fontsize=12)
plt.text(T, beat[T], 'T', color='green', fontsize=12)

plt.title(f"Patient {pid} - One Beat with P/Q/R/S/T")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()

