import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from utils._config import TGT_SAMPLING_RATE, DET

MIN_DISTANCE_SECOND = 0.3  # 200bpm


def detect_r_peaks_basic(ecg_signal, sampling_rate, height=None):
    """Detect R peaks using basic find_peaks method"""
    if height is None:
        height = np.max(ecg_signal) * 0.5
    peaks, _ = find_peaks(
        ecg_signal, height=height, distance=int(sampling_rate * MIN_DISTANCE_SECOND)
    )
    return peaks


def detect_r_peaks_adaptive(ecg_signal, sampling_rate):
    """Detect R peaks using adaptive threshold"""
    window_size = int(sampling_rate * 0.5)
    threshold = np.zeros_like(ecg_signal)

    for i in range(len(ecg_signal)):
        start = max(0, i - window_size // 2)
        end = min(len(ecg_signal), i + window_size // 2)
        threshold[i] = np.mean(ecg_signal[start:end]) + np.std(ecg_signal[start:end])

    peaks, _ = find_peaks(
        ecg_signal, height=threshold, distance=int(sampling_rate * MIN_DISTANCE_SECOND)
    )
    return peaks


def detect_r_peaks_derivative(ecg_signal, sampling_rate):
    """
    Detect R peaks using a second-derivative (curvature) method.
    Second derivative highlights rapid curvature changes at R peaks.
    """
    # first derivative
    d1 = np.diff(ecg_signal)
    d1 = np.insert(d1, 0, 0.0)

    # second derivative (same length as original after padding)
    d2 = np.diff(d1)
    d2 = np.insert(d2, 0, 0.0)

    # use absolute curvature for peak detection
    d2_abs = np.abs(d2)

    # threshold and minimum distance (keep similar spacing constraint)
    height = np.max(d2_abs) * 0.5 if np.max(d2_abs) > 0 else 0.0
    peaks, _ = find_peaks(
        d2_abs, height=height, distance=int(sampling_rate * MIN_DISTANCE_SECOND)
    )
    return peaks


def detect_r_peaks_envelope(signal, fs):
    """
    Peak detection based on Hilbert envelope, unaffected by signal polarity.

    Parameters:
        signal: Input 1D signal (band-pass filtered signal recommended)
        fs: Sampling rate (Hz)

    Returns:
        peaks: Indices of peak positions (corresponding to heartbeat events)
    """
    n = len(signal)
    # Hilbert transform (implemented via FFT)
    fft = np.fft.fft(signal)
    h = np.zeros(n, dtype=complex)
    h[0] = 1
    if n % 2 == 0:
        h[1 : n // 2] = 2
        h[n // 2] = 1
    else:
        h[1 : (n + 1) // 2] = 2
    analytic = np.fft.ifft(fft * h)

    envelope = np.abs(analytic)  # Envelope signal

    # Detect peaks on the envelope
    min_distance = int(MIN_DISTANCE_SECOND * fs)  # Minimum interval
    height = 0.5 * np.max(envelope)  # Threshold
    peaks, _ = find_peaks(envelope, distance=min_distance, height=height)

    return peaks


def detect_r_peaks_hamilton_ECG_Detectors(ecg_signal, sampling_rate):
    """Detect R peaks using Hamilton algorithm from py-ecg-detectors (ecgdetectors)."""
    peaks = DET.hamilton_detector(ecg_signal)
    return np.array(peaks, dtype=int)


def detect_r_peaks_christov_ECG_Detectors(ecg_signal, sampling_rate):
    """Detect R peaks using Christov algorithm from py-ecg-detectors (ecgdetectors)."""
    peaks = DET.christov_detector(ecg_signal)
    return np.array(peaks, dtype=int)


def detect_r_peaks_engelese_kulp_ECG_Detectors(ecg_signal, sampling_rate):
    """Detect R peaks using Engelse-Kulp algorithm"""
    if sampling_rate < TGT_SAMPLING_RATE:
        print(
            f"Warning: Engelse-Kulp requires sampling_rate >= tgt_sample_rate Hz (current: {sampling_rate} Hz)"
        )
        print("Skipping Engelse-Kulp detector")
        return np.array([], dtype=int)

    peaks = DET.engzee_detector(ecg_signal)
    return np.array(peaks, dtype=int)


def detect_r_peaks_pan_tompkins_ECG_Detectors(ecg_signal, sampling_rate):
    """Detect R peaks using Pan-Tompkins algorithm"""
    peaks = DET.pan_tompkins_detector(ecg_signal)
    return np.array(peaks, dtype=int)


def detect_r_peaks_swt_ECG_Detectors(ecg_signal, sampling_rate):
    """Detect R peaks using Stationary Wavelet Transform (SWT) algorithm"""
    peaks = DET.swt_detector(ecg_signal)
    return np.array(peaks, dtype=int)


def detect_r_peaks_matched_filter_ECG_Detectors(ecg_signal, sampling_rate):
    """Detect R peaks using Matched Filter algorithm"""
    supported_rates = [TGT_SAMPLING_RATE, 360, 400, 500]
    if sampling_rate not in supported_rates:
        print(
            f"Warning: Matched Filter requires sampling_rate in {supported_rates} Hz (current: {sampling_rate} Hz)"
        )
        print("Skipping Matched Filter detector")
        return np.array([], dtype=int)

    peaks = DET.matched_filter_detector(ecg_signal)
    return np.array(peaks, dtype=int)


def detect_r_peaks_wqrs_ECG_Detectors(ecg_signal, sampling_rate):
    # Ensure minimum signal length for WQRS algorithm
    min_length = int(sampling_rate * 2)  # At least 2 seconds of signal
    if len(ecg_signal) < min_length:
        print(
            f"Warning: Signal length {len(ecg_signal)} is too short for WQRS detection"
        )
        return np.array([], dtype=int)

    try:
        peaks = DET.wqrs_detector(ecg_signal)
        return np.array(peaks, dtype=int)
    except IndexError as e:
        print(f"WQRS detection failed: {str(e)}")
        return np.array([], dtype=int)


def detect_r_peaks_two_moving_average_ECG_Detectors(ecg_signal, sampling_rate):
    """Detect R peaks using Two Moving Average algorithm"""
    peaks = DET.two_average_detector(ecg_signal)
    return np.array(peaks, dtype=int)


def detect_r_peaks_pantompkins1985_NeuroKit2(ecg_signal, sampling_rate=500):
    """Pan-Tompkins (1985) algorithm."""
    cleaned = nk.ecg_clean(
        ecg_signal, sampling_rate=sampling_rate, method="pantompkins1985"
    )
    _, info = nk.ecg_peaks(
        cleaned, sampling_rate=sampling_rate, method="pantompkins1985"
    )
    return np.array(info["ECG_R_Peaks"], dtype=int)


def detect_r_peaks_hamilton2002_NeuroKit2(ecg_signal, sampling_rate=500):
    """Hamilton (2002) algorithm."""
    cleaned = nk.ecg_clean(
        ecg_signal, sampling_rate=sampling_rate, method="hamilton2002"
    )
    _, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate, method="hamilton2002")
    return np.array(info["ECG_R_Peaks"], dtype=int)


def detect_r_peaks_christov2004_NeuroKit2(ecg_signal, sampling_rate=500):
    """Christov (2004) algorithm."""
    cleaned = nk.ecg_clean(
        ecg_signal, sampling_rate=sampling_rate, method="christov2004"
    )
    _, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate, method="christov2004")
    return np.array(info["ECG_R_Peaks"], dtype=int)


def detect_r_peaks_engzeemod2012_NeuroKit2(ecg_signal, sampling_rate=500):
    """Engelse-Zeelenberg modified (2012) algorithm."""
    cleaned = nk.ecg_clean(
        ecg_signal, sampling_rate=sampling_rate, method="engzeemod2012"
    )
    _, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate, method="engzeemod2012")
    return np.array(info["ECG_R_Peaks"], dtype=int)


def detect_r_peaks_elgendi2010_NeuroKit2(ecg_signal, sampling_rate=500):
    """Elgendi (2010) frequency band-based algorithm."""
    cleaned = nk.ecg_clean(
        ecg_signal, sampling_rate=sampling_rate, method="elgendi2010"
    )
    _, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate, method="elgendi2010")
    return np.array(info["ECG_R_Peaks"], dtype=int)


def detect_r_peaks_zong2003_NeuroKit2(ecg_signal, sampling_rate=500):
    """Zong (2003) algorithm (also known as ssf/slope sum function)."""
    cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="zong2003")
    _, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate, method="zong2003")
    return np.array(info["ECG_R_Peaks"], dtype=int)


def detect_r_peaks_martinez2004_NeuroKit2(ecg_signal, sampling_rate=500):
    """Martinez (2004) wavelet-based algorithm."""
    _, info = nk.ecg_peaks(
        ecg_signal, sampling_rate=sampling_rate, method="martinez2004"
    )
    return np.array(info["ECG_R_Peaks"], dtype=int)


def detect_r_peaks_kalidas2017_NeuroKit2(ecg_signal, sampling_rate=500):
    """Kalidas (2017) stationary wavelet transform (SWT) algorithm."""
    cleaned = nk.ecg_clean(
        ecg_signal, sampling_rate=sampling_rate, method="kalidas2017"
    )
    _, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate, method="kalidas2017")
    return np.array(info["ECG_R_Peaks"], dtype=int)


def detect_r_peaks_khamis2016_NeuroKit2(ecg_signal, sampling_rate=500):
    """Khamis (2016) algorithm (UNSW)."""
    _, info = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate, method="khamis2016")
    return np.array(info["ECG_R_Peaks"], dtype=int)


def detect_r_peaks_manikandan2012_NeuroKit2(ecg_signal, sampling_rate=500):
    """Manikandan & Soman (2012) Shannon energy envelope algorithm."""
    _, info = nk.ecg_peaks(
        ecg_signal, sampling_rate=sampling_rate, method="manikandan2012"
    )
    return np.array(info["ECG_R_Peaks"], dtype=int)


def detect_r_peaks_nabian2018_NeuroKit2(ecg_signal, sampling_rate=500):
    """Nabian (2018) modified Pan-Tompkins algorithm."""
    _, info = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate, method="nabian2018")
    return np.array(info["ECG_R_Peaks"], dtype=int)


def detect_r_peaks_rodrigues2020_NeuroKit2(ecg_signal, sampling_rate=500):
    """Rodrigues (2020) algorithm."""
    _, info = nk.ecg_peaks(
        ecg_signal, sampling_rate=sampling_rate, method="rodrigues2020"
    )
    return np.array(info["ECG_R_Peaks"], dtype=int)


def detect_r_peaks_emrich2023_NeuroKit2(ecg_signal, sampling_rate=500):
    """Emrich (2023) FastNVG algorithm."""
    _, info = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate, method="emrich2023")
    return np.array(info["ECG_R_Peaks"], dtype=int)


def detect_r_peaks_neurokit_NeuroKit2(ecg_signal, sampling_rate=500):
    """NeuroKit2 default algorithm."""
    cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="neurokit")
    _, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate, method="neurokit")
    return np.array(info["ECG_R_Peaks"], dtype=int)


def detect_r_peaks_gamboa2008_NeuroKit2(ecg_signal, sampling_rate=500):
    """Gamboa (2008) multi-modal biometric algorithm."""
    cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="gamboa2008")
    _, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate, method="gamboa2008")
    return np.array(info["ECG_R_Peaks"], dtype=int)


def detect_r_peaks_promac_NeuroKit2(ecg_signal, sampling_rate=500):
    """Promac (probabilistic agreement via convolution) - combines multiple detectors."""
    _, info = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate, method="promac")
    return np.array(info["ECG_R_Peaks"], dtype=int)


def detect_r_peaks_asi_NeuroKit2(ecg_signal, sampling_rate=500):
    """ASI (Adaptive Slope Integration) algorithm."""
    _, info = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate, method="asi")
    return np.array(info["ECG_R_Peaks"], dtype=int)


def evaluate_r_peak_detection(r_peaks, ecg_signal, sampling_rate):
    """Evaluate R peak detection quality"""
    if len(r_peaks) == 0:
        return {
            "peak_count": 0,
            "avg_peak_distance": 0,
            "peak_regularity": 0,
            "detection_reliability": 0,
        }

    peak_count = len(r_peaks)

    if peak_count > 1:
        peak_distances = np.diff(r_peaks)
        avg_peak_distance = np.mean(peak_distances) / sampling_rate
        heart_rate = 60 / avg_peak_distance
        peak_regularity = np.std(peak_distances) / np.mean(peak_distances) * 100
    else:
        heart_rate = 0
        peak_regularity = 0

    peak_amplitudes = ecg_signal[r_peaks]
    avg_peak_amplitude = np.mean(peak_amplitudes)
    noise_level = np.std(ecg_signal)
    detection_reliability = avg_peak_amplitude / noise_level if noise_level > 0 else 0

    return {
        "peak_count": peak_count,
        "avg_peak_distance_sec": avg_peak_distance if peak_count > 1 else 0,
        "heart_rate_bpm": heart_rate if peak_count > 1 else 0,
        "peak_regularity_percent": peak_regularity if peak_count > 1 else 0,
        "detection_reliability": detection_reliability,
        "avg_peak_amplitude": avg_peak_amplitude,
    }


def plot_r_peak_detection_comparison(
    filtered_signal,
    r_peaks_dict,
    sampling_rate,
    lead_number,
    col = 4,
):
    """Plot R peak detection comparison"""
    time_axis = np.arange(len(filtered_signal)) / sampling_rate
    methods = list(r_peaks_dict.keys())
    n_methods = len(methods)

    # Create subplots
    fig, axes = plt.subplots(
        ((n_methods + col - 1) // col),
        col,
        figsize=(48, 6 * (n_methods + col - 1) // col),
    )

    for idx, (method, peaks) in enumerate(r_peaks_dict.items()):
        ax = axes[idx // col, idx % col] if n_methods > 1 else axes

        # Plot ECG signal
        ax.plot(
            time_axis,
            filtered_signal,
            "b-",
            linewidth=1,
            # label="ECG Signal",f
        )

        # Plot R peaks
        ax.plot(
            time_axis[peaks],
            filtered_signal[peaks],
            "o",
            color="r",
            markersize=3,
            # label="R Peaks",
        )

        # Set title and labels
        ax.set_title(
            f"{method}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_ylabel("Amplitude (mV)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add x-axis label only for the last subplot
        if idx == n_methods - 1:
            ax.set_xlabel("Time (seconds)")

    plt.subplots_adjust(
        left=0.036, bottom=0.029, right=0.997, top=0.974, wspace=0.152, hspace=0.426
    )
    plt.show()
