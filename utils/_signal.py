import numpy as np


def analyze_signal_fft(ecg_signal, sampling_rate):
    """
    Perform FFT analysis and estimate baseline from low-frequency components

    分析信号的FFT并从低频成分估计基线
    """
    # Compute FFT
    fft_values = np.fft.fft(ecg_signal)
    fft_freq = np.fft.fftfreq(len(ecg_signal), 1 / sampling_rate)
    fft_magnitude = np.abs(fft_values)

    # Only keep positive frequencies
    positive_freq_idx = fft_freq > 0
    fft_freq_pos = fft_freq[positive_freq_idx]
    fft_magnitude_pos = fft_magnitude[positive_freq_idx]

    # Estimate baseline from very low frequencies (< 1 Hz)
    baseline_freq_threshold = 1  # Hz
    baseline_freq_mask = fft_freq_pos < baseline_freq_threshold
    baseline_fft = np.zeros_like(fft_values)

    low_freq_mask = np.abs(fft_freq) < baseline_freq_threshold
    baseline_fft[low_freq_mask] = fft_values[low_freq_mask]

    # Reconstruct baseline signal from inverse FFT
    baseline_signal = np.real(np.fft.ifft(baseline_fft))

    return {
        "fft_values": fft_values,
        "fft_freq": fft_freq,
        "fft_magnitude": fft_magnitude,
        "fft_freq_pos": fft_freq_pos,
        "fft_magnitude_pos": fft_magnitude_pos,
        "baseline_signal": baseline_signal,
    }


def resample_heartbeat(heartbeat_signal, target_length):
    """
    Resample heartbeat to target length using linear interpolation

    Parameters:
    -----------
    heartbeat_signal : array
        Original heartbeat signal
    target_length : int
        Target length in samples

    Returns:
    --------
    array: resampled heartbeat signal
    """
    original_length = len(heartbeat_signal)
    if original_length == target_length:
        return heartbeat_signal

    x_old = np.linspace(0, 1, original_length)
    x_new = np.linspace(0, 1, target_length)
    resampled = np.interp(x_new, x_old, heartbeat_signal)
    return resampled
