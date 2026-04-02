import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.signal import find_peaks

from utils._config import (
    SAMPLING_RATE,
    LEAD_NAMES,
)


def calc_bpm_by_fft(filtered_ecg_signal, leads_to_use=None, cutoff=1):
    if leads_to_use == -1:
        filtered_ecg_signal = filtered_ecg_signal.reshape(-1, 1)
        leads_to_use = (0,)
    elif leads_to_use is None:
        leads_to_use = list(range(len(LEAD_NAMES)))

    bpss = []
    for lead_idx in leads_to_use:
        filtered_signal = filtered_ecg_signal[:, lead_idx]

        n = len(filtered_signal)
        fft_signal = np.fft.rfft(filtered_signal)
        fft_freq = np.fft.rfftfreq(n, 1 / SAMPLING_RATE)

        # Keep only positive frequencies
        positive_freq_idx = fft_freq > 0
        fft_freq = fft_freq[positive_freq_idx]
        fft_signal = fft_signal[positive_freq_idx]

        # Compute magnitude spectrum
        fft_magnitude = np.abs(fft_signal) / n

        # using scipy find peaks
        peaks, _ = find_peaks(fft_magnitude, distance=10 * SAMPLING_RATE / 100)

        if len(peaks) != 0:
            i = 0
            while i < len(peaks):
                if fft_freq[peaks][i] < cutoff:
                    i += 1
                else:
                    bpss.append(fft_freq[peaks][i])
                    break

    return np.mean(np.array(bpss) * 60)


def calculate_bpm_from_r_peaks(r_peaks, sampling_rate=SAMPLING_RATE):
    """
    Calculate BPM from R-peak positions

    Parameters:
        r_peaks: Array of R-peak positions (in samples)
        sampling_rate: Sampling rate of the ECG signal

    Returns:
        bpm: Calculated beats per minute
    """
    if len(r_peaks) < 2:
        return 0  # Not enough peaks to calculate BPM

    # Calculate RR intervals in samples
    rr_intervals = np.diff(r_peaks)

    # Convert RR intervals to seconds
    rr_intervals_sec = rr_intervals / sampling_rate

    # Calculate average RR interval
    avg_rr_interval = np.mean(rr_intervals_sec)

    # Convert to BPM
    bpm = 60 / avg_rr_interval

    return bpm


# add functionf comment
def is_bpm_diff_significant(r_peaks, fft_bpm, adaptive_threshold=False, interval=-1):
    """Check if bpm diff between r_peaks and fft is significant

    Args:
        r_peaks (int[]): r_peaks position
        fft_bpm (float): fft based bpm
        adaptive_threshold (bool, optional): If using adaptive threshold. Defaults to True.
        interval (float, optional): Signal inerval (seconds). Required when adaptive_threshold is setted to True. Defaults to -1.

    Raises:
        Exception: [description]

    Returns:
        bool: True if bpm diff is significant, False otherwise
    """
    if adaptive_threshold:
        if interval == -1:
            raise ValueError(
                "interval must be specified when adaptive_threshold is True"
            )
        return not (
            60 * (len(r_peaks) - 1) / interval < fft_bpm
            and 60 * (len(r_peaks) + 1) / interval > fft_bpm
        )
    else:
        return not abs(calculate_bpm_from_r_peaks(r_peaks) - fft_bpm) < fft_bpm * 0.1
