import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks, butter, filtfilt

from utils._data import SAMPLING_RATE, X
from utils._baseline import remove_baseline_wander_hp_filter


def find_heartbeats_via_envelope(signal, fs):
    """
    Peak detection based on Hilbert envelope, insensitive to signal polarity.

    Parameters:
        signal: 1D input signal (bandpass filtering recommended)
        fs: sampling rate (Hz)

    Returns:
        peaks: indices of peak positions (corresponding to heartbeat events)
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
        h[0] = 1
        h[1 : (n + 1) // 2] = 2
    analytic = np.fft.ifft(fft * h)

    envelope = np.abs(analytic)  # envelope signal

    # Detect peaks on the envelope
    min_distance = int(0.3 * fs)  # minimum interval 300 ms
    height = 0.5 * np.max(envelope)  # threshold
    peaks, _ = find_peaks(envelope, distance=min_distance, height=height)

    return peaks


def plot_hilbert_envelope(
    signal, fs, filter_signal=True, lowcut=0.5, highcut=5.0, show_peaks=True
):
    """
    Plot the Hilbert envelope of a signal for visualising heartbeat envelope characteristics.

    Parameters:
        signal : array_like
            1D input signal (recommended to be bandpass filtered, or the function can filter it)
        fs : float
            sampling rate (Hz)
        filter_signal : bool, optional
            whether to bandpass filter the signal (default True), with range [lowcut, highcut]
        lowcut : float, optional
            lower cutoff frequency for bandpass filter (Hz), default 0.5
        highcut : float, optional
            upper cutoff frequency for bandpass filter (Hz), default 5.0
        show_peaks : bool, optional
            whether to mark detected heartbeat peaks on the envelope (default True)

    Returns:
        None, displays the figure directly
    """
    # 1. Optional filtering
    if filter_signal:
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(3, [low, high], btype="band")
        signal_proc = filtfilt(b, a, signal)
    else:
        signal_proc = signal.copy()

    # 2. Hilbert envelope
    analytic = hilbert(signal_proc)
    envelope = np.abs(analytic)

    # 3. Time axis
    t = np.arange(len(signal)) / fs

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    # Overlay the processed signal and its envelope
    ax.plot(t, signal_proc, "b-", linewidth=0.8, label="Processed signal")
    ax.plot(t, envelope, "g-", linewidth=1, label="Hilbert envelope")

    if show_peaks:
        # Detect peaks on the envelope
        min_dist = int(0.3 * fs)
        height_th = 0.5 * np.max(envelope)
        envelope_peaks, _ = find_peaks(envelope, distance=min_dist, height=height_th)
        ax.plot(
            t[envelope_peaks],
            envelope[envelope_peaks],
            "rx",
            markersize=8,
            label="Envelope peaks",
        )

        # Detect peaks on the original signal
        signal_height_th = 0.5 * np.max(signal_proc)
        signal_peaks, _ = find_peaks(
            signal_proc, distance=min_dist, height=signal_height_th
        )
        ax.plot(
            t[signal_peaks],
            signal_proc[signal_peaks],
            "bo",
            markersize=6,
            label="Signal peaks",
        )

        ax.set_title(
            "Filtered Signal with Hilbert Envelope and Detected Peaks"
            if filter_signal
            else "Original Signal with Hilbert Envelope and Detected Peaks"
        )
    else:
        ax.set_title(
            "Filtered Signal with Hilbert Envelope"
            if filter_signal
            else "Original Signal with Hilbert Envelope"
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


plot_hilbert_envelope(
    remove_baseline_wander_hp_filter(X[0, :, 11], SAMPLING_RATE),
    SAMPLING_RATE,
    filter_signal=False,
    lowcut=0.5,
    highcut=5.0,
    show_peaks=True,
)