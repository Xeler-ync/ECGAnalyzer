import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import uniform_filter1d

from utils._config import LEAD_NAMES
from utils._signal import analyze_signal_fft


def remove_baseline_wander_hp_filter(ecg_signal, sampling_rate, cutoff=0.5):
    """Remove baseline wander using high-pass filter"""
    nyquist = sampling_rate / 2
    normalized_cutoff = cutoff / nyquist
    b, a = signal.butter(4, normalized_cutoff, btype="high")
    filtered_signal = signal.filtfilt(b, a, ecg_signal)
    return filtered_signal


def remove_baseline_wander_savgol(ecg_signal, window_length=201, polyorder=3):
    """Remove baseline wander using Savitzky-Golay filter"""
    if window_length % 2 == 0:
        window_length += 1
    baseline = signal.savgol_filter(ecg_signal, window_length, polyorder)
    filtered_signal = ecg_signal - baseline
    return filtered_signal


def remove_baseline_wander_morphological(ecg_signal, kernel_size=51):
    """Remove baseline wander using morphological operations"""
    kernel = np.ones(kernel_size) / kernel_size
    baseline = uniform_filter1d(ecg_signal, size=kernel_size, mode="nearest")
    filtered_signal = ecg_signal - baseline
    return filtered_signal


def plot_fft_and_baseline_analysis(
    ecg_signal, baseline_signal, sampling_rate, lead_number
):
    """
    Plot FFT, baseline estimation, and comparison with original signal

    Plot the FFT, baseline estimate, and comparison with the original signal.
    """
    fft_analysis = analyze_signal_fft(ecg_signal, sampling_rate)

    time_axis = np.arange(len(ecg_signal)) / sampling_rate

    fig = plt.figure(figsize=(18, 14))

    # Create grid spec for better layout control
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # ============ Plot 1: Original Signal ============
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_axis, ecg_signal, "b-", linewidth=1, label="Original ECG Signal")
    ax1.set_title(
        f"Original ECG Signal (Lead {LEAD_NAMES[lead_number]})", fontsize=13, fontweight="bold"
    )
    ax1.set_ylabel("Amplitude (mV)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    ax1.set_xlim(0, min(10, time_axis[-1]))  # Show first 10 seconds

    # ============ Plot 2: FFT Magnitude Spectrum (Linear Scale) ============
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(
        fft_analysis["fft_freq_pos"],
        fft_analysis["fft_magnitude_pos"],
        "g-",
        linewidth=1.5,
    )
    ax2.axvline(
        x=1, color="r", linestyle="--", linewidth=2, label="Baseline cutoff (1 Hz)"
    )
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_title(
        "FFT Magnitude Spectrum (Linear Scale)", fontsize=12, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, 50)  # Focus on 0-50 Hz

    # ============ Plot 3: FFT Magnitude Spectrum (Log Scale) ============
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.semilogy(
        fft_analysis["fft_freq_pos"],
        fft_analysis["fft_magnitude_pos"],
        "g-",
        linewidth=1.5,
    )
    ax3.axvline(
        x=1, color="r", linestyle="--", linewidth=2, label="Baseline cutoff (1 Hz)"
    )

    # Highlight different frequency bands
    ax3.axvspan(0, 1, alpha=0.1, color="red", label="Baseline band (<1 Hz)")
    ax3.axvspan(1, 10, alpha=0.1, color="yellow", label="ECG band (1-10 Hz)")
    ax3.axvspan(10, 50, alpha=0.1, color="cyan", label="High-freq noise (>10 Hz)")

    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Magnitude (log scale)")
    ax3.set_title("FFT Magnitude Spectrum (Log Scale)", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, which="both")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.set_xlim(0, 50)

    # ============ Plot 4: Baseline Estimation ============
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(
        time_axis, ecg_signal, "b-", linewidth=1, alpha=0.7, label="Original Signal"
    )
    ax4.plot(
        time_axis,
        fft_analysis["baseline_signal"],
        "r-",
        linewidth=2.5,
        label="Estimated Baseline (FFT < 1Hz)",
    )
    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylabel("Amplitude (mV)")
    ax4.set_title(
        "Baseline Estimation from Low-Frequency Components",
        fontsize=12,
        fontweight="bold",
    )
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xlim(0, min(10, time_axis[-1]))

    # ============ Plot 5: Overlay - Original, Baseline, and Detrended ============
    ax5 = fig.add_subplot(gs[2, 1])
    detrended_signal = ecg_signal - fft_analysis["baseline_signal"]

    ax5.plot(
        time_axis, ecg_signal, "b-", linewidth=1, alpha=0.6, label="Original Signal"
    )
    ax5.plot(
        time_axis,
        fft_analysis["baseline_signal"],
        "r-",
        linewidth=2.5,
        alpha=0.8,
        label="Estimated Baseline",
    )
    ax5.plot(
        time_axis,
        detrended_signal,
        "g-",
        linewidth=1,
        alpha=0.7,
        label="Detrended (Original - Baseline)",
    )

    ax5.set_xlabel("Time (seconds)")
    ax5.set_ylabel("Amplitude (mV)")
    ax5.set_title(
        "Signal Overlay: Original, Baseline, and Detrended",
        fontsize=12,
        fontweight="bold",
    )
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc="upper right")
    ax5.set_xlim(0, min(10, time_axis[-1]))

    plt.suptitle(
        f"FFT-based Baseline Estimation and Analysis (Lead V{lead_number})",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    plt.show()

    # Print FFT analysis results
    print(f"\n{'='*70}")
    print(f"FFT ANALYSIS RESULTS (Lead V{lead_number})")
    print(f"{'='*70}")
    print(f"Signal duration: {time_axis[-1]:.2f} seconds")
    print(f"Signal length: {len(ecg_signal)} samples")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Frequency resolution: {sampling_rate/len(ecg_signal):.4f} Hz/bin")

    # Find dominant frequencies
    sorted_freq_idx = np.argsort(fft_analysis["fft_magnitude_pos"])[::-1]
    print(f"\nTop 5 dominant frequencies:")
    for i, idx in enumerate(sorted_freq_idx[:5]):
        freq = fft_analysis["fft_freq_pos"][idx]
        mag = fft_analysis["fft_magnitude_pos"][idx]
        print(f"  {i+1}. {freq:.3f} Hz (Magnitude: {mag:.2f})")

    # Baseline statistics
    baseline_amp = fft_analysis["baseline_signal"]
    detrended = ecg_signal - baseline_amp

    print(f"\nBaseline Statistics:")
    print(f"  Baseline min: {np.min(baseline_amp):.4f} mV")
    print(f"  Baseline max: {np.max(baseline_amp):.4f} mV")
    print(f"  Baseline amplitude: {np.max(baseline_amp) - np.min(baseline_amp):.4f} mV")
    print(f"  Baseline std: {np.std(baseline_amp):.4f} mV")

    print(f"\nDetrended Signal Statistics:")
    print(f"  Original signal std: {np.std(ecg_signal):.4f} mV")
    print(f"  Detrended signal std: {np.std(detrended):.4f} mV")
    print(
        f"  Noise reduction: {(np.std(ecg_signal) - np.std(detrended))/np.std(ecg_signal)*100:.2f}%"
    )
    print(f"{'='*70}\n")

    return fft_analysis, detrended_signal


def evaluate_baseline_removal(original_signal, filtered_signal):
    """Evaluate baseline removal quality"""
    noise_original = np.std(original_signal)
    noise_filtered = np.std(filtered_signal)
    rms_error = np.sqrt(np.mean(filtered_signal**2))
    smoothness = np.mean(np.abs(np.diff(filtered_signal)))
    variance_reduction = (
        (np.var(original_signal) - np.var(filtered_signal))
        / np.var(original_signal)
        * 100
    )

    return {
        "noise_std_original": noise_original,
        "noise_std_filtered": noise_filtered,
        "rms_error": rms_error,
        "smoothness": smoothness,
        "variance_reduction": variance_reduction,
    }


def plot_baseline_removal_comparison(
    original_signal,
    filtered_hp,
    filtered_savgol,
    filtered_morpho,
    sampling_rate,
    lead_number,
):
    """Plot baseline removal comparison"""
    time_axis = np.arange(len(original_signal)) / sampling_rate

    fig, axes = plt.subplots(4, 1, figsize=(16, 12))

    axes[0].plot(time_axis, original_signal, "b-", linewidth=1)
    axes[0].set_title(
        f"Original ECG Signal (Lead V{lead_number})", fontsize=12, fontweight="bold"
    )
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_axis, filtered_hp, "g-", linewidth=1)
    axes[1].set_title(
        "Baseline Removal: High-Pass Filter (Butterworth)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_ylabel("Amplitude (mV)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_axis, filtered_savgol, "r-", linewidth=1)
    axes[2].set_title(
        "Baseline Removal: Savitzky-Golay Filter", fontsize=12, fontweight="bold"
    )
    axes[2].set_ylabel("Amplitude (mV)")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(time_axis, filtered_morpho, "purple", linewidth=1)
    axes[3].set_title(
        "Baseline Removal: Morphological Filter", fontsize=12, fontweight="bold"
    )
    axes[3].set_xlabel("Time (seconds)")
    axes[3].set_ylabel("Amplitude (mV)")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
