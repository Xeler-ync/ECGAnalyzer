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
    return signal.filtfilt(b, a, ecg_signal)


def remove_baseline_wander_savgol(ecg_signal, window_length=201, polyorder=3):
    """Remove baseline wander using Savitzky-Golay filter"""
    if window_length % 2 == 0:
        window_length += 1
    baseline = signal.savgol_filter(ecg_signal, window_length, polyorder)
    return ecg_signal - baseline


def remove_baseline_wander_morphological(ecg_signal, kernel_size=51):
    """Remove baseline wander using morphological operations"""
    kernel = np.ones(kernel_size) / kernel_size
    baseline = uniform_filter1d(ecg_signal, size=kernel_size, mode="nearest")
    return ecg_signal - baseline


def setup_axis(
    ax, time_axis_last, title, ylabel=None, xlabel=None, legend_loc="upper right"
):
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)
    if legend_loc:
        ax.legend(loc=legend_loc)
    ax.set_xlim(0, min(10, time_axis_last))


def plot_fft_spectrum(ax, fft_analysis, cutoff, yscale="linear"):
    ax.plot(
        fft_analysis["fft_freq_pos"],
        fft_analysis["fft_magnitude_pos"],
        "g-",
        linewidth=1.5,
    )
    ax.axvline(
        x=1,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Baseline cutoff ({cutoff} Hz)",
    )
    ax.set_xlim(0, 50)
    if yscale == "log":
        ax.set_yscale("log")
        ax.axvspan(
            0, cutoff, alpha=0.1, color="red", label=f"Baseline band ({cutoff} Hz)"
        )
        ax.axvspan(
            cutoff,
            10,
            alpha=0.1,
            color="yellow",
            label=f"ECG band ({cutoff}-10 Hz)",
        )
        ax.axvspan(10, 50, alpha=0.1, color="cyan", label="High-freq noise (>10 Hz)")
    ax.grid(True, alpha=0.3, which="both" if yscale == "log" else "major")


def print_fft_results(lead_number, ecg_signal, sampling_rate, time_axis, fft_analysis):
    print(f"\n{'='*70}")
    print(f"FFT ANALYSIS RESULTS (Lead {LEAD_NAMES[lead_number]})")
    print(f"{'='*70}")
    print(f"Signal duration: {time_axis[-1]:.2f} seconds")
    print(f"Signal length: {len(ecg_signal)} samples")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Frequency resolution: {sampling_rate/len(ecg_signal):.4f} Hz/bin")

    sorted_freq_idx = np.argsort(fft_analysis["fft_magnitude_pos"])[::-1]
    print(f"\nTop 5 dominant frequencies:")
    for i, idx in enumerate(sorted_freq_idx[:5]):
        freq = fft_analysis["fft_freq_pos"][idx]
        mag = fft_analysis["fft_magnitude_pos"][idx]
        print(f"  {i+1}. {freq:.3f} Hz (Magnitude: {mag:.2f})")

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


def plot_fft_and_baseline_analysis(
    ecg_signal, baseline_signal, sampling_rate, lead_number, cutoff
):
    fft_analysis = analyze_signal_fft(ecg_signal, sampling_rate)
    time_axis = np.arange(len(ecg_signal)) / sampling_rate

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_axis, ecg_signal, "b-", linewidth=1, label="Original ECG Signal")
    setup_axis(
        ax1,
        time_axis[-1],
        f"Original ECG Signal (Lead {LEAD_NAMES[lead_number]})",
        "Amplitude (mV)",
    )

    ax2 = fig.add_subplot(gs[1, 0])
    plot_fft_spectrum(ax2, fft_analysis, cutoff, yscale="linear")
    ax2.set_title(
        "FFT Magnitude Spectrum (Linear Scale)", fontsize=12, fontweight="bold"
    )
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 1])
    plot_fft_spectrum(ax3, fft_analysis, cutoff, yscale="log")
    ax3.set_title("FFT Magnitude Spectrum (Log Scale)", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=9)

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
    setup_axis(
        ax4,
        time_axis[-1],
        "Baseline Estimation from Low-Frequency Components",
        "Amplitude (mV)",
        "Time (seconds)",
    )

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
    setup_axis(
        ax5,
        time_axis[-1],
        "Signal Overlay: Original, Baseline, and Detrended",
        "Amplitude (mV)",
        "Time (seconds)",
    )

    plt.suptitle(
        f"FFT-based Baseline Estimation and Analysis (Lead {LEAD_NAMES[lead_number]})",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.show()

    print_fft_results(lead_number, ecg_signal, sampling_rate, time_axis, fft_analysis)
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
        f"Original ECG Signal (Lead {LEAD_NAMES[lead_number]})",
        fontsize=12,
        fontweight="bold",
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
