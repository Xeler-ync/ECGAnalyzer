import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures
from scipy.signal import find_peaks

from utils._baseline import remove_baseline_wander_hp_filter
from utils._r_peaks import detect_r_peaks_neurokit_NeuroKit2
from utils._config import (
    SAMPLING_RATE,
    RESULTS_PATH,
    PATH,
    LEAD_NAMES,
    ECG_INDEX,
)
from utils._data import load_raw_data, Y
from utils._helpers import _round_and_clip_indices


def plot_baseline_removed_signal(X, signal_index, leads_to_plot=None, auto_layout=True):
    """
    Plot the baseline removed ECG signal and its FFT for specified leads with R-peak markers
    The FFT analysis is performed on the baseline removed signal, not the original signal
    Also show pure baseline and pure filtered signals for each lead

    Parameters:
        X: Original ECG data
        signal_index: Signal index
        leads_to_plot: List of lead indices to plot
                      If None, plot all 12 leads
        auto_layout: Boolean, if True, automatically adjust layout based on number of leads
    """
    # Determine which leads to plot
    if leads_to_plot is None:
        leads_to_plot = list(range(len(LEAD_NAMES)))

    num_leads = len(leads_to_plot)

    # Calculate optimal layout
    if auto_layout:
        # Calculate number of rows and columns for time domain plots
        # Each lead now gets 4 subplots: time domain, frequency domain, baseline, and filtered
        n_cols = int(np.ceil(np.sqrt(num_leads * 4)))
        n_rows = int(np.ceil(num_leads / (n_cols / 4)))

        # Create figure with dynamic size based on number of leads
        fig_width = n_cols * 3
        fig_height = n_rows * 4
    else:
        # Use default layout (6 rows, 8 columns for 12 leads)
        n_cols = 8
        n_rows = 6
        fig_width = 24
        fig_height = 24

    # Create subplot layout (each lead gets four subplots)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # Flatten axes if it's 2D
    if n_rows > 1 and n_cols > 1:
        axes = axes.flatten()
    elif n_rows == 1 and n_cols > 1:
        axes = axes
    elif n_rows > 1 and n_cols == 1:
        axes = axes.flatten()
    else:
        axes = np.array([axes])

    # Get signal data
    ecg_signal = X[0, :, :]

    # Create time axis
    time_axis = np.arange(len(ecg_signal)) / SAMPLING_RATE

    # Process Lead II to get R-peak positions
    lead_II_idx = 1
    lead_II_signal = ecg_signal[:, lead_II_idx]
    cutoff = 0.5
    filtered_II = remove_baseline_wander_hp_filter(
        lead_II_signal, SAMPLING_RATE, cutoff=cutoff
    )
    r_peaks_II = _round_and_clip_indices(
        detect_r_peaks_neurokit_NeuroKit2(filtered_II, SAMPLING_RATE),
        len(filtered_II),
        filtered_II,
        "NeuroKit2",
    )

    # Convert R-peak indices to time points
    r_peaks_time = r_peaks_II / SAMPLING_RATE

    # Calculate display range based on sampling rate
    display_samples = 5000  # Number of samples to display
    display_time = display_samples / SAMPLING_RATE  # Convert to seconds

    # Process and plot each lead
    for i, lead_idx in enumerate(leads_to_plot):
        lead_name = LEAD_NAMES[lead_idx]

        # Four subplots for each lead:
        # 0: Time domain (baseline removed signal)
        # 1: Frequency domain (FFT analysis)
        # 2: Pure baseline
        # 3: Pure filtered signal
        ax_time = axes[i * 4]
        ax_freq = axes[i * 4 + 1]
        ax_time_domain_baseline = axes[i * 4 + 2]
        ax_time_domain_filtered = axes[i * 4 + 3]

        # Get original signal for current lead
        original_signal = ecg_signal[:, lead_idx]

        # ============ Plot 1: Time Domain (Baseline Removed) ============
        # Process lead to get baseline removed signal
        filtered_signal = remove_baseline_wander_hp_filter(
            original_signal, SAMPLING_RATE, cutoff=cutoff
        )

        # Extract baseline (original - filtered)
        baseline_signal = original_signal - filtered_signal
        ax_time.plot(
            time_axis,
            filtered_signal,
            "b-",
            linewidth=0.8,
            label="Baseline Removed",
        )

        # Add markers at R-peak positions
        ax_time.plot(
            r_peaks_time,
            filtered_signal[r_peaks_II],
            "ro",
            markersize=4,
            alpha=0.7,
            label="R-peaks",
        )

        # Set title and labels for time domain
        ax_time.set_title(
            f"Lead {lead_name} (Time Domain) (Signal Index: {signal_index})",
            fontsize=10,
            fontweight="bold",
        )
        ax_time.set_xlabel("Time (seconds)")
        ax_time.set_ylabel("Amplitude (mV)")
        ax_time.grid(True, alpha=0.3)

        # Show legend only in first subplot
        if i == 0:
            ax_time.legend(loc="upper right", fontsize="small")

        # Set x-axis range based on sampling rate
        ax_time.set_xlim(0, min(display_time, time_axis[-1]))

        # ============ Plot 2: Frequency Domain (FFT Analysis) ============
        # Compute FFT for frequency domain analysis on the filtered signal
        n = len(filtered_signal)
        fft_signal = np.fft.rfft(filtered_signal)
        fft_freq = np.fft.rfftfreq(n, 1 / SAMPLING_RATE)

        # Keep only positive frequencies
        positive_freq_idx = fft_freq > 0
        fft_freq = fft_freq[positive_freq_idx]
        fft_signal = fft_signal[positive_freq_idx]

        # Compute magnitude spectrum
        fft_magnitude = np.abs(fft_signal) / n

        # Apply high-pass filter to extract baseline from frequency domain
        # The interval between each peak in the comb spectrum is
        # actually a heartbeat cycle, so it is greater than 1 Hz.
        # We only need to remove frequencies less than 1 Hz.
        cutoff_freq = 1  # Hz

        # Apply FFT on the magnitude spectrum to get frequency components of the magnitude
        fft_of_magnitude = np.fft.rfft(fft_magnitude)
        freq_of_magnitude = np.fft.rfftfreq(
            len(fft_magnitude), d=fft_freq[1] - fft_freq[0]
        )

        # Keep only low-frequency components (< 1 Hz) of the magnitude spectrum
        low_freq_mask = np.abs(freq_of_magnitude) < cutoff_freq
        high_freq_mask = np.abs(freq_of_magnitude) >= cutoff_freq
        fft_baseline_magnitude = fft_of_magnitude * low_freq_mask
        fft_filtered_magnitude = fft_of_magnitude * high_freq_mask

        # Convert back to frequency domain using inverse FFT
        baseline_magnitude = np.fft.irfft(fft_baseline_magnitude, n=len(fft_freq))

        # Compute filtered magnitude
        filtered_magnitude = np.fft.irfft(fft_filtered_magnitude, n=len(fft_freq))

        # Plot FFT in frequency domain
        ax_freq.plot(
            fft_freq,
            fft_magnitude,
            "b-",
            linewidth=0.8,
            label="Original",
        )

        ax_freq.plot(
            fft_freq,
            filtered_magnitude,
            "r-",
            linewidth=0.8,
            label="Filtered",
        )

        ax_freq.plot(
            fft_freq,
            baseline_magnitude,
            "g-",
            linewidth=1.2,
            label="Baseline",
        )

        # Add a vertical line at 1 Hz to show baseline cutoff frequency
        ax_freq.axvline(
            x=1,
            color="r",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Baseline Cutoff (1 Hz)",
        )

        # mark the max point in original fft_magnitude
        max_fft_idx = np.argmax(fft_magnitude)
        ax_freq.plot(
            fft_freq[max_fft_idx],
            fft_magnitude[max_fft_idx],
            "bo",
            markersize=8,
            label="Max Original",
        )

        # mark the max point in filered fft_magnitude
        max_filtered_idx = np.argmax(filtered_magnitude)
        ax_freq.plot(
            fft_freq[max_filtered_idx],
            filtered_magnitude[max_filtered_idx],
            "ro",
            markersize=8,
            label="Max Filtered",
        )

        # Add markers at R-peak frequencies
        peaks, _ = find_peaks(fft_magnitude, distance=10 * SAMPLING_RATE / 100)
        ax_freq.plot(
            fft_freq[peaks],
            fft_magnitude[peaks],
            "go",
            markersize=8,
            label="Peaks",
        )

        # Set title and labels for frequency domain
        ax_freq.set_title(
            f"Lead {lead_name} (Frequency Domain)\n"
            + f"Max Original A Freq: {fft_freq[max_fft_idx]} Hz"
            + f"Max Filtered A Freq: {fft_freq[max_filtered_idx]} Hz",
            fontsize=10,
            fontweight="bold",
        )
        ax_freq.set_xlabel("Frequency (Hz)")
        ax_freq.set_ylabel("Magnitude")
        ax_freq.grid(True, alpha=0.3)

        # Show legend only in first subplot
        if i == 0:
            ax_freq.legend(loc="upper right", fontsize="small")

        # Limit frequency range to 0-100 Hz for better visualization
        ax_freq.set_xlim(0, SAMPLING_RATE / 2)

        # ============ Plot 3: FFT Baseline in Time Domain ============
        # Convert FFT baseline back to time domain
        # First, create a complex signal with the baseline magnitude and original phase
        fft_baseline_complex = np.zeros_like(fft_signal)
        fft_baseline_complex = baseline_magnitude * np.exp(1j * np.angle(fft_signal))

        # Perform inverse FFT to get time domain baseline
        time_domain_baseline = np.real(
            np.fft.ifft(fft_baseline_complex, n=len(original_signal))
        )

        # Adjust length to match original signal
        if len(time_domain_baseline) < len(original_signal):
            # Pad with zeros if shorter
            time_domain_baseline = np.pad(
                time_domain_baseline,
                (0, len(original_signal) - len(time_domain_baseline)),
            )
        elif len(time_domain_baseline) > len(original_signal):
            # Truncate if longer
            time_domain_baseline = time_domain_baseline[: len(original_signal)]

        ax_time_domain_baseline.plot(
            time_axis,
            time_domain_baseline,
            "g-",
            linewidth=0.8,
            label="FFT Baseline (Time Domain)",
        )

        # Set title and labels for time domain baseline
        ax_time_domain_baseline.set_title(
            f"Lead {lead_name} (FFT Baseline in Time Domain)",
            fontsize=10,
            fontweight="bold",
        )
        ax_time_domain_baseline.set_xlabel("Time (seconds)")
        ax_time_domain_baseline.set_ylabel("Amplitude (mV)")
        ax_time_domain_baseline.grid(True, alpha=0.3)

        # Set x-axis range based on sampling rate
        ax_time_domain_baseline.set_xlim(0, min(display_time, time_axis[-1]))

        # ============ Plot 4: FFT Filtered Signal in Time Domain ============
        # Convert FFT filtered signal back to time domain
        # First, create a complex signal with the filtered magnitude and original phase
        fft_filtered_complex = np.zeros_like(fft_signal)
        fft_filtered_complex = filtered_magnitude * np.exp(1j * np.angle(fft_signal))

        # Perform inverse FFT to get time domain filtered signal
        time_domain_filtered = np.real(
            np.fft.ifft(fft_filtered_complex, n=len(original_signal))
        )

        # Adjust length to match original signal
        if len(time_domain_filtered) < len(original_signal):
            # Pad with zeros if shorter
            time_domain_filtered = np.pad(
                time_domain_filtered,
                (0, len(original_signal) - len(time_domain_filtered)),
            )
        elif len(time_domain_filtered) > len(original_signal):
            # Truncate if longer
            time_domain_filtered = time_domain_filtered[: len(original_signal)]

        ax_time_domain_filtered.plot(
            time_axis,
            time_domain_filtered,
            "r-",
            linewidth=0.8,
            label="FFT Filtered (Time Domain)",
        )

        # Add markers at R-peak positions
        ax_time_domain_filtered.plot(
            r_peaks_time,
            time_domain_filtered[r_peaks_II],
            "bo",
            markersize=4,
            alpha=0.7,
            label="R-peaks",
        )

        # Set title and labels for time domain filtered signal
        ax_time_domain_filtered.set_title(
            f"Lead {lead_name} (FFT Filtered in Time Domain)",
            fontsize=10,
            fontweight="bold",
        )
        ax_time_domain_filtered.set_xlabel("Time (seconds)")
        ax_time_domain_filtered.set_ylabel("Amplitude (mV)")
        ax_time_domain_filtered.grid(True, alpha=0.3)

        # Show legend only in first subplot
        if i == 0:
            ax_time_domain_filtered.legend(loc="upper right", fontsize="small")

        # Set x-axis range based on sampling rate
        ax_time_domain_filtered.set_xlim(0, min(display_time, time_axis[-1]))

    # Hide unused subplots
    for i in range(num_leads * 4, len(axes)):
        axes[i].axis("off")

    # Adjust subplot spacing
    plt.subplots_adjust(
        left=0.02, bottom=0.052, right=0.98, top=0.914, wspace=0.38, hspace=0.488
    )

    # Save figure
    # plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join(output_dir, f"{signal_index}_fft_baseline.png"))

    # Close the figure to free memory
    plt.close(fig)


output_dir = os.path.join(os.path.dirname(RESULTS_PATH), "irregular")
os.makedirs(output_dir, exist_ok=True)
X = load_raw_data(Y, SAMPLING_RATE, PATH, ECG_INDEX)

# Plot the baseline removed signal
# plot_baseline_removed_signal(X, ECG_INDEX, leads_to_plot=[i for i in range(12)])
plot_baseline_removed_signal(X, ECG_INDEX, leads_to_plot=[1])

# stop script running
exit()

# List of irregular ECG signal indices
with open("./data/irregular_indices.json", "r") as f:
    irregular = json.load(f)["irregular"]


def process_irregular_signal(signal_index):
    plot_baseline_removed_signal(X, signal_index, leads_to_plot=[i for i in range(12)])
    return signal_index


current_backend = matplotlib.get_backend()
matplotlib.use("Agg")
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    futures = [executor.submit(process_irregular_signal, i) for i in irregular]

    for future, idx in enumerate(
        tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing irregular signals",
            unit="signal",
        )
    ):
        try:
            result = future.result()
        except Exception as e:
            print(f"Error processing signal {irregular[idx]}: {e}")

matplotlib.use(current_backend)
