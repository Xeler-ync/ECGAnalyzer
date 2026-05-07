import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as signal


def extract_heartbeats(ecg_signal, r_peaks, sampling_rate):
    """
    Extract individual heartbeats centered at R peaks

    Each R-R interval is divided equally:
    Left boundary = (R[i-1] + R[i]) / 2
    Right boundary = (R[i] + R[i+1]) / 2

    Parameters:
    -----------
    ecg_signal : array
        Filtered ECG signal
    r_peaks : array
        Indices of detected R peaks
    sampling_rate : int
        Sampling rate in Hz

    Returns:
    --------
    list of dicts: heartbeat data
    """
    heartbeats = []

    if len(r_peaks) < 3:
        return heartbeats

    # Only use middle heartbeats (exclude first and last)
    for i in range(1, len(r_peaks) - 1):
        # Calculate boundaries: divide each RR interval in half
        left_boundary = int((r_peaks[i - 1] + r_peaks[i]) / 2)
        right_boundary = int((r_peaks[i] + r_peaks[i + 1]) / 2)

        # Extract heartbeat segment
        heartbeat = ecg_signal[left_boundary:right_boundary]

        # Calculate relative R peak position within this heartbeat
        r_peak_relative = r_peaks[i] - left_boundary

        heartbeats.append(
            {
                "signal": heartbeat,
                "start_index": left_boundary,
                "end_index": right_boundary,
                "r_peak_index": r_peaks[i],
                "r_peak_relative": r_peak_relative,
                "duration_sec": (right_boundary - left_boundary) / sampling_rate,
                "duration_ms": (right_boundary - left_boundary) / sampling_rate * 1000,
            }
        )

    return heartbeats


def split_and_resample_heartbeats(heartbeats, sampling_rate):
    """
    Normalize heartbeats by aligning to R using scaling (resampling).

    - Determine target pre/post lengths from the minimum available samples across all beats.
    - Resample each beat's pre-R and post-R segments to these target lengths using linear interpolation.
    """
    normalized_heartbeats = []

    if len(heartbeats) == 0:
        return normalized_heartbeats, 0, 0, 0

    # collect actual pre/post lengths
    pre_lengths = []
    post_lengths = []
    for hb in heartbeats:
        rpos = hb["r_peak_relative"]
        sig_len = len(hb["signal"])
        pre_len = rpos + 1               # samples from start to R inclusive
        post_len = sig_len - rpos        # samples from R to end inclusive
        pre_lengths.append(pre_len)
        post_lengths.append(post_len)

    # Target lengths: smallest observed (no upsampling, only downsampling)
    target_pre = max(1, int(min(pre_lengths)))
    target_post = max(1, int(min(post_lengths)))
    total_samples = target_pre + target_post

    def resample_segment(seg, target_len):
        """Resample a 1D signal to target_len using linear interpolation."""
        if len(seg) == target_len:
            return seg
        x_old = np.linspace(0, 1, len(seg))
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, seg)

    for hb in heartbeats:
        sig = hb["signal"]
        rpos = hb["r_peak_relative"]

        # Extract full pre-R segment (including R) and post-R segment (including R)
        pre_seg = sig[:rpos + 1]          # length = rpos+1
        post_seg = sig[rpos:]             # length = len(sig)-rpos

        # Resample to target lengths
        resampled_pre = resample_segment(pre_seg, target_pre)
        resampled_post = resample_segment(post_seg, target_post)

        # Combine: avoid duplicate R sample (R is the last of pre, first of post)
        if target_pre > 1:
            normalized = np.concatenate([resampled_pre[:-1], resampled_post])
        else:
            # If target_pre == 1, pre_seg is just the R sample; use it as is
            normalized = np.concatenate([resampled_pre, resampled_post])

        normalized_heartbeats.append({
            "signal": normalized,
            "original_duration_ms": hb["duration_ms"],
            "normalized_duration_ms": (total_samples / sampling_rate) * 1000,
            "r_peak_relative": target_pre - 1,   # R peak position after resampling
            "pre_r_samples": target_pre,
            "post_r_samples": target_post,
            "original_r_peak_index": hb["r_peak_index"],
        })

    return normalized_heartbeats, target_pre, target_post, total_samples


def plot_heartbeats_overlay_normalized(
    normalized_heartbeats, sampling_rate, lead_name, max_beats=20
):
    """
    Plot normalized heartbeats overlaid with aligned R peaks

    Parameters:
    -----------
    normalized_heartbeats : list of dicts
        Normalized heartbeat data with fixed length
    sampling_rate : int
        Sampling rate in Hz
    lead_name : str
        Lead name
    max_beats : int
        Maximum number of beats to plot
    """
    if len(normalized_heartbeats) == 0:
        print(f"No heartbeats found for lead {lead_name}")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    num_beats = min(len(normalized_heartbeats), max_beats)
    total_samples = len(normalized_heartbeats[0]["signal"])
    time_axis = np.arange(total_samples) / sampling_rate * 1000  # Convert to ms

    # Plot all heartbeats
    for i, hb in enumerate(normalized_heartbeats[:num_beats]):
        ax.plot(time_axis, hb["signal"], alpha=0.6, linewidth=1.5, label=f"Beat {i+1}")

    # Mark R peak location (vertical line)
    r_peak_time = normalized_heartbeats[0]["r_peak_relative"] / sampling_rate * 1000
    ax.axvline(
        r_peak_time,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label="R Peak Location",
    )

    ax.set_title(
        f"Aligned Normalized Heartbeats - Lead {lead_name} (First {num_beats} Beats)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Time (ms) - R Peak Aligned")
    ax.set_ylabel("Amplitude (mV)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_heartbeats_overlay_original(
    heartbeats, sampling_rate, lead_name, max_beats=20
):
    """
    Plot multiple original heartbeats overlaid for comparison

    Parameters:
    -----------
    heartbeats : list of dicts
        Original heartbeat data
    sampling_rate : int
        Sampling rate in Hz
    lead_name : str
        Lead name
    max_beats : int
        Maximum number of beats to plot
    """
    if len(heartbeats) == 0:
        print(f"No heartbeats found for lead {lead_name}")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    num_beats = min(len(heartbeats), max_beats)

    for i, hb in enumerate(heartbeats[:num_beats]):
        time_axis = np.arange(len(hb["signal"])) / sampling_rate
        ax.plot(time_axis, hb["signal"], alpha=0.6, label=f"Beat {i+1}")

    ax.set_title(
        f"Original Heartbeats (Unaligned) - Lead {lead_name} (First {num_beats} Beats)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Time within beat (seconds)")
    ax.set_ylabel("Amplitude (mV)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_average_heartbeat_with_variance(
    normalized_heartbeat, sampling_rate, lead_name
):
    """
    Plot average heartbeat with variance overlay and all individual heartbeats in background

    Args:
    heartbeats: List of heartbeat data (dictionaries)
    sampling_rate: Sampling rate in Hz
    lead_name: Name of the ECG lead
    """
    if not normalized_heartbeat:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract signals from dictionaries
    signals = [hb["signal"] for hb in normalized_heartbeat]
    # Calculate average heartbeat
    avg_heartbeat = np.mean(signals, axis=0)
    # Calculate standard deviation
    std_heartbeat = np.std(signals, axis=0)
    # Calculate pairwise standard deviation
    avg_std = np.mean(std_heartbeat)

    std_std_heartbeat = np.std(std_heartbeat)

    # Create time axis
    duration = len(avg_heartbeat) / sampling_rate
    time_axis = np.linspace(0, duration, len(avg_heartbeat))

    # Plot all individual heartbeats with high transparency
    for signal in signals:
        ax.plot(time_axis, signal, alpha=0.7, linewidth=0.5)

    # Get R-peak position
    r_peak_pos = normalized_heartbeat[0]["r_peak_relative"]
    r_peak_time = time_axis[r_peak_pos]

    # Plot vertical dashed line at R-peak position
    ax.axvline(x=r_peak_time, color="red", linestyle="--", alpha=0.8, label="R Peak")

    # Add a point at the R-peak position on the average signal
    ax.plot(r_peak_time, avg_heartbeat[r_peak_pos], "ro", markersize=8)

    # Plot average heartbeat
    ax.plot(time_axis, avg_heartbeat, "b-", label="Average Heartbeat", linewidth=2)

    # Plot positive and negative standard deviation regions
    ax.fill_between(
        time_axis,
        avg_heartbeat - std_heartbeat,
        avg_heartbeat + std_heartbeat,
        alpha=0.3,
        color="blue",
        label="+/-1 std",
    )
    ax.fill_between(
        time_axis,
        avg_heartbeat - 2 * std_heartbeat,
        avg_heartbeat + 2 * std_heartbeat,
        alpha=0.2,
        color="blue",
        label="+/-2 std",
    )

    # Plot pairwise standard deviation
    ax.plot(time_axis, std_heartbeat, "r-", label="Std", linewidth=2)

    # Get R-peak position from first heartbeat
    r_peak_pos = normalized_heartbeat[0]["r_peak_relative"]

    # Check if R-peak position is actually the maximum value
    max_pos = np.argmax(avg_heartbeat)
    ax.set_title(
        f"Average Heartbeat with Variance - Lead {lead_name}\nAvg Std: {avg_std:.4f}, Std of Std: {std_std_heartbeat:.4f}, Max Std: {np.max(std_heartbeat):.4f}"
        + f"{f"\nWarning: R-peak position (index {r_peak_pos}) is not the maximum value in averaged signal."
             if max_pos != r_peak_pos else ""}"
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (mV)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add annotations
    pre_r_time = normalized_heartbeat[0]["pre_r_samples"] / sampling_rate * 1000
    post_r_time = normalized_heartbeat[0]["post_r_samples"] / sampling_rate * 1000
    ax.text(
        0.02,
        0.95,
        f"Pre-R: {pre_r_time:.2f} ms | Post-R: {post_r_time:.2f} ms\nTotal Duration: {normalized_heartbeat[0]['normalized_duration_ms']:.2f} ms",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    plt.tight_layout()
    plt.show()


def plot_heartbeat_evaluation_all(r_peaks, ecg_signal, sampling_rate, lead_name):
    """
    Plot standard deviation comparison for all R peak detection methods

    Args:
        r_peaks: Dictionary of R peaks from different detection methods
        ecg_signal: Original ECG signal
        sampling_rate: Sampling rate in Hz
        lead_name: Name of the ECG lead
    """
    if not r_peaks:
        return

    # Create figure for standard deviations comparison
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)

    # Store statistics for all methods
    all_stats = {}
    min_length = float("inf")  # Track minimum heartbeat length

    # First pass: find minimum heartbeat length
    for method, peaks in r_peaks.items():
        if len(peaks) < 3:
            continue
        heartbeats = extract_heartbeats(ecg_signal, peaks, sampling_rate)
        normalized_heartbeats, _, _, _ = split_and_resample_heartbeats(
            heartbeats, sampling_rate
        )
        if normalized_heartbeats:
            min_length = min(min_length, len(normalized_heartbeats[0]["signal"]))

    # Second pass: process each method with consistent length
    for method, peaks in r_peaks.items():
        if len(peaks) < 3:
            continue

        # Extract heartbeats
        heartbeats = extract_heartbeats(ecg_signal, peaks, sampling_rate)
        normalized_heartbeats, _, _, _ = split_and_resample_heartbeats(
            heartbeats, sampling_rate
        )

        if not normalized_heartbeats:
            continue

        # Truncate all heartbeats to minimum length
        signals = [hb["signal"][:min_length] for hb in normalized_heartbeats]
        std_heartbeat = np.std(signals, axis=0)

        # Store statistics
        all_stats[method] = {
            "avg_std": np.mean(std_heartbeat),
            "std_std": np.std(std_heartbeat),
            "max_std": np.max(std_heartbeat),
        }

    if all_stats:
        methods = list(all_stats.keys())
        avg_stds = [all_stats[m]["avg_std"] for m in methods]
        std_stds = [all_stats[m]["std_std"] for m in methods]
        max_stds = [all_stats[m]["max_std"] for m in methods]

        x = np.arange(len(methods))
        width = 0.25  # Width of the bars

        # Create bars for each metric
        bars1 = ax.bar(x - width, avg_stds, width, label="Average Std", color="skyblue")
        bars2 = ax.bar(x, std_stds, width, label="Std of Std", color="lightgreen")
        bars3 = ax.bar(x + width, max_stds, width, label="Max Std", color="salmon")

        ax.set_title("Standard Deviation Metrics Comparison")
        ax.set_xlabel("Detection Methods")
        ax.set_ylabel("Standard Deviation (mV)")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
    plt.show()
