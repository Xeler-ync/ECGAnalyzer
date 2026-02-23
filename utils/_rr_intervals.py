import numpy as np
import matplotlib.pyplot as plt


def calculate_rr_intervals(r_peaks, sampling_rate):
    """
    Calculate R-R intervals and heart rate statistics

    Parameters:
    -----------
    r_peaks : array
        Indices of detected R peaks (in samples)
    sampling_rate : int
        Sampling rate in Hz

    Returns:
    --------
    dict with RR intervals analysis
    """
    if len(r_peaks) < 2:
        return {
            "rr_intervals_sec": np.array([]),
            "rr_intervals_ms": np.array([]),
            "mean_rr_sec": 0,
            "mean_rr_ms": 0,
            "heart_rate_bpm": 0,
            "heart_rate_std": 0,
            "rr_min_ms": 0,
            "rr_max_ms": 0,
        }

    # Calculate RR intervals in samples
    rr_intervals_samples = np.diff(r_peaks)

    # Convert to seconds and milliseconds
    rr_intervals_sec = rr_intervals_samples / sampling_rate
    rr_intervals_ms = rr_intervals_sec * 1000

    # Calculate statistics
    mean_rr_sec = np.mean(rr_intervals_sec)
    mean_rr_ms = np.mean(rr_intervals_ms)

    # Heart rate = 60 / mean_RR_interval (in seconds)
    heart_rate_bpm = 60 / mean_rr_sec

    # Instantaneous heart rate for each beat
    heart_rates = 60 / rr_intervals_sec
    heart_rate_std = np.std(heart_rates)

    return {
        "rr_intervals_sec": rr_intervals_sec,
        "rr_intervals_ms": rr_intervals_ms,
        "mean_rr_sec": mean_rr_sec,
        "mean_rr_ms": mean_rr_ms,
        "heart_rate_bpm": heart_rate_bpm,
        "heart_rate_std": heart_rate_std,
        "rr_min_ms": np.min(rr_intervals_ms),
        "rr_max_ms": np.max(rr_intervals_ms),
    }


def plot_rr_intervals(r_peaks, sampling_rate):
    """Plot RR interval statistics"""
    rr_analysis = calculate_rr_intervals(r_peaks, sampling_rate)

    if len(rr_analysis["rr_intervals_ms"]) == 0:
        print("Not enough R peaks for RR interval analysis")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # RR intervals timeline
    time_axis = np.arange(len(rr_analysis["rr_intervals_ms"]))
    axes[0, 0].plot(
        time_axis, rr_analysis["rr_intervals_ms"], "b-o", linewidth=1, markersize=4
    )
    axes[0, 0].axhline(
        rr_analysis["mean_rr_ms"],
        color="r",
        linestyle="--",
        label=f"Mean: {rr_analysis['mean_rr_ms']:.1f} ms",
    )
    axes[0, 0].set_title("RR Intervals Over Time", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Beat Number")
    axes[0, 0].set_ylabel("RR Interval (ms)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # RR intervals histogram
    axes[0, 1].hist(
        rr_analysis["rr_intervals_ms"],
        bins=20,
        color="green",
        alpha=0.7,
        edgecolor="black",
    )
    axes[0, 1].axvline(
        rr_analysis["mean_rr_ms"],
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {rr_analysis['mean_rr_ms']:.1f} ms",
    )
    axes[0, 1].set_title("RR Intervals Distribution", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("RR Interval (ms)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # Heart rate timeline
    heart_rates = 60000 / rr_analysis["rr_intervals_ms"]
    axes[1, 0].plot(time_axis, heart_rates, "r-o", linewidth=1, markersize=4)
    axes[1, 0].axhline(
        rr_analysis["heart_rate_bpm"],
        color="b",
        linestyle="--",
        label=f"Mean: {rr_analysis['heart_rate_bpm']:.1f} BPM",
    )
    axes[1, 0].set_title("Instantaneous Heart Rate", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Beat Number")
    axes[1, 0].set_ylabel("Heart Rate (BPM)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Statistics text box
    axes[1, 1].axis("off")
    stats_text = f"""RR Interval Statistics:

Mean RR Interval: {rr_analysis['mean_rr_ms']:.2f} ms
Min RR Interval: {rr_analysis['rr_min_ms']:.2f} ms
Max RR Interval: {rr_analysis['rr_max_ms']:.2f} ms

Heart Rate Statistics:

Mean Heart Rate: {rr_analysis['heart_rate_bpm']:.2f} BPM
Heart Rate Std Dev: {rr_analysis['heart_rate_std']:.2f} BPM
Total Beats Detected: {len(rr_analysis['rr_intervals_ms']) + 1}
"""
    axes[1, 1].text(
        0.1,
        0.5,
        stats_text,
        fontsize=11,
        family="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.show()
