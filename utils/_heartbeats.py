import pandas as pd
import numpy as np
import wfdb
from wfdb import processing as wfdb_processing
import neurokit2 as nk
import ast
import matplotlib.pyplot as plt
from scipy import signal as signal
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from ecgdetectors import Detectors


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
    Normalize heartbeats by aligning to R without using fixed pre/post durations.

    - Remove fixed pre_r_duration_ms and post_r_duration_ms inputs.
    - Determine target pre/post lengths from the minimum available samples across all beats.
    - Align by truncation/padding (no interpolation) to preserve original sampling.
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
        pre_len = max(1, rpos + 1)
        post_len = max(1, sig_len - rpos)
        pre_lengths.append(pre_len)
        post_lengths.append(post_len)

    # choose minimal observed lengths
    target_pre = max(1, int(min(pre_lengths)))
    target_post = max(1, int(min(post_lengths)))
    total_samples = target_pre + target_post

    for hb in heartbeats:
        sig = hb["signal"]
        rpos = hb["r_peak_relative"]

        # Pre: right-align to R (take last target_pre samples ending at rpos)
        start_pre = max(0, rpos + 1 - target_pre)
        pre_seg = sig[start_pre : rpos + 1]

        # Post: left-align at R (take first target_post samples starting at rpos)
        end_post = min(len(sig), rpos + target_post)
        post_seg = sig[rpos:end_post]

        # Pad with edge values if shorter than target
        if len(pre_seg) < target_pre:
            pad_val = pre_seg[0] if len(pre_seg) > 0 else 0.0
            pre_seg = np.concatenate(
                [np.full(target_pre - len(pre_seg), pad_val), pre_seg]
            )
        if len(post_seg) < target_post:
            pad_val = post_seg[-1] if len(post_seg) > 0 else 0.0
            post_seg = np.concatenate(
                [post_seg, np.full(target_post - len(post_seg), pad_val)]
            )

        # Combine without duplicating the R sample (R at index target_pre-1)
        if len(pre_seg) > 1:
            normalized = np.concatenate([pre_seg[:-1], post_seg])
        else:
            normalized = np.concatenate([pre_seg, post_seg])

        normalized_heartbeats.append(
            {
                "signal": normalized,
                "original_duration_ms": hb["duration_ms"],
                "normalized_duration_ms": (total_samples / sampling_rate) * 1000,
                "r_peak_relative": target_pre - 1,
                "pre_r_samples": target_pre,
                "post_r_samples": target_post,
                "original_r_peak_index": hb["r_peak_index"],
            }
        )

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


def plot_single_heartbeat_normalized(
    normalized_heartbeat, sampling_rate, lead_name, beat_number
):
    """Plot a single normalized heartbeat with R peak marked"""
    fig, ax = plt.subplots(figsize=(12, 6))

    total_samples = len(normalized_heartbeat["signal"])
    time_axis = np.arange(total_samples) / sampling_rate * 1000  # Convert to ms

    ax.plot(
        time_axis, normalized_heartbeat["signal"], "b-", linewidth=2, label="ECG Signal"
    )
    ax.plot(
        time_axis[normalized_heartbeat["r_peak_relative"]],
        normalized_heartbeat["signal"][normalized_heartbeat["r_peak_relative"]],
        "ro",
        markersize=10,
        label="R Peak",
    )

    ax.set_title(
        f"Single Normalized Heartbeat - Lead {lead_name} (Beat #{beat_number})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (mV)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add annotations
    pre_r_time = normalized_heartbeat["pre_r_samples"] / sampling_rate * 1000
    post_r_time = normalized_heartbeat["post_r_samples"] / sampling_rate * 1000
    ax.text(
        0.02,
        0.95,
        f"Pre-R: {pre_r_time:.2f} ms | Post-R: {post_r_time:.2f} ms\nTotal Duration: {normalized_heartbeat['normalized_duration_ms']:.2f} ms",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.show()
