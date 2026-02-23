import numpy as np
import matplotlib.pyplot as plt


# Plot normalized heartbeats from multiple leads
def plot_multiple_leads_normalized(
    all_leads_normalized, lead_indices, lead_names, sampling_rate, max_beats=10
):
    """Plot normalized heartbeats from multiple leads side by side"""
    num_leads = len(lead_indices)
    fig, axes = plt.subplots(num_leads, 1, figsize=(14, 3 * num_leads))

    if num_leads == 1:
        axes = [axes]

    for plot_idx, lead_idx in enumerate(lead_indices):
        if (
            lead_idx not in all_leads_normalized
            or len(all_leads_normalized[lead_idx]) == 0
        ):
            continue

        normalized_hbs = all_leads_normalized[lead_idx]
        lead_name = lead_names[lead_idx]

        num_beats = min(len(normalized_hbs), max_beats)
        total_samples = len(normalized_hbs[0]["signal"])
        time_axis = np.arange(total_samples) / sampling_rate * 1000  # Convert to ms

        # Plot all heartbeats
        for i, hb in enumerate(normalized_hbs[:num_beats]):
            axes[plot_idx].plot(
                time_axis, hb["signal"], alpha=0.6, linewidth=1.5, label=f"Beat {i+1}"
            )

        # Mark R peak location
        r_peak_time = normalized_hbs[0]["r_peak_relative"] / sampling_rate * 1000
        axes[plot_idx].axvline(
            r_peak_time, color="red", linestyle="--", linewidth=2, alpha=0.8
        )

        axes[plot_idx].set_title(
            f"Lead {lead_name} - Aligned Normalized Heartbeats (First {num_beats} Beats)",
            fontsize=12,
            fontweight="bold",
        )
        axes[plot_idx].set_ylabel("Amplitude (mV)")
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (ms) - R Peak Aligned")
    plt.tight_layout()
    plt.show()


# Create a comparison plot of original vs normalized for multiple leads
def plot_original_vs_normalized_multiple_leads(
    all_leads_data,
    all_leads_normalized,
    lead_indices,
    lead_names,
    sampling_rate,
    max_beats=8,
):
    """Plot original and normalized heartbeats for multiple leads side by side"""
    num_leads = len(lead_indices)
    fig, axes = plt.subplots(num_leads, 2, figsize=(16, 3 * num_leads))

    if num_leads == 1:
        axes = [axes]

    for plot_idx, lead_idx in enumerate(lead_indices):
        if lead_idx not in all_leads_data:
            continue

        lead_name = lead_names[lead_idx]

        # Original heartbeats
        heartbeats = all_leads_data[lead_idx]
        num_beats = min(len(heartbeats), max_beats)

        for i, hb in enumerate(heartbeats[:num_beats]):
            time_axis = np.arange(len(hb["signal"])) / sampling_rate
            axes[plot_idx, 0].plot(
                time_axis, hb["signal"], alpha=0.6, label=f"Beat {i+1}"
            )

        axes[plot_idx, 0].set_title(
            f"Lead {lead_name} - Original (Unaligned)", fontsize=11, fontweight="bold"
        )
        axes[plot_idx, 0].set_ylabel("Amplitude (mV)")
        axes[plot_idx, 0].grid(True, alpha=0.3)
        axes[plot_idx, 0].legend(loc="upper right", fontsize=8)

        # Normalized heartbeats
        if lead_idx in all_leads_normalized:
            normalized_hbs = all_leads_normalized[lead_idx]
            total_samples = len(normalized_hbs[0]["signal"])
            time_axis = np.arange(total_samples) / sampling_rate * 1000

            for i, hb in enumerate(normalized_hbs[:num_beats]):
                axes[plot_idx, 1].plot(
                    time_axis, hb["signal"], alpha=0.6, label=f"Beat {i+1}"
                )

            r_peak_time = normalized_hbs[0]["r_peak_relative"] / sampling_rate * 1000
            axes[plot_idx, 1].axvline(
                r_peak_time, color="red", linestyle="--", linewidth=2, alpha=0.8
            )

            axes[plot_idx, 1].set_title(
                f"Lead {lead_name} - Normalized (R Peak Aligned)",
                fontsize=11,
                fontweight="bold",
            )
            axes[plot_idx, 1].set_ylabel("Amplitude (mV)")
            axes[plot_idx, 1].grid(True, alpha=0.3)
            axes[plot_idx, 1].legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("Time (seconds)")
    axes[-1, 1].set_xlabel("Time (ms)")
    plt.tight_layout()
    plt.show()
