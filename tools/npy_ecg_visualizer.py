import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from utils._config import LEAD_NAMES, SAMPLING_RATE


def visualize_single_ecg(npy_file, save_path=None):
    """
    Visualize a single ECG signal from a npy file.

    Parameters:
    - npy_file: Path to the npy file containing ECG signal
    - save_path: Path to save the visualization (optional)
    """
    # Load the ECG signal
    signal = np.load(npy_file)

    # Extract lead name from filename
    filename = os.path.basename(npy_file)
    parts = filename.split("_")
    ecg_id = parts[0]
    lead_name = parts[1]

    # Create time axis in milliseconds
    time_ms = np.arange(signal.shape[1]) * (1000 / SAMPLING_RATE)

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot each heartbeat with low opacity (使用灰色)
    for beat in signal:
        plt.plot(time_ms, beat, color="gray", alpha=0.2, linewidth=0.8)

    # Calculate and plot average heartbeat (使用蓝色)
    avg_beat = np.mean(signal, axis=0)
    std_beat = np.std(signal, axis=0)

    # Plot average heartbeat with higher opacity
    plt.plot(time_ms, avg_beat, color="blue", linewidth=2, label="Average")

    # Plot standard deviation area (使用蓝色)
    plt.fill_between(
        time_ms,
        avg_beat - std_beat,
        avg_beat + std_beat,
        color="blue",
        alpha=0.2,
        label="±1 SD",
    )

    # Add title and labels
    plt.title(
        f"ECG Signal - {ecg_id} - Lead {lead_name} (n={len(signal)} beats)", fontsize=14
    )
    plt.xlabel("Time (ms)", fontsize=12)
    plt.ylabel("Amplitude (mV)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # Save or show the figure
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_all_leads(ecg_id, directory, save_path=None):
    """
    Visualize all 12 leads of a specific ECG signal.

    Parameters:
    - ecg_id: ID of the ECG signal
    - directory: Directory containing the npy files
    - save_path: Path to save the visualization (optional)
    """
    # Create a 3x4 grid for the 12 leads
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    # Process each lead
    for lead_idx, lead_name in enumerate(LEAD_NAMES):
        # Construct the filename
        filename = f"{ecg_id}_{lead_name}_normalized_heartbeats.npy"
        filepath = os.path.join(directory, filename)

        # Check if file exists
        if not os.path.exists(filepath):
            axes[lead_idx].text(
                0.5,
                0.5,
                f"No data for {lead_name}",
                ha="center",
                va="center",
                transform=axes[lead_idx].transAxes,
            )
            axes[lead_idx].set_title(f"Lead {lead_name}")
            continue

        # Load the ECG signal
        signal = np.load(filepath)

        # Create time axis in milliseconds
        time_ms = np.arange(signal.shape[1]) * (1000 / SAMPLING_RATE)

        # Plot each heartbeat with low opacity (使用灰色)
        for beat in signal:
            axes[lead_idx].plot(time_ms, beat, color="gray", alpha=0.2, linewidth=0.8)

        # Calculate and plot average heartbeat (使用蓝色)
        avg_beat = np.mean(signal, axis=0)
        std_beat = np.std(signal, axis=0)

        # Plot average heartbeat with higher opacity
        axes[lead_idx].plot(time_ms, avg_beat, color="blue", linewidth=2)

        # Plot standard deviation area (使用蓝色)
        axes[lead_idx].fill_between(
            time_ms, avg_beat - std_beat, avg_beat + std_beat, color="blue", alpha=0.2
        )

        # Set title and labels
        axes[lead_idx].set_title(
            f"Lead {lead_name} (n={len(signal)})\nMax Std: {np.max(std_beat):.4f} | Avg Std: {np.mean(std_beat):.4f}"
        )
        axes[lead_idx].set_xlabel("Time (ms)")
        axes[lead_idx].set_ylabel("Amplitude (mV)")
        axes[lead_idx].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show the figure
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


def interactive_input():
    """
    Interactive input for the user when no command line arguments are provided.
    """
    print("=" * 80)
    print("ECG Signal Visualizer")
    print("=" * 80)

    # Ask for the npy file path
    npy_file = input("Enter the path to the npy file: ").strip()

    # Check if file exists
    if not os.path.exists(npy_file):
        print(f"Error: File '{npy_file}' does not exist.")
        return

    # Ask if user wants to visualize all leads
    all_leads_input = input("Visualize all 12 leads? (y/n): ").strip().lower()
    all_leads = all_leads_input == "y" or all_leads_input == "yes"

    # Ask if user wants to save the visualization
    save_input = input("Save the visualization? (y/n): ").strip().lower()
    save_path = None
    if save_input == "y" or save_input == "yes":
        save_path = input("Enter the path to save the visualization: ").strip()

    # Extract ECG ID and directory from the file path
    filename = os.path.basename(npy_file)
    parts = filename.split("_")
    ecg_id = parts[0]
    directory = os.path.dirname(npy_file)

    # Visualize based on user choice
    if all_leads:
        visualize_all_leads(ecg_id, directory, save_path)
    else:
        visualize_single_ecg(npy_file, save_path)


def main():
    parser = argparse.ArgumentParser(description="Visualize ECG signals from npy files")
    parser.add_argument("npy_file", nargs="?", help="Path to the npy file")
    parser.add_argument(
        "--all_leads",
        action="store_true",
        help="Visualize all 12 leads of the ECG signal",
    )
    parser.add_argument("--save", help="Path to save the visualization", default=None)

    args = parser.parse_args()

    # If no npy_file is provided, use interactive input
    if args.npy_file is None:
        interactive_input()
        return

    # Extract ECG ID and directory from the file path
    filename = os.path.basename(args.npy_file)
    parts = filename.split("_")
    ecg_id = parts[0]
    directory = os.path.dirname(args.npy_file)

    # Visualize based on user choice
    if args.all_leads:
        visualize_all_leads(ecg_id, directory, args.save)
    else:
        visualize_single_ecg(args.npy_file, args.save)


if __name__ == "__main__":
    main()
