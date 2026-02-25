from ecgdetectors import Detectors

PATH = "./PTB-XL/"
SAMPLING_RATE = 100
ECG_INDEX = 8
TGT_SAMPLING_RATE = 250
DET = Detectors(TGT_SAMPLING_RATE)

PLOT_CONFIG = {
    "fft_and_baseline_analysis": True,
    "baseline_removal_comparison": True,
    "r_peak_detection_comparison": True,
    "evaluation_comparison": True,
    "rr_intervals": True,
    "heartbeats_overlay_original": True,
    "heartbeats_overlay_normalized": True,
    "single_heartbeat_normalized": True,
    "multiple_leads_normalized(V1,V2,V3)": True,
    "multiple_leads_normalized(ALL)": True,
    "original_vs_normalized_multiple_leads": True,
}