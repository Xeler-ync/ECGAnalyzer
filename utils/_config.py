import json
import os

from ecgdetectors import Detectors


def _load_config():
    if not os.path.exists("./config/config.json")
        with open("./config/config.json", 'w+', encoding='utf-8') as f:
            json.dump({
                "path": "./PTB-XL",
                "sampling_rate": 100,
                "ecg_index": 8,
                "tgt_sampling_rate": 250,
                "plot_config": {
                    "fft_and_baseline_analysis": True,
                    "baseline_removal_comparison": True,
                    "r_peak_detection_comparison": True,
                    "evaluation_comparison": True,
                    "rr_intervals": True,
                    "heartbeats_overlay_original": True,
                    "heartbeats_overlay_normalized": True,
                    "single_heartbeat_normalized": True,
                    "heartbeat_evaluation_all": True,
                    "multiple_leads_normalized(V1,V2,V3)": True,
                    "multiple_leads_normalized(ALL)": True,
                    "original_vs_normalized_multiple_leads": True
                }
            }, f, ensure_ascii=True, separators=(',',':'))
    with open("./config/config.json") as f:
         return json.load(f)

_c = _load_config()

PATH = _c["path"] or "./PTB-XL/"
SAMPLING_RATE = _c["sampling_rate"] or 100
ECG_INDEX = _c["ecg_index"] or 8
TGT_SAMPLING_RATE = _c["target_sampling_rate"] or 250
DET = Detectors(TGT_SAMPLING_RATE)

PLOT_CONFIG = _c["plot_config"] or {
    "fft_and_baseline_analysis": True,
    "baseline_removal_comparison": True,
    "r_peak_detection_comparison": True,
    "evaluation_comparison": True,
    "rr_intervals": True,
    "heartbeats_overlay_original": True,
    "heartbeats_overlay_normalized": True,
    "single_heartbeat_normalized": True,
    "heartbeat_evaluation_all": True,
    "multiple_leads_normalized(V1,V2,V3)": True,
    "multiple_leads_normalized(ALL)": True,
    "original_vs_normalized_multiple_leads": True,
}