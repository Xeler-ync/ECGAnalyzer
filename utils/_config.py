import json
import os

from ecgdetectors import Detectors

DEFAULT_CONFIG = {
    "path": "./PTB-XL/",
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
        "original_vs_normalized_multiple_leads": True,
    },
    "results_path": "./results/",
    "max_workers": 4,
}

LEAD_NAMES = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

def _load_config():
    if not os.path.exists("./config/config.json"):
        os.makedirs("./config", exist_ok=True)
        with open("./config/config.json", "w+", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=True, separators=(",", ":"))
    with open("./config/config.json") as f:
        try:
            return json.load(f)
        except json.decoder.JSONDecodeError:
            return {}


_c = _load_config()

PATH = _c.get("path", DEFAULT_CONFIG["path"])
SAMPLING_RATE = _c.get("sampling_rate", DEFAULT_CONFIG["sampling_rate"])
ECG_INDEX = _c.get("ecg_index", DEFAULT_CONFIG["ecg_index"])
TGT_SAMPLING_RATE = _c.get("tgt_sampling_rate", DEFAULT_CONFIG["tgt_sampling_rate"])
DET = Detectors(TGT_SAMPLING_RATE)

PLOT_CONFIG = _c.get("plot_config", DEFAULT_CONFIG["plot_config"])
RESULTS_PATH = "./results/"
os.makedirs(RESULTS_PATH, exist_ok=True)
MAX_WORKERS = _c.get("max_workers", DEFAULT_CONFIG["max_workers"])