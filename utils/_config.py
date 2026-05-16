import json
import os
from dataclasses import dataclass
from typing import Optional, Union

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
        "bpm_comparision": True,
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
    "SCP_CODES": ["LVH"],
    "MIN_VALUES": [100],
    "random_forest_params": {
        "n_estimators": 600,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "min_weight_fraction_leaf": 0.0,
        "max_features": "sqrt",
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "bootstrap": True,
        "oob_score": True,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": 0,
        "warm_start": False,
        "class_weight": "balanced_subsample",
        "ccp_alpha": 0.0,
        "max_samples": None,
        "monotonic_cst": None,
    },
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


@dataclass
class RandomForestParams:
    n_estimators: int
    criterion: str
    max_depth: Optional[int]
    min_samples_split: int
    min_samples_leaf: int
    min_weight_fraction_leaf: float
    max_features: Union[str, int, float]
    max_leaf_nodes: Optional[int]
    min_impurity_decrease: float
    bootstrap: bool
    oob_score: bool
    n_jobs: int
    random_state: int
    verbose: int
    warm_start: bool
    class_weight: Optional[Union[str, dict]]
    ccp_alpha: float
    max_samples: Optional[Union[int, float]]
    monotonic_cst: Optional[list]

    def __post_init__(self):
        if isinstance(self.class_weight, dict):
            self.class_weight = {int(k): v for k, v in self.class_weight.items()}


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
SCP_CODES = _c.get("SCP_CODES", DEFAULT_CONFIG["SCP_CODES"])
MIN_VALUES = _c.get("MIN_VALUES", DEFAULT_CONFIG["MIN_VALUES"])

_user_params = _c.get("random_forest_params", {})
_merged = {**DEFAULT_CONFIG["random_forest_params"], **_user_params}
RANDOM_FOREST_PARAMS = RandomForestParams(**_merged)
