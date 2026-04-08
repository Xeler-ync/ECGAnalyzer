import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

# Make sure repo root is on sys.path.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils._config import LEAD_NAMES, PATH, SAMPLING_RATE
from utils._data import Y, load_raw_data
from tools.ecg_heartbeat_normalization_pipeline import (
    process_ecg_signal,
    process_lead_with_r_peaks,
)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency at runtime
    plt = None

POSITIVE_LABEL = "MI"
NEGATIVE_LABEL = "NORM"
TARGET_THRESHOLD = 100
TARGET_BEAT_LEN = 64
LEFT_POINTS = 32
RIGHT_POINTS = 32
MIN_BEATS_PER_LEAD = 3
RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.20
DEFAULT_VAL_SIZE = 0.20
DEFAULT_N_ESTIMATORS = 600
DEFAULT_MIN_SAMPLES_LEAF = 2
DEFAULT_THRESHOLD_GRID = np.round(np.arange(0.10, 0.91, 0.02), 2)
DEFAULT_OUT_DIR = Path("results/random_forest_mi_vs_norm_fixed64_qc")
PLOT_OUT_DIR = DEFAULT_OUT_DIR
TARGET_NAMES = [NEGATIVE_LABEL, POSITIVE_LABEL]
MAD_SCALE = 1.4826
SUPERCLASS_ORDER = ["NORM", "HYP", "MI", "CD", "STTC"]


def ensure_dict(x: Any) -> dict:
    if isinstance(x, dict):
        return x
    if pd.isna(x):
        return {}
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        try:
            parsed = json.loads(x)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}



def find_scp_statements_file() -> Path | None:
    candidates = [
        Path(PATH) / "scp_statements.csv",
        Path(PATH).parent / "scp_statements.csv",
        REPO_ROOT / "scp_statements.csv",
        REPO_ROOT / "data" / "scp_statements.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None



def load_scp_statements() -> pd.DataFrame | None:
    p = find_scp_statements_file()
    if p is None:
        return None
    df = pd.read_csv(p)
    df.index = df["scp_code"] if "scp_code" in df.columns else df.iloc[:, 0]
    return df



def build_superclass_lookup(scp_df: pd.DataFrame | None) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if scp_df is None:
        return lookup

    candidate_cols = ["diagnostic_class", "diagnostic superclass", "diagnostic_superclass", "diagnostic"]
    super_col = None
    for col in candidate_cols:
        if col in scp_df.columns:
            super_col = col
            break
    if super_col is None:
        return lookup

    for code, row in scp_df.iterrows():
        value = row.get(super_col)
        if pd.notna(value) and str(value).strip():
            lookup[str(code)] = str(value).strip()
    return lookup



def map_superclasses(scp_codes: dict, superclass_lookup: dict[str, str], threshold: int = TARGET_THRESHOLD) -> list[str]:
    out = set()
    for code, score in scp_codes.items():
        try:
            is_positive = float(score) >= threshold
        except Exception:
            is_positive = False
        if is_positive:
            sc = superclass_lookup.get(code)
            if sc:
                out.add(sc)
    if scp_codes.get("NORM", 0) >= threshold:
        out.add("NORM")
    return sorted(out)


SCP_DF = load_scp_statements()
SUPERCLASS_LOOKUP = build_superclass_lookup(SCP_DF)



def resample_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) == 0:
        raise ValueError("Cannot resample empty signal")
    if len(x) == target_len:
        return x.astype(np.float32)
    old_axis = np.linspace(0.0, 1.0, len(x))
    new_axis = np.linspace(0.0, 1.0, target_len)
    return np.interp(new_axis, old_axis, x).astype(np.float32)



def is_pure_norm(scp_codes: dict, threshold: int = TARGET_THRESHOLD) -> bool:
    if not isinstance(scp_codes, dict):
        return False
    if scp_codes.get(NEGATIVE_LABEL, 0) < threshold:
        return False
    other_positive_codes = [
        code for code, score in scp_codes.items() if code != NEGATIVE_LABEL and score >= threshold
    ]
    return len(other_positive_codes) == 0



def is_strict_mi(scp_codes: dict, threshold: int = TARGET_THRESHOLD) -> tuple[bool, list[str]]:
    """
    Strict MI definition for the first MI baseline:
      - At least one SCP code maps to superclass MI
      - The set of diagnostic superclasses is exactly {'MI'}
      - Mixed superclass records are excluded
    """
    if not isinstance(scp_codes, dict):
        return False, []
    superclasses = map_superclasses(scp_codes, SUPERCLASS_LOOKUP, threshold=threshold)
    return (len(superclasses) == 1 and superclasses[0] == POSITIVE_LABEL), superclasses



def get_binary_label_mi_norm(scp_codes: dict, threshold: int = TARGET_THRESHOLD):
    if not isinstance(scp_codes, dict):
        return None, [], False, False
    is_mi, superclasses = is_strict_mi(scp_codes, threshold)
    is_norm = is_pure_norm(scp_codes, threshold)
    if is_mi:
        return 1, superclasses, True, False
    if is_norm:
        return 0, superclasses if superclasses else [NEGATIVE_LABEL], False, True
    return None, superclasses, False, False



def infer_r_index_from_beat(beat: dict, signal: np.ndarray) -> int:
    candidate_keys = [
        "r_peak_relative",
        "r_peak_idx",
        "r_idx",
        "r_peak",
        "center_idx",
        "peak_idx",
    ]
    for key in candidate_keys:
        if key in beat and beat[key] is not None:
            try:
                idx = int(round(float(beat[key])))
                return max(0, min(idx, len(signal) - 1))
            except Exception:
                pass
    return len(signal) // 2



def normalize_heartbeat_fixed64(beat: dict) -> np.ndarray:
    """
    Re-normalize a heartbeat into 64 points using separate linear interpolation
    before and after the R peak:
      - left side (including R): 32 points
      - right side (excluding duplicated R): 32 points
    """
    if "signal" not in beat:
        raise ValueError("Heartbeat dict does not contain 'signal'")

    signal = np.asarray(beat["signal"], dtype=float).reshape(-1)
    if len(signal) < 4:
        raise ValueError("Heartbeat too short")

    r_idx = infer_r_index_from_beat(beat, signal)
    left = signal[: r_idx + 1]
    right = signal[r_idx + 1 :]

    if len(left) < 2:
        raise ValueError("Heartbeat left side too short around R peak")
    if len(right) < 2:
        raise ValueError("Heartbeat right side too short around R peak")

    left_rs = resample_1d(left, LEFT_POINTS)
    right_rs = resample_1d(right, RIGHT_POINTS)
    out = np.concatenate([left_rs, right_rs]).astype(np.float32)
    if len(out) != TARGET_BEAT_LEN:
        raise ValueError(f"Expected 64 points, got {len(out)}")
    return out



def robust_outlier_filter(beats_2d: np.ndarray, mad_k: float = 3.5) -> tuple[np.ndarray, np.ndarray, float]:
    beats_2d = np.asarray(beats_2d, dtype=float)
    if beats_2d.ndim != 2 or beats_2d.shape[0] == 0:
        raise ValueError("beats_2d must have shape [n_beats, 64]")
    if beats_2d.shape[0] <= 2:
        keep_mask = np.ones(beats_2d.shape[0], dtype=bool)
        return beats_2d.astype(np.float32), keep_mask, float("inf")

    center = np.median(beats_2d, axis=0)
    distances = np.mean(np.abs(beats_2d - center), axis=1)
    med = float(np.median(distances))
    mad = float(np.median(np.abs(distances - med)))

    if mad < 1e-12:
        spread = float(np.std(distances))
        threshold = med + 3.0 * spread
    else:
        threshold = med + mad_k * MAD_SCALE * mad

    keep_mask = distances <= threshold
    if not np.any(keep_mask):
        keep_mask[np.argmin(distances)] = True
    return beats_2d[keep_mask].astype(np.float32), keep_mask, float(threshold)



def normalize_mode_apply(x: np.ndarray, normalize_mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if normalize_mode == "none":
        return x.astype(np.float32)
    if normalize_mode == "zscore":
        mu = float(x.mean())
        sigma = float(x.std())
        if sigma < 1e-8:
            return (x - mu).astype(np.float32)
        return ((x - mu) / sigma).astype(np.float32)
    raise ValueError(f"Unknown normalize_mode: {normalize_mode}")



def summarize_lead_beats(normalized_heartbeats: list[dict], normalize_mode: str = "none") -> dict[str, Any]:
    fixed_beats = []
    rejected_short = 0
    for beat in normalized_heartbeats:
        try:
            fixed_beats.append(normalize_heartbeat_fixed64(beat))
        except Exception:
            rejected_short += 1

    if len(fixed_beats) < MIN_BEATS_PER_LEAD:
        raise ValueError(f"Too few valid fixed64 beats after normalization: {len(fixed_beats)}")

    beats_2d = np.vstack(fixed_beats)
    kept_beats, keep_mask, qc_threshold = robust_outlier_filter(beats_2d)
    if kept_beats.shape[0] < MIN_BEATS_PER_LEAD:
        raise ValueError(f"Too few beats left after QC: {kept_beats.shape[0]}")

    mean_beat = kept_beats.mean(axis=0)
    mean_beat = normalize_mode_apply(mean_beat, normalize_mode)
    beat_std = kept_beats.std(axis=0)

    return {
        "mean_beat": mean_beat.astype(np.float32),
        "all_beats": beats_2d.astype(np.float32),
        "kept_beats": kept_beats.astype(np.float32),
        "n_input_beats": int(len(normalized_heartbeats)),
        "n_valid_fixed64": int(beats_2d.shape[0]),
        "n_kept_after_qc": int(kept_beats.shape[0]),
        "n_removed_as_outliers": int((~keep_mask).sum()),
        "n_rejected_short": int(rejected_short),
        "mean_std": float(np.mean(beat_std)),
        "qc_threshold": float(qc_threshold),
    }



def build_feature_for_record(
    signal_index: int,
    normalize_mode: str = "none",
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    raw = load_raw_data(Y, SAMPLING_RATE, PATH, signal_index)
    ecg = raw[0]  # [time, 12]

    lead_ii_idx = 1
    results_ii = process_ecg_signal(ecg[:, lead_ii_idx], lead_ii_idx)
    r_peaks = np.asarray(results_ii["r_peaks"], dtype=int)
    if len(r_peaks) < 3:
        raise ValueError("Too few detected R peaks")

    features = []
    lead_summaries = []
    for lead_idx, lead_name in enumerate(LEAD_NAMES):
        if lead_idx == lead_ii_idx:
            normalized_beats = results_ii["normalized_heartbeats"]
        else:
            normalized_beats = process_lead_with_r_peaks(ecg[:, lead_idx], r_peaks, lead_idx)

        lead_summary = summarize_lead_beats(normalized_beats, normalize_mode=normalize_mode)
        lead_summary["lead_name"] = lead_name
        lead_summary["lead_idx"] = int(lead_idx)
        features.append(lead_summary["mean_beat"])
        lead_summaries.append(lead_summary)

    feature_3d = np.stack(features, axis=0).astype(np.float32)
    qc_meta = {
        "detected_r_peaks": int(len(r_peaks)),
        "lead_min_valid_fixed64": int(min(s["n_valid_fixed64"] for s in lead_summaries)),
        "lead_min_kept_after_qc": int(min(s["n_kept_after_qc"] for s in lead_summaries)),
        "lead_mean_kept_after_qc": float(np.mean([s["n_kept_after_qc"] for s in lead_summaries])),
        "total_removed_as_outliers": int(sum(s["n_removed_as_outliers"] for s in lead_summaries)),
        "mean_alignment_std": float(np.mean([s["mean_std"] for s in lead_summaries])),
    }
    return feature_3d, qc_meta, ecg, lead_summaries



def save_qc_plot(
    out_dir: Path,
    signal_index: int,
    ecg: np.ndarray,
    lead_summaries: list[dict[str, Any]],
    label_name: str,
    lead_name: str = "II",
):
    if plt is None:
        return
    lead_idx = LEAD_NAMES.index(lead_name)
    lead_summary = lead_summaries[lead_idx]
    raw_signal = ecg[:, lead_idx]
    all_beats = lead_summary["all_beats"]
    kept_beats = lead_summary["kept_beats"]
    mean_beat = lead_summary["mean_beat"]

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.0, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(raw_signal)
    ax1.set_title(f"Raw ECG Lead {lead_name} | record={signal_index} | label={label_name}")
    ax1.set_xlabel("Time sample")
    ax1.set_ylabel("Amplitude")

    ax2 = fig.add_subplot(gs[1, 0])
    for beat in all_beats[: min(20, len(all_beats))]:
        ax2.plot(beat, alpha=0.25)
    ax2.set_title(f"Fixed64 normalized beats before QC (n={len(all_beats)})")
    ax2.set_xlabel("Normalized sample (0-63)")
    ax2.set_ylabel("Amplitude")

    ax3 = fig.add_subplot(gs[2, 0])
    for beat in kept_beats[: min(20, len(kept_beats))]:
        ax3.plot(beat, alpha=0.20)
    ax3.plot(mean_beat, linewidth=2.5)
    ax3.set_title(f"Beats after QC + average heartbeat (kept={len(kept_beats)})")
    ax3.set_xlabel("Normalized sample (0-63)")
    ax3.set_ylabel("Amplitude")

    fig.tight_layout()
    plot_dir = out_dir / "qc_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"record_{signal_index}_{label_name}_lead_{lead_name}.png", dpi=150)
    plt.close(fig)



def build_dataset(
    max_records: int | None = None,
    normalize_mode: str = "none",
    save_qc_plots: int = 0,
):
    X_3d_list, y_list, patient_ids = [], [], []
    metadata_rows = []
    skipped = []

    total_records = len(Y) if max_records is None else min(max_records, len(Y))
    qc_plots_written = 0
    out_dir_for_plots = PLOT_OUT_DIR

    for signal_index in tqdm(range(total_records), desc="Building RF dataset"):
        try:
            scp_codes = ensure_dict(Y.scp_codes.iloc[signal_index])
            label, superclasses, is_strict_mi_label, is_pure_norm_label = get_binary_label_mi_norm(
                scp_codes,
                threshold=TARGET_THRESHOLD,
            )
            if label is None:
                skipped.append((signal_index, "not strict MI or pure NORM"))
                continue

            feat_3d, qc_meta, ecg, lead_summaries = build_feature_for_record(
                signal_index,
                normalize_mode=normalize_mode,
            )

            patient_id = Y.patient_id.iloc[signal_index]
            ecg_id = Y.index[signal_index]
            label_name = POSITIVE_LABEL if label == 1 else NEGATIVE_LABEL

            X_3d_list.append(feat_3d)
            y_list.append(label)
            patient_ids.append(patient_id)
            metadata_rows.append(
                {
                    "ecg_id": ecg_id,
                    "patient_id": patient_id,
                    "label": label,
                    "label_name": label_name,
                    "strict_mi": int(is_strict_mi_label),
                    "pure_norm": int(is_pure_norm_label),
                    "diagnostic_superclasses": ";".join(superclasses),
                    "n_superclasses": int(len(superclasses)),
                    "raw_scp_codes": json.dumps(scp_codes, ensure_ascii=False),
                    **qc_meta,
                }
            )

            if save_qc_plots > 0 and qc_plots_written < save_qc_plots:
                save_qc_plot(out_dir_for_plots, signal_index, ecg, lead_summaries, label_name)
                qc_plots_written += 1

        except Exception as e:
            skipped.append((signal_index, str(e)))

    if not X_3d_list:
        raise RuntimeError("No usable ECG records were produced. Check dataset path/config.")

    X_3d = np.stack(X_3d_list, axis=0).astype(np.float32)
    X_flat = X_3d.reshape(X_3d.shape[0], -1).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int32)
    groups = np.asarray(patient_ids)

    meta = pd.DataFrame(metadata_rows)
    skipped_df = pd.DataFrame(skipped, columns=["signal_index", "reason"])
    return X_3d, X_flat, y, groups, meta, skipped_df



def save_processed_dataset(
    out_dir: Path,
    X_3d: np.ndarray,
    X_flat: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    skipped_df: pd.DataFrame,
    normalize_mode: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_3d.npy", X_3d)
    np.save(out_dir / "X_flat.npy", X_flat)
    np.save(out_dir / "y.npy", y)
    meta.to_csv(out_dir / "metadata.csv", index=False)
    skipped_df.to_csv(out_dir / "skipped_records.csv", index=False)

    summary = {
        "n_samples": int(len(y)),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "positive_label": POSITIVE_LABEL,
        "negative_label": NEGATIVE_LABEL,
        "threshold": TARGET_THRESHOLD,
        "strict_positive_definition": "single diagnostic superclass only: MI",
        "negative_definition": "pure NORM only",
        "x_3d_shape": list(X_3d.shape),
        "x_flat_shape": list(X_flat.shape),
        "normalize_mode": normalize_mode,
        "fixed64_rule": {
            "left_points": LEFT_POINTS,
            "right_points": RIGHT_POINTS,
            "duplicate_r_peak": False,
        },
    }
    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    qc_summary = {
        "detected_r_peaks_mean": float(meta["detected_r_peaks"].mean()) if len(meta) else float("nan"),
        "lead_min_kept_after_qc_mean": float(meta["lead_min_kept_after_qc"].mean()) if len(meta) else float("nan"),
        "lead_mean_kept_after_qc_mean": float(meta["lead_mean_kept_after_qc"].mean()) if len(meta) else float("nan"),
        "total_removed_as_outliers_mean": float(meta["total_removed_as_outliers"].mean()) if len(meta) else float("nan"),
        "mean_alignment_std_mean": float(meta["mean_alignment_std"].mean()) if len(meta) else float("nan"),
    }
    with open(out_dir / "qc_summary.json", "w", encoding="utf-8") as f:
        json.dump(qc_summary, f, indent=2, ensure_ascii=False)



def load_processed_dataset(out_dir: Path):
    X_3d = np.load(out_dir / "X_3d.npy")
    X_flat = np.load(out_dir / "X_flat.npy")
    y = np.load(out_dir / "y.npy")
    meta = pd.read_csv(out_dir / "metadata.csv")
    groups = meta["patient_id"].to_numpy()
    skipped_path = out_dir / "skipped_records.csv"
    if skipped_path.exists():
        skipped_df = pd.read_csv(skipped_path)
    else:
        skipped_df = pd.DataFrame(columns=["signal_index", "reason"])
    return X_3d, X_flat, y, groups, meta, skipped_df



def _has_both_classes(y_part: np.ndarray) -> bool:
    return len(np.unique(y_part)) == 2



def grouped_split_indices(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = RANDOM_STATE,
    max_tries: int = 30,
):
    for offset in range(max_tries):
        rs = random_state + offset
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        train_val_idx, test_idx = next(gss_test.split(X, y, groups=groups))

        train_val_groups = groups[train_val_idx]
        train_val_y = y[train_val_idx]
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=rs + 1000)
        rel_train_idx, rel_val_idx = next(gss_val.split(X[train_val_idx], train_val_y, groups=train_val_groups))

        train_idx = train_val_idx[rel_train_idx]
        val_idx = train_val_idx[rel_val_idx]
        if _has_both_classes(y[train_idx]) and _has_both_classes(y[val_idx]) and _has_both_classes(y[test_idx]):
            return train_idx, val_idx, test_idx
    raise RuntimeError("Failed to create group-aware splits with both classes in each split")



def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
    }



def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold_grid=DEFAULT_THRESHOLD_GRID):
    rows = []
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in threshold_grid:
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        rows.append({"threshold": float(threshold), "precision": float(precision), "recall": float(recall), "f1": float(f1)})
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold, pd.DataFrame(rows)



def train_and_evaluate(
    X_flat: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
):
    train_idx, val_idx, test_idx = grouped_split_indices(X_flat, y, groups)
    X_train, X_val, X_test = X_flat[train_idx], X_flat[val_idx], X_flat[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        oob_score=True,
    )
    clf.fit(X_train, y_train)

    val_prob = clf.predict_proba(X_val)[:, 1]
    best_threshold, threshold_df = find_best_threshold(y_val, val_prob)

    test_prob = clf.predict_proba(X_test)[:, 1]
    test_pred_default = (test_prob >= 0.5).astype(int)
    test_pred_tuned = (test_prob >= best_threshold).astype(int)

    default_metrics = compute_binary_metrics(y_test, test_pred_default, test_prob)
    tuned_metrics = compute_binary_metrics(y_test, test_pred_tuned, test_prob)

    split_info = {
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "positive_train": int(y_train.sum()),
        "positive_val": int(y_val.sum()),
        "positive_test": int(y_test.sum()),
        "negative_train": int((y_train == 0).sum()),
        "negative_val": int((y_val == 0).sum()),
        "negative_test": int((y_test == 0).sum()),
        "oob_score": float(clf.oob_score_),
        "best_threshold_from_val_f1": float(best_threshold),
    }

    default_cm = confusion_matrix(y_test, test_pred_default, labels=[0, 1])
    tuned_cm = confusion_matrix(y_test, test_pred_tuned, labels=[0, 1])

    default_report = classification_report(y_test, test_pred_default, labels=[0, 1], target_names=TARGET_NAMES, digits=4, zero_division=0)
    tuned_report = classification_report(y_test, test_pred_tuned, labels=[0, 1], target_names=TARGET_NAMES, digits=4, zero_division=0)

    lead_importance = clf.feature_importances_.reshape(len(LEAD_NAMES), TARGET_BEAT_LEN).sum(axis=1)
    lead_importance_df = pd.DataFrame({"lead": LEAD_NAMES, "importance": lead_importance}).sort_values("importance", ascending=False)

    return {
        "model": clf,
        "split_info": split_info,
        "threshold_df": threshold_df,
        "default_metrics": default_metrics,
        "tuned_metrics": tuned_metrics,
        "default_cm": default_cm,
        "tuned_cm": tuned_cm,
        "default_report": default_report,
        "tuned_report": tuned_report,
        "lead_importance_df": lead_importance_df,
    }



def save_training_outputs(out_dir: Path, results: dict):
    metrics_rows = []
    for setting_name, metric_dict in [("default_0.50", results["default_metrics"]), ("tuned_threshold", results["tuned_metrics"] )]:
        row = {"setting": setting_name}
        row.update(results["split_info"])
        row.update(metric_dict)
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics_comparison.csv", index=False)
    results["threshold_df"].to_csv(out_dir / "validation_threshold_search.csv", index=False)
    results["lead_importance_df"].to_csv(out_dir / "lead_importance.csv", index=False)
    np.savetxt(out_dir / "confusion_matrix_default.txt", results["default_cm"], fmt="%d")
    np.savetxt(out_dir / "confusion_matrix_tuned.txt", results["tuned_cm"], fmt="%d")
    (out_dir / "classification_report_default.txt").write_text(results["default_report"], encoding="utf-8")
    (out_dir / "classification_report_tuned.txt").write_text(results["tuned_report"], encoding="utf-8")



def print_metrics_block(title: str, metrics: dict):
    print(title)
    for k, v in metrics.items():
        print(f"- {k}: {v}")
    print()



def main():
    parser = argparse.ArgumentParser(description="Random Forest for strict MI vs pure NORM with fixed64 + QC")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--normalize-mode", type=str, choices=["none", "zscore"], default="none")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--min-samples-leaf", type=int, default=DEFAULT_MIN_SAMPLES_LEAF)
    parser.add_argument("--save-qc-plots", type=int, default=0, help="Save this many QC example plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    global PLOT_OUT_DIR
    PLOT_OUT_DIR = out_dir

    dataset_files = [out_dir / "X_3d.npy", out_dir / "X_flat.npy", out_dir / "y.npy", out_dir / "metadata.csv"]

    print("=" * 80)
    print("Random Forest for strict MI vs pure NORM classification (fixed64 + QC)")
    print("=" * 80)
    print(f"Positive class     : strict {POSITIVE_LABEL} (single diagnostic superclass only)")
    print(f"Negative class     : pure {NEGATIVE_LABEL} >= {TARGET_THRESHOLD}")
    print("Other records      : excluded")
    print(f"Heartbeat rule     : {LEFT_POINTS} samples before/including R + {RIGHT_POINTS} after R")
    print(f"Beat feature       : {len(LEAD_NAMES)} leads x {TARGET_BEAT_LEN} samples")
    print(f"Normalize mode     : {args.normalize_mode}")
    print("QC rule            : robust beat outlier removal per lead")
    print("Split strategy     : GroupShuffleSplit by patient_id")
    print("Threshold strategy : tune on validation set using best F1")
    print()

    if args.rebuild_dataset or not all(p.exists() for p in dataset_files):
        print("Building processed dataset from raw ECG files...")
        X_3d, X_flat, y, groups, meta, skipped_df = build_dataset(
            max_records=args.max_records,
            normalize_mode=args.normalize_mode,
            save_qc_plots=args.save_qc_plots,
        )
        save_processed_dataset(out_dir, X_3d, X_flat, y, meta, skipped_df, args.normalize_mode)
    else:
        print("Loading cached processed dataset...")
        X_3d, X_flat, y, groups, meta, skipped_df = load_processed_dataset(out_dir)

    print(f"Usable records      : {len(y)}")
    print(f"Skipped records     : {len(skipped_df)}")
    print(f"MI samples          : {int((y == 1).sum())}")
    print(f"NORM samples        : {int((y == 0).sum())}")
    if len(meta):
        print(f"Mean detected R     : {meta['detected_r_peaks'].mean():.2f}")
        print(f"Mean kept beats     : {meta['lead_mean_kept_after_qc'].mean():.2f}")
        print(f"Mean removed outlier beats : {meta['total_removed_as_outliers'].mean():.2f}")
    print(f"Saved dataset path  : {out_dir.resolve()}")
    print()

    if len(np.unique(y)) < 2:
        raise RuntimeError("Dataset contains fewer than 2 classes after filtering. Cannot train binary classifier.")

    if args.build_only:
        print("Dataset build completed. Training skipped because --build-only was set.")
        return

    results = train_and_evaluate(X_flat, y, groups, n_estimators=args.n_estimators, min_samples_leaf=args.min_samples_leaf)
    save_training_outputs(out_dir, results)

    print("Split info")
    for k, v in results["split_info"].items():
        print(f"- {k}: {v}")
    print()

    print_metrics_block("Test metrics @ default threshold = 0.50", results["default_metrics"])
    print_metrics_block(f"Test metrics @ tuned threshold = {results['split_info']['best_threshold_from_val_f1']:.2f}", results["tuned_metrics"])

    print(f"Confusion matrix @ default threshold (rows=true, cols=pred; order={TARGET_NAMES})")
    print(results["default_cm"])
    print()

    print(f"Confusion matrix @ tuned threshold (rows=true, cols=pred; order={TARGET_NAMES})")
    print(results["tuned_cm"])
    print()

    print("Lead importance (summed over 64 points)")
    for _, row in results["lead_importance_df"].iterrows():
        print(f"- {row['lead']}: {row['importance']:.6f}")
    print()
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
