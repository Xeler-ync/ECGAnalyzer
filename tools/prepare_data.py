import sys
import os.path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import ast
import contextlib
import json
from pathlib import Path
from typing import List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import concurrent.futures
from tqdm import tqdm

from utils._baseline import remove_baseline_wander_hp_filter
from utils._r_peaks import detect_r_peaks_envelope, detect_r_peaks_neurokit_NeuroKit2
from utils._config import (
    SAMPLING_RATE,
    RESULTS_PATH,
    MAX_WORKERS,
    PATH,
    LEAD_NAMES,
)
from utils._data import load_raw_data, Y
from utils._heartbeats import extract_heartbeats, split_and_resample_heartbeats
from utils._helpers import _round_and_clip_indices

TARGET_THRESHOLD = 100
SUPERCLASS_ORDER = ["NORM", "HYP", "MI", "CD", "STTC"]
REPO_ROOT = Path(__file__).resolve().parent

# Heartbeat resampling parameters
TARGET_HEARTBEAT_LEN = 65  # final number of points per heartbeat

# List of irregular ECG signal indices
with open("./data/irregular_indices.json", "r") as f:
    irregular = json.load(f)["irregular"]


def ensure_dict(x: Any) -> dict:
    if isinstance(x, dict):
        return x
    if pd.isna(x):
        return {}
    if isinstance(x, str):
        with contextlib.suppress(Exception):
            parsed = ast.literal_eval(x)
            if isinstance(parsed, dict):
                return parsed
        with contextlib.suppress(Exception):
            parsed = json.loads(x)
            if isinstance(parsed, dict):
                return parsed
    return {}


def is_pure_norm(scp_codes: dict, threshold: int = TARGET_THRESHOLD) -> bool:
    return is_pure_negative(scp_codes, negative_code="NORM", threshold=threshold)


def is_pure_negative(
    scp_codes: dict, negative_code: str, threshold: int = TARGET_THRESHOLD
) -> bool:
    """Check if record is pure negative using SCP code (only negative_code with score >= threshold, no other positive codes)."""
    if scp_codes.get(negative_code, 0) < threshold:
        return False
    other_positive = any(
        code != negative_code and score >= threshold
        for code, score in scp_codes.items()
    )
    return not other_positive


def is_pure_negative_by_superclass(
    scp_codes: dict,
    superclass_lookup: dict,
    negative_superclass: str,
    threshold: int = TARGET_THRESHOLD,
) -> bool:
    """Check if record is pure negative using superclass: only negative_superclass present and no other superclass."""
    superclasses = map_superclasses(scp_codes, superclass_lookup, threshold)
    return superclasses == [negative_superclass]


def find_scp_statements_file() -> Path | None:
    candidates = [
        Path(PATH) / "scp_statements.csv",
        Path(PATH).parent / "scp_statements.csv",
        REPO_ROOT / "scp_statements.csv",
        REPO_ROOT / "data" / "scp_statements.csv",
    ]
    return next((c for c in candidates if c.exists()), None)


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

    candidate_cols = [
        "diagnostic_class",
        "diagnostic superclass",
        "diagnostic_superclass",
        "diagnostic",
    ]
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


def map_superclasses(
    scp_codes: dict,
    superclass_lookup: dict[str, str],
    threshold: int = TARGET_THRESHOLD,
) -> list[str]:
    out = set()
    for code, score in scp_codes.items():
        if score >= threshold:
            if sc := superclass_lookup.get(code):
                out.add(sc)
    if scp_codes.get("NORM", 0) >= threshold:
        out.add("NORM")
    return sorted(out)


def filter_others_mixed(
    positive_val: str,
    positive_is_superclass: bool,
    negative_val: str,
    negative_is_superclass: bool,
    threshold: int = TARGET_THRESHOLD,
) -> List[Tuple[int, str, dict]]:
    """
    Mixed OTHERS filter.
    - positive exclusion: if positive_is_superclass, exclude records whose superclasses contain positive_val;
      else exclude records with SCP code positive_val >= threshold.
    - negative exclusion: if negative_is_superclass, exclude records that are pure negative by superclass (only negative_val superclass);
      else exclude records that are pure negative by SCP code (only negative_val code with score >= threshold and no other positive codes).
    - Keep records that are NOT positive-excluded, NOT negative-excluded, and have at least one positive SCP code (any).
    """
    scp_df = load_scp_statements()
    superclass_lookup = build_superclass_lookup(scp_df)

    indices = []
    for idx in range(len(Y)):
        scp_codes = ensure_dict(Y.scp_codes.iloc[idx])
        any_positive = any(score >= threshold for score in scp_codes.values())
        if not any_positive:
            continue

        # Positive exclusion
        is_positive = False
        if positive_is_superclass:
            superclasses = map_superclasses(scp_codes, superclass_lookup, threshold)
            is_positive = positive_val in superclasses
        else:
            is_positive = scp_codes.get(positive_val, 0) >= threshold

        # Negative exclusion (pure negative)
        is_pure_neg = False
        if negative_is_superclass:
            is_pure_neg = is_pure_negative_by_superclass(
                scp_codes, superclass_lookup, negative_val, threshold
            )
        else:
            is_pure_neg = is_pure_negative(scp_codes, negative_val, threshold)

        if not is_positive and not is_pure_neg:
            superclasses = map_superclasses(scp_codes, superclass_lookup, threshold)
            info = {
                "superclasses": superclasses or ["OTHERS"],
                "primary": "OTHERS",
                "is_pure_norm": 0,
                "patient_id": Y.iloc[idx].get("patient_id", None),
            }
            indices.append((idx, Y.index[idx], info))
    return indices


def filter_others_code(
    positive_code: str,
    negative_code: str,
    threshold: int = TARGET_THRESHOLD,
) -> List[Tuple[int, str, dict]]:
    """Code-based OTHERS filter (legacy)."""
    return filter_others_mixed(
        positive_val=positive_code,
        positive_is_superclass=False,
        negative_val=negative_code,
        negative_is_superclass=False,
        threshold=threshold,
    )


def calculate_total_std_from_filtered_ecg(
    filtered_ecg: np.ndarray, r_peaks: np.ndarray, sampling_rate: int = SAMPLING_RATE
) -> float:
    """Used for selecting best R-peak set; unchanged."""
    total_std = 0.0
    for lead_idx in range(12):
        lead_signal = filtered_ecg[:, lead_idx]
        heartbeats = extract_heartbeats(lead_signal, r_peaks, sampling_rate)
        normalized_heartbeats, _, _, _ = split_and_resample_heartbeats(
            heartbeats, sampling_rate
        )
        if normalized_heartbeats:
            signals = np.array([beat["signal"] for beat in normalized_heartbeats])
            lead_std = np.mean(np.std(signals, axis=0))
            total_std += lead_std
    return total_std


def resample_segment(segment: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a 1D signal to a target length using linear interpolation."""
    if len(segment) == target_len:
        return segment
    x_old = np.linspace(0, 1, len(segment))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, segment)


def extract_dynamic_heartbeats(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    target_len: int = TARGET_HEARTBEAT_LEN,
) -> np.ndarray:
    """
    For each R peak, take a window of length equal to the current RR interval
    (i.e., from R to next R) and resample to target_len.
    The window is centered at the R peak: [R - RR/2, R + RR/2].
    Returns array of shape (n_beats, target_len).
    """
    n_peaks = len(r_peaks)
    if n_peaks == 0:
        return np.empty((0, target_len))
    heartbeats = []
    for i, r in enumerate(r_peaks):
        # Determine RR interval (distance to next R peak)
        if i < n_peaks - 1:
            rr = r_peaks[i + 1] - r
        else:
            # Last beat: use previous RR interval or fallback to 2 seconds
            rr = r - r_peaks[i - 1] if i > 0 else int(2 * SAMPLING_RATE)
        half_rr = rr // 2
        start = r - half_rr
        end = r + half_rr + 1  # inclusive of R peak
        # Extract segment with boundary padding (zero padding)
        if start < 0 or end > len(signal):
            segment = np.zeros(end - start)
            valid_start = max(start, 0)
            valid_end = min(end, len(signal))
            src_start = max(0, -start)
            src_end = src_start + (valid_end - valid_start)
            segment[src_start:src_end] = signal[valid_start:valid_end]
        else:
            segment = signal[start:end]
        # Resample to target length
        beat = resample_segment(segment, target_len)
        heartbeats.append(beat)
    return np.array(heartbeats)


def process_single_ecg(
    idx: int, output_dir: str, ecg_id_override: Optional[str] = None
) -> bool:
    ecg_id = ecg_id_override if ecg_id_override is not None else Y.index[idx]
    try:
        filtered_ecg, all_r_peaks = filter_ecg_and_detect_peaks(idx)
        best_rp = find_best_r_peaks(filtered_ecg, all_r_peaks)
        save_normalized_heartbeats(filtered_ecg, best_rp, output_dir, ecg_id)
        return True
    except Exception as e:
        print(f"Error processing {ecg_id} (index {idx}): {e}")
        return False


def filter_ecg_and_detect_peaks(idx: int) -> tuple[np.ndarray, dict]:
    """Load signal, filter and detect R peaks in each lead"""
    X = load_raw_data(Y, SAMPLING_RATE, PATH, idx)
    samples = X.shape[1]
    filtered_ecg = np.zeros((samples, 12))
    all_r_peaks = {}

    r_peak_func = (
        detect_r_peaks_envelope
        if idx in irregular
        else detect_r_peaks_neurokit_NeuroKit2
    )
    for lead in range(12):
        sig = X[0, :, lead]
        filt = remove_baseline_wander_hp_filter(sig, SAMPLING_RATE, cutoff=0.5)
        filtered_ecg[:, lead] = filt
        rp = _round_and_clip_indices(
            r_peak_func(filt, SAMPLING_RATE), len(filt), filt, "Envelope"
        )
        all_r_peaks[lead] = rp
    return filtered_ecg, all_r_peaks


def find_best_r_peaks(filtered_ecg: np.ndarray, all_r_peaks: dict) -> np.ndarray:
    """By calculating the alignment standard deviation, the optimal set of R peaks is selected."""
    unique_sets = []
    for lead, rp in all_r_peaks.items():
        t = tuple(rp)
        found = any(
            t == existing[0] and (existing[1].append(lead) or True)
            for existing in unique_sets
        )
        if not found:
            unique_sets.append([t, [lead]])

    best_rp, best_std = None, float("inf")
    for t, _ in unique_sets:
        rp_arr = np.array(t)
        std_val = calculate_total_std_from_filtered_ecg(filtered_ecg, rp_arr)
        if std_val < best_std:
            best_std = std_val
            best_rp = rp_arr
    return best_rp


def save_normalized_heartbeats(
    filtered_ecg: np.ndarray, best_rp: np.ndarray, output_dir: str, ecg_id: str
) -> None:
    """Extract dynamic heartbeats and save them to disk."""
    for lead in range(12):
        lead_signal = filtered_ecg[:, lead]
        heartbeats_arr = extract_dynamic_heartbeats(lead_signal, best_rp)
        if heartbeats_arr.size > 0:
            filename = os.path.join(
                output_dir, f"{ecg_id}_{LEAD_NAMES[lead]}_normalized_heartbeats.npy"
            )
            np.save(filename, heartbeats_arr)


def filter_by_scp_codes(
    scp_code_keys: List[str], min_values: List[float]
) -> List[Tuple[int, str]]:
    indices = []
    for idx in range(len(Y)):
        scp_codes = ensure_dict(Y.scp_codes.iloc[idx])
        if all(scp_codes.get(k, 0) >= v for k, v in zip(scp_code_keys, min_values)):
            indices.append((idx, Y.index[idx]))
    return indices


def filter_by_superclass(
    superclass_list: List[str],
    pure_norm_only: bool = False,
    single_superclass_only: bool = False,
    threshold: int = TARGET_THRESHOLD,
) -> List[Tuple[int, str, dict]]:
    scp_df = load_scp_statements()
    superclass_lookup = build_superclass_lookup(scp_df)

    indices = []
    for idx in range(len(Y)):
        scp_codes = ensure_dict(Y.scp_codes.iloc[idx])
        if pure_norm_only and not is_pure_norm(scp_codes, threshold=threshold):
            continue
        superclasses = map_superclasses(
            scp_codes, superclass_lookup, threshold=threshold
        )
        if single_superclass_only and len(superclasses) != 1:
            continue
        if superclass_list and all(sc not in superclasses for sc in superclass_list):
            continue
        info = {
            "superclasses": superclasses,
            "primary": superclasses[0] if len(superclasses) == 1 else "MIXED",
            "is_pure_norm": is_pure_norm(scp_codes, threshold=threshold),
            "patient_id": Y.iloc[idx].get("patient_id", None),
        }
        indices.append((idx, Y.index[idx], info))
    return indices


def process_batch(
    records: List[Tuple[int, str]],
    output_dir: str,
    max_workers: int = MAX_WORKERS,
    desc: str = "Processing signals",
) -> Tuple[int, int]:
    processed = 0
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_ecg, idx, output_dir, ecg_id): idx
            for idx, ecg_id in records
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=desc,
        ):
            if future.result():
                processed += 1
            else:
                errors += 1
    return processed, errors


def save_metadata(records_info: List[Tuple[int, str, dict]], output_dir: str):
    rows = []
    rows.extend(
        {
            "ecg_id": ecg_id,
            "patient_id": info["patient_id"],
            "superclasses": ";".join(info["superclasses"]),
            "primary_superclass": info["primary"],
            "is_pure_norm": int(info["is_pure_norm"]),
        }
        for idx, ecg_id, info in records_info
    )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Extract normalized heartbeats from PTB-XL with flexible filtering."
    )
    parser.add_argument(
        "--scp_codes",
        nargs="+",
        default=None,
        help="SCP code keys (e.g., NORM STTC) - original AND mode",
    )
    parser.add_argument(
        "--min_value",
        nargs="+",
        type=float,
        default=None,
        help="Minimum values for each SCP code",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root output directory (subfolder will be auto-created).",
    )
    parser.add_argument(
        "--superclass",
        nargs="+",
        help="Superclass names (e.g., NORM MI) - OR mode, any match",
    )
    parser.add_argument(
        "--pure",
        action="store_true",
        help="(Only with --superclass NORM) Keep only pure NORM records.",
    )
    parser.add_argument(
        "--single-superclass-only",
        action="store_true",
        help="Keep only records that belong to exactly one superclass.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=TARGET_THRESHOLD,
        help="Confidence threshold for SCP codes (default 100).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=MAX_WORKERS,
        help="Number of parallel threads.",
    )
    # OTHERS mode arguments - mixed code/superclass
    parser.add_argument(
        "--others",
        action="store_true",
        help="Generate OTHERS dataset (exclude positive and pure negative records).",
    )
    # Positive exclusion (code or superclass)
    parser.add_argument(
        "--positive-code",
        type=str,
        default="LVH",
        help="Positive SCP code to exclude (default LVH). Used if --positive-is-superclass is False.",
    )
    parser.add_argument(
        "--positive-superclass",
        type=str,
        default=None,
        help="Positive superclass name to exclude (e.g., HYP). Must be provided if --positive-is-superclass is set.",
    )
    parser.add_argument(
        "--positive-is-superclass",
        action="store_true",
        help="If set, use --positive-superclass instead of --positive-code for positive exclusion.",
    )
    # Negative exclusion (code or superclass)
    parser.add_argument(
        "--negative-code",
        type=str,
        default="NORM",
        help="Negative SCP code for pure negative exclusion (default NORM). Used if --negative-is-superclass is False.",
    )
    parser.add_argument(
        "--negative-superclass",
        type=str,
        default=None,
        help="Negative superclass name for pure negative exclusion (e.g., NORM). Must be provided if --negative-is-superclass is set.",
    )
    parser.add_argument(
        "--negative-is-superclass",
        action="store_true",
        help="If set, use --negative-superclass instead of --negative-code for pure negative exclusion.",
    )

    args = parser.parse_args()

    # OTHERS mode
    if args.others:
        # Validate arguments
        if args.positive_is_superclass and args.positive_superclass is None:
            parser.error("--positive-is-superclass requires --positive-superclass")
        if args.negative_is_superclass and args.negative_superclass is None:
            parser.error("--negative-is-superclass requires --negative-superclass")

        # Determine positive value and type
        pos_val = (
            args.positive_superclass
            if args.positive_is_superclass
            else args.positive_code
        )
        pos_type = "superclass" if args.positive_is_superclass else "code"
        # Determine negative value and type
        neg_val = (
            args.negative_superclass
            if args.negative_is_superclass
            else args.negative_code
        )
        neg_type = "superclass" if args.negative_is_superclass else "code"

        output_root = args.output_dir or RESULTS_PATH
        os.makedirs(output_root, exist_ok=True)

        records_info = filter_others_mixed(
            positive_val=pos_val,
            positive_is_superclass=args.positive_is_superclass,
            negative_val=neg_val,
            negative_is_superclass=args.negative_is_superclass,
            threshold=args.threshold,
        )
        subfolder = f"OTHERS_exclude_{pos_val}_{pos_type}_pure_{neg_val}_{neg_type}"
        print(
            f"Using mixed filter: exclude positive ({pos_type}={pos_val}), exclude pure negative ({neg_type}={neg_val})"
        )

        if not records_info:
            print("No records match the OTHERS criteria.")
            return

        output_dir = os.path.join(output_root, subfolder)
        os.makedirs(output_dir, exist_ok=True)

        records = [(idx, ecg_id) for idx, ecg_id, _ in records_info]
        print("=" * 80)
        print(f"Processing {len(records)} records for OTHERS")
        print(f"Output directory: {output_dir}")
        print("=" * 80)

        processed, errors = process_batch(
            records, output_dir, max_workers=args.max_workers
        )
        print(f"Processed: {processed}, Errors: {errors}")

        save_metadata(records_info, output_dir)
        print(f"Metadata saved to {output_dir}/metadata.csv")
        return

    # Original non-OTHERS logic (unchanged)
    if (
        args.scp_codes is None
        and args.superclass is None
        and not args.single_superclass_only
    ):
        args.scp_codes = ["LVH"]
        args.min_value = [100]
        print("No filter specified. Defaulting to --scp_codes LVH --min_value 100")

    output_root = args.output_dir or RESULTS_PATH
    os.makedirs(output_root, exist_ok=True)

    records = []
    records_info = []

    if args.scp_codes is not None:
        if args.min_value is None:
            args.min_value = [TARGET_THRESHOLD] * len(args.scp_codes)
        if len(args.scp_codes) != len(args.min_value):
            raise ValueError("Number of scp_codes must match number of min_values")
        raw_records = filter_by_scp_codes(args.scp_codes, args.min_value)
        records = list(raw_records)
        scp_df = load_scp_statements()
        superclass_lookup = build_superclass_lookup(scp_df)
        for idx, ecg_id in raw_records:
            scp_codes = ensure_dict(Y.scp_codes.iloc[idx])
            superclasses = map_superclasses(
                scp_codes, superclass_lookup, threshold=args.threshold
            )
            info = {
                "superclasses": superclasses,
                "primary": superclasses[0] if len(superclasses) == 1 else "MIXED",
                "is_pure_norm": is_pure_norm(scp_codes, threshold=args.threshold),
                "patient_id": Y.iloc[idx].get("patient_id", None),
            }
            records_info.append((idx, ecg_id, info))
        subfolder = "_".join(
            [f"{k}_{v}" for k, v in zip(args.scp_codes, args.min_value)]
        )

    else:
        superclass_list = args.superclass or []
        pure = args.pure and ("NORM" in superclass_list)
        raw_records = filter_by_superclass(
            superclass_list=superclass_list,
            pure_norm_only=pure,
            single_superclass_only=args.single_superclass_only,
            threshold=args.threshold,
        )
        records = [(idx, ecg_id) for idx, ecg_id, _ in raw_records]
        records_info = raw_records
        if args.single_superclass_only and args.superclass is None:
            subfolder = "single_superclass"
        else:
            subfolder = "_".join(superclass_list) + ("_pure" if pure else "")

    if not records:
        print("No records match the criteria.")
        return

    output_dir = os.path.join(output_root, subfolder)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(f"Processing {len(records)} records")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    processed, errors = process_batch(records, output_dir, max_workers=args.max_workers)
    print(f"Processed: {processed}, Errors: {errors}")

    save_metadata(records_info, output_dir)
    print(f"Metadata saved to {output_dir}/metadata.csv")


if __name__ == "__main__":
    main()
