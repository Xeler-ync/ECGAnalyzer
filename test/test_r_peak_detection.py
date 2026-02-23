import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import wfdb
from wfdb import processing as wfdb_processing
import neurokit2 as nk
import matplotlib.pyplot as plt
import random
from scipy import ndimage
from scipy import signal as signal
from scipy.signal import find_peaks

from utils._baseline import (
    remove_baseline_wander_hp_filter,
)
from utils._r_peaks import (
    detect_r_peaks_basic,
    detect_r_peaks_adaptive,
    detect_r_peaks_derivative,
    detect_r_peaks_hamilton_ECG_Detectors,
    detect_r_peaks_christov_ECG_Detectors,
    detect_r_peaks_engelese_kulp_ECG_Detectors,
    detect_r_peaks_pan_tompkins_ECG_Detectors,
    detect_r_peaks_swt_ECG_Detectors,
    detect_r_peaks_matched_filter_ECG_Detectors,
    detect_r_peaks_wqrs_ECG_Detectors,
    detect_r_peaks_two_moving_average_ECG_Detectors,
    detect_r_peaks_pantompkins1985_NeuroKit2,
    detect_r_peaks_hamilton2002_NeuroKit2,
    detect_r_peaks_christov2004_NeuroKit2,
    detect_r_peaks_engzeemod2012_NeuroKit2,
    detect_r_peaks_elgendi2010_NeuroKit2,
    detect_r_peaks_zong2003_NeuroKit2,
    detect_r_peaks_martinez2004_NeuroKit2,
    detect_r_peaks_kalidas2017_NeuroKit2,
    detect_r_peaks_khamis2016_NeuroKit2,
    detect_r_peaks_manikandan2012_NeuroKit2,
    detect_r_peaks_nabian2018_NeuroKit2,
    detect_r_peaks_rodrigues2020_NeuroKit2,
    detect_r_peaks_emrich2023_NeuroKit2,
    detect_r_peaks_neurokit_NeuroKit2,
    detect_r_peaks_gamboa2008_NeuroKit2,
    detect_r_peaks_promac_NeuroKit2,
    detect_r_peaks_asi_NeuroKit2,
    evaluate_r_peak_detection,
)
from utils._config import PATH, SAMPLING_RATE, TGT_SAMPLING_RATE
from utils._data import Y
from utils._helpers import _safe_scale_and_clip


def _run_all_detectors_on_signal(sig, sr):
    """Run all detectors defined in notebook on single-channel sig (use functions already in file)."""
    # resample for ecgdetectors if needed
    sig_res, _ = wfdb_processing.resample_sig(sig, sr, TGT_SAMPLING_RATE) if sr != TGT_SAMPLING_RATE else (sig, sr)

    methods = {}
    # Local/simple
    methods["Basic"] = detect_r_peaks_basic(sig, sr)
    methods["Adaptive"] = detect_r_peaks_adaptive(sig, sr)
    methods["Derivative"] = detect_r_peaks_derivative(sig, sr)

    # ecgdetectors (run on resampled then scale back)
    def _scale(pks, from_sr, to_sr, L):
        if pks is None or len(pks) == 0:
            return np.array([], int)
        s = np.round(np.asarray(pks) * (to_sr / from_sr)).astype(int)
        return np.unique(np.clip(s, 0, L - 1))

    edet_fns = {
        "Hamilton (ecgdetectors)": detect_r_peaks_hamilton_ECG_Detectors,
        "Christov (ecgdetectors)": detect_r_peaks_christov_ECG_Detectors,
        "Engzee (ecgdetectors)": detect_r_peaks_engelese_kulp_ECG_Detectors,
        "PanTompkins (ecgdetectors)": detect_r_peaks_pan_tompkins_ECG_Detectors,
        "SWT (ecgdetectors)": detect_r_peaks_swt_ECG_Detectors,
        "MatchedFilter (ecgdetectors)": detect_r_peaks_matched_filter_ECG_Detectors,
        "WQRS (ecgdetectors)": detect_r_peaks_wqrs_ECG_Detectors,
        "TwoMA (ecgdetectors)": detect_r_peaks_two_moving_average_ECG_Detectors,
    }
    for name, fn in edet_fns.items():
        try:
            p = fn(sig_res, TGT_SAMPLING_RATE)
            methods[name] = _scale(p, TGT_SAMPLING_RATE, sr, len(sig))
        except Exception:
            methods[name] = np.array([], int)

    # NeuroKit2 variants (use the per-method wrapper functions in file if available)
    nk_names = [
        ("PanTompkins (NeuroKit2)", detect_r_peaks_pantompkins1985_NeuroKit2),
        ("Hamilton (NeuroKit2)", detect_r_peaks_hamilton2002_NeuroKit2),
        ("Christov (NeuroKit2)", detect_r_peaks_christov2004_NeuroKit2),
        ("Engzee (NeuroKit2)", detect_r_peaks_engzeemod2012_NeuroKit2),
        ("Elgendi (NeuroKit2)", detect_r_peaks_elgendi2010_NeuroKit2),
        ("Zong (NeuroKit2)", detect_r_peaks_zong2003_NeuroKit2),
        ("Martinez (NeuroKit2)", detect_r_peaks_martinez2004_NeuroKit2),
        ("Kalidas (NeuroKit2)", detect_r_peaks_kalidas2017_NeuroKit2),
        ("Khamis (NeuroKit2)", detect_r_peaks_khamis2016_NeuroKit2),
        ("Manikandan (NeuroKit2)", detect_r_peaks_manikandan2012_NeuroKit2),
        ("Nabian (NeuroKit2)", detect_r_peaks_nabian2018_NeuroKit2),
        ("Rodrigues (NeuroKit2)", detect_r_peaks_rodrigues2020_NeuroKit2),
        ("Emrich (NeuroKit2)", detect_r_peaks_emrich2023_NeuroKit2),
        ("NeuroKitDefault", detect_r_peaks_neurokit_NeuroKit2),
        ("Gamboa (NeuroKit2)", detect_r_peaks_gamboa2008_NeuroKit2),
        ("Promac (NeuroKit2)", detect_r_peaks_promac_NeuroKit2),
        ("ASI (NeuroKit2)", detect_r_peaks_asi_NeuroKit2),
    ]
    for name, fn in nk_names:
        try:
            p = fn(sig, sr)
            methods[name] = _safe_scale_and_clip(p, from_sr=sr, to_sr=sr, L=len(sig))
        except Exception:
            # fallback to nk.ecg_peaks generic if wrapper absent
            try:
                out = nk.ecg_peaks(
                    sig, sampling_rate=sr, method=name.split()[0].lower()
                )
                peaks_dict = out[0] if isinstance(out, tuple) else out
                if "ECG_R_Peaks" in peaks_dict:
                    methods[name] = np.where(
                        np.asarray(peaks_dict["ECG_R_Peaks"]) == 1
                    )[0]
                else:
                    methods[name] = np.array([], int)
            except Exception:
                methods[name] = np.array([], int)

    # ensure int arrays and in-bounds
    for k in list(methods.keys()):
        a = np.asarray(methods[k], int) if methods[k] is not None else np.array([], int)
        a = np.unique(a[(a >= 0) & (a < len(sig))])
        methods[k] = a
    return methods


def _consensus_from_methods(methods_dict, sig_len, sr, tol_ms=50, vote_frac=0.5):
    """Build pseudo-ground-truth by majority vote within tol_ms window.
    For each peak from each method add +1 to votes over [p-tol:p+tol], then peaks where votes>=threshold
    """
    n_methods = len(methods_dict)
    tol = max(1, int(sr * tol_ms / 1000.0))
    votes = np.zeros(sig_len, int)
    for pks in methods_dict.values():
        for p in pks:
            lo = max(0, p - tol)
            hi = min(sig_len, p + tol + 1)
            votes[lo:hi] += 1
    threshold = max(1, int(np.ceil(n_methods * vote_frac)))
    # find local maxima above threshold as consensus peaks
    smoothed = ndimage.maximum_filter1d(votes, size=tol * 2 + 1)
    peaks, props = find_peaks(smoothed, height=threshold)
    return peaks


def _match_peaks_to_reference(pred, ref, tol_ms, sr):
    """Return TP, FP, FN counts for pred vs ref using tol_ms tolerance."""
    tol = max(1, int(sr * tol_ms / 1000.0))
    matched_ref = np.zeros(len(ref), bool)
    tp = 0
    for p in pred:
        # find any ref within tol
        idx = np.where(np.abs(ref - p) <= tol)[0]
        if idx.size > 0:
            tp += 1
            matched_ref[idx[0]] = True
    fp = len(pred) - tp
    fn = np.count_nonzero(~matched_ref)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_methods_on_random_ecgs(
    n=10, lead=1, tol_ms=50, vote_frac=0.5, random_seed=42, target_nk_sr=250
):
    random.seed(random_seed)
    indices = list(Y.index)
    if len(indices) == 0:
        print("No records in Y")
        return {}, {}

    sampled = random.sample(indices, min(n, len(indices)))
    summary_metrics = {}

    for rec_id in sampled:
        row = Y.loc[rec_id]
        fname = row.filename_lr if "filename_lr" in row.index else row.filename
        record_path = PATH + fname
        try:
            sig, meta = wfdb.rdsamp(record_path)
            sig = np.asarray(sig)
        except Exception as e:
            print(f"Failed to load {record_path}: {e}")
            continue
        if lead >= sig.shape[1]:
            continue

        channel = sig[:, lead]
        # baseline remove in original rate
        sig_f = remove_baseline_wander_hp_filter(channel, SAMPLING_RATE, cutoff=0.5)

        # choose resample rate for NeuroKit/ecgdetectors (>= target_nk_sr)
        # NeuroKit sample rate must >= 250
        sr_use = max(SAMPLING_RATE, target_nk_sr)
        if sr_use != SAMPLING_RATE:
            sig_resampled, _ = wfdb_processing.resample_sig(
                sig_f, SAMPLING_RATE, sr_use
            )
        else:
            sig_resampled = sig_f

        # run detectors on resampled signal (functions handle internal resampling for ecgdetectors)
        methods = _run_all_detectors_on_signal(sig_resampled, sr_use)

        # --- SANITIZE detector outputs to avoid out-of-bounds indices (fixes manikandan error) ---
        def _sanitize_peaks(peaks, sig_len):
            if peaks is None:
                return np.array([], dtype=int)
            a = np.asarray(peaks)
            # if detector returned a binary vector (0/1) of length == signal, convert to indices
            if a.ndim == 1 and a.size == sig_len and set(np.unique(a)).issubset({0, 1}):
                idx = np.where(a == 1)[0]
            else:
                # round/floor floats and clip to valid range
                idx = np.round(a).astype(int)
            idx = np.unique(idx[(idx >= 0) & (idx < sig_len)])
            return idx

        for k in list(methods.keys()):
            methods[k] = _sanitize_peaks(methods[k], len(sig_resampled))
        # --- end sanitize ---

        # build consensus (pseudo-GT) in resampled domain
        consensus = _consensus_from_methods(
            methods, len(sig_resampled), sr_use, tol_ms=tol_ms, vote_frac=vote_frac
        )

        # evaluate each method vs consensus, reuse evaluate_r_peak_detection with sr_use
        for mname, pks in methods.items():
            acc = _match_peaks_to_reference(pks, consensus, tol_ms, sr_use)
            intrinsic = evaluate_r_peak_detection(pks, sig_resampled, sr_use)
            combined = {
                "precision": acc["precision"],
                "recall": acc["recall"],
                "f1": acc["f1"],
                "peak_count": intrinsic.get("peak_count", 0),
                "avg_peak_distance_sec": intrinsic.get("avg_peak_distance_sec", 0),
                "heart_rate_bpm": intrinsic.get("heart_rate_bpm", 0),
            }
            summary_metrics.setdefault(mname, []).append(combined)

    # aggregate (mean) across records
    methods_list = sorted(summary_metrics.keys())
    agg = {}
    for m in methods_list:
        arr = summary_metrics[m]
        agg[m] = {
            "precision_mean": np.mean([x["precision"] for x in arr]) if arr else 0,
            "recall_mean": np.mean([x["recall"] for x in arr]) if arr else 0,
            "f1_mean": np.mean([x["f1"] for x in arr]) if arr else 0,
            "mean_peak_count": np.mean([x["peak_count"] for x in arr]) if arr else 0,
            "mean_hr": np.mean([x["heart_rate_bpm"] for x in arr]) if arr else 0,
        }

    # plot summary
    if methods_list:
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        names = methods_list
        precision_vals = [agg[n]["precision_mean"] for n in names]
        recall_vals = [agg[n]["recall_mean"] for n in names]
        f1_vals = [agg[n]["f1_mean"] for n in names]

        ax[0].barh(names, precision_vals)
        ax[0].set_title("Precision (mean)")
        ax[0].set_xlim(0, 1)
        ax[1].barh(names, recall_vals, color="orange")
        ax[1].set_title("Recall (mean)")
        ax[1].set_xlim(0, 1)
        ax[2].barh(names, f1_vals, color="green")
        ax[2].set_title("F1 (mean)")
        ax[2].set_xlim(0, 1)
        plt.tight_layout()
        plt.show()

    return agg, summary_metrics


agg_metrics, per_record = evaluate_methods_on_random_ecgs(
    n=78, lead=1, tol_ms=50, vote_frac=0.5, random_seed=114514
)
print("Evaluation complete. Methods evaluated:", len(agg_metrics))
