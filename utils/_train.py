import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import label_binarize
import joblib

from utils._config import LEAD_NAMES, RandomForestParams, RANDOM_FOREST_PARAMS

DEFAULT_TEST_SIZE = 0.20
DEFAULT_VAL_SIZE = 0.20
DEFAULT_THRESHOLD_GRID = np.round(np.arange(0.10, 0.91, 0.02), 2)


def load_data(pos_dir: Path, neg_dir: Path, pure_neg: bool = False):
    """
    Load data from two directories (positive and negative classes).
    Each directory must contain metadata.csv and normalized heartbeat .npy files.

    Returns:
        X_3d: (n_samples, beat_length, 12)
        X_flat: (n_samples, beat_length * 12)
        y: labels (1 for positive, 0 for negative)
        groups: patient_id for each sample
    """

    def load_one_dir(data_dir: Path, label: int, require_pure: bool = False):
        meta_path = data_dir / "metadata.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.csv not found in {data_dir}")
        metadata = pd.read_csv(meta_path)

        # For negative class (label=0) and require_pure=True, keep only pure normals
        if label == 0 and require_pure:
            metadata = metadata[metadata["is_pure_norm"] == 1]

        beat_length = None
        X_list = []
        groups_list = []
        for _, row in metadata.iterrows():
            ecg_id = row["ecg_id"]
            patient_id = row["patient_id"]
            lead_data = []
            for lead in LEAD_NAMES:
                fpath = data_dir / f"{ecg_id}_{lead}_normalized_heartbeats.npy"
                if not fpath.exists():
                    raise FileNotFoundError(f"Missing file: {fpath}")
                heartbeats = np.load(fpath)
                if beat_length is None:
                    beat_length = heartbeats.shape[1]
                else:
                    if heartbeats.shape[1] != beat_length:
                        raise ValueError(f"Inconsistent beat length in {fpath}")
                mean_beat = np.mean(heartbeats, axis=0)
                lead_data.append(mean_beat)
            X_sample = np.stack(lead_data, axis=1)  # (beat_length, 12)
            X_list.append(X_sample)
            groups_list.append(patient_id)

        X_3d = np.array(X_list)
        X_flat = X_3d.reshape(len(X_3d), -1)
        y = np.full(len(X_3d), label)
        groups = np.array(groups_list)
        return X_flat, y, groups, beat_length

    X_pos, y_pos, groups_pos, _ = load_one_dir(pos_dir, label=1, require_pure=False)
    X_neg, y_neg, groups_neg, beat_len = load_one_dir(
        neg_dir, label=0, require_pure=pure_neg
    )

    X_flat = np.vstack([X_pos, X_neg])
    y = np.hstack([y_pos, y_neg])
    groups = np.hstack([groups_pos, groups_neg])

    # Reconstruct X_3d
    X_3d = X_flat.reshape(-1, beat_len, 12)
    return X_3d, X_flat, y, groups


def grouped_split_indices(
    y, groups, test_size=0.2, val_size=0.2, random_state=42, max_tries=30
):
    """Split indices into train/val/test while preserving patient groups."""
    X_dummy = np.zeros(len(y))
    for offset in range(max_tries):
        rs = random_state + offset
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        train_val_idx, test_idx = next(gss_test.split(X_dummy, y, groups=groups))
        train_val_groups = groups[train_val_idx]
        train_val_y = y[train_val_idx]
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=val_size, random_state=rs + 1000
        )
        rel_train_idx, rel_val_idx = next(
            gss_val.split(X_dummy[train_val_idx], train_val_y, groups=train_val_groups)
        )
        train_idx = train_val_idx[rel_train_idx]
        val_idx = train_val_idx[rel_val_idx]
        if (
            len(np.unique(y[train_idx])) == 2
            and len(np.unique(y[val_idx])) == 2
            and len(np.unique(y[test_idx])) == 2
        ):
            return train_idx, val_idx, test_idx
    raise RuntimeError("Failed to create splits with both classes in each set")


def find_best_threshold(y_true, y_prob, threshold_grid=DEFAULT_THRESHOLD_GRID):
    best_threshold = 0.5
    best_f1 = -1.0
    rows = []
    for thresh in threshold_grid:
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        rows.append({"threshold": thresh, "f1": f1})
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    return best_threshold, pd.DataFrame(rows)


def compute_binary_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_true, y_prob))
            if len(np.unique(y_true)) > 1
            else float("nan")
        ),
    }


def run_training(
    pos_dir,
    neg_dir,
    out_dir,
    pure_neg=True,
    test_size=DEFAULT_TEST_SIZE,
    val_size=DEFAULT_VAL_SIZE,
    rf_params: RandomForestParams = RANDOM_FOREST_PARAMS,
):
    """
    Full training pipeline:
    - Load data
    - Split with patient groups
    - Train Random Forest
    - Tune threshold on validation set
    - Evaluate on test set
    - Save model, predictions, metrics
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pos_dir = Path(pos_dir)
    neg_dir = Path(neg_dir)

    print(f"Loading positive class from {pos_dir} ...")
    print(f"Loading negative class from {neg_dir} ...")
    X_3d, X_flat, y, groups = load_data(pos_dir, neg_dir, pure_neg=pure_neg)

    print(f"Loaded {len(y)} samples (positive: {sum(y==1)}, negative: {sum(y==0)})")

    train_idx, val_idx, test_idx = grouped_split_indices(
        y,
        groups,
        test_size=test_size,
        val_size=val_size,
        random_state=rf_params.random_state,
    )

    X_train, X_val, X_test = X_flat[train_idx], X_flat[val_idx], X_flat[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    clf = RandomForestClassifier(
        n_estimators=rf_params.n_estimators,
        bootstrap=rf_params.bootstrap,
        max_samples=rf_params.max_samples,
        oob_score=rf_params.oob_score,
        max_depth=rf_params.max_depth,
        min_samples_split=rf_params.min_samples_split,
        min_samples_leaf=rf_params.min_samples_leaf,
        max_features=rf_params.max_features,
        class_weight=rf_params.class_weight,
        random_state=rf_params.random_state,
        n_jobs=rf_params.n_jobs,
        criterion=rf_params.criterion,
        warm_start=rf_params.warm_start,
        verbose=rf_params.verbose,
    )
    clf.fit(X_train, y_train)

    val_prob = clf.predict_proba(X_val)[:, 1]
    best_thresh, thresh_df = find_best_threshold(y_val, val_prob)

    # Save artifacts
    joblib.dump(clf, out_dir / "model.joblib")
    np.save(out_dir / "y_test.npy", y_test)
    np.save(out_dir / "X_test_flat.npy", X_test)
    np.save(out_dir / "val_prob.npy", val_prob)
    test_prob = clf.predict_proba(X_test)[:, 1]
    np.save(out_dir / "test_prob.npy", test_prob)
    thresh_df.to_csv(out_dir / "threshold_tuning.csv", index=False)

    split_info = {
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "positive_train": int(y_train.sum()),
        "positive_val": int(y_val.sum()),
        "positive_test": int(y_test.sum()),
        "negative_train": int((y_train == 0).sum()),
        "negative_val": int((y_val == 0).sum()),
        "negative_test": int((y_test == 0).sum()),
        "oob_score": float(clf.oob_score_) if rf_params.oob_score else None,
        "best_threshold": float(best_thresh),
    }
    pd.DataFrame([split_info]).to_csv(out_dir / "split_info.csv", index=False)

    test_pred = (test_prob >= best_thresh).astype(int)
    test_metrics = compute_binary_metrics(y_test, test_pred, test_prob)

    print("\nTest set metrics (using tuned threshold):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nBest validation threshold: {best_thresh:.3f}")
    print(f"Model saved to {out_dir / 'model.joblib'}")

    pd.DataFrame([test_metrics]).to_csv(out_dir / "test_metrics.csv", index=False)

    return clf, test_metrics


def load_data_multiclass(
    pos_dir: Path,
    neg_dir: Path,
    others_dir: Path,
    pure_norm: bool = True,
    pure_others: bool = False,
) -> tuple:
    """
    Load data from three directories (positive and negative and other classes).
    Each directory must contain metadata.csv and normalized heartbeat .npy files.

    Returns:
        X_3d: (n_samples, beat_length, 12)
        X_flat: (n_samples, beat_length * 12)
        y: labels (0 for positive, 1 for negative, 2 for other)
        groups: patient_id for each sampled
    """

    def load_one_dir(data_dir: Path, label: int, require_pure: bool = False):
        meta_path = data_dir / "metadata.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.csv not found in {data_dir}")
        metadata = pd.read_csv(meta_path)

        # If purity filtering is needed (only for NORM)
        if label == 0 and require_pure:
            if "is_pure_norm" in metadata.columns:
                metadata = metadata[metadata["is_pure_norm"] == 1]
            else:
                print(
                    f"Warning: {data_dir} does not have 'is_pure_norm' column, skipping purity filter"
                )

        beat_length = None
        X_list = []
        groups_list = []
        for _, row in metadata.iterrows():
            ecg_id = row["ecg_id"]
            patient_id = row["patient_id"]
            lead_data = []
            for lead in LEAD_NAMES:
                fpath = data_dir / f"{ecg_id}_{lead}_normalized_heartbeats.npy"
                if not fpath.exists():
                    raise FileNotFoundError(f"Missing file: {fpath}")
                heartbeats = np.load(fpath)  # shape (n_beats, beat_len)
                if beat_length is None:
                    beat_length = heartbeats.shape[1]
                else:
                    if heartbeats.shape[1] != beat_length:
                        raise ValueError(f"Inconsistent beat length in {fpath}")
                mean_beat = np.mean(heartbeats, axis=0)  # Use mean heartbeat as feature
                lead_data.append(mean_beat)
            X_sample = np.stack(lead_data, axis=1)  # (beat_length, 12)
            X_list.append(X_sample)
            groups_list.append(patient_id)

        X_3d = np.array(X_list)  # (n_samples, beat_len, 12)
        X_flat = X_3d.reshape(len(X_3d), -1)
        y = np.full(len(X_3d), label)
        groups = np.array(groups_list)
        return X_flat, y, groups, beat_length

    # Load the three categories separately
    X_norm, y_norm, groups_norm, beat_len = load_one_dir(
        pos_dir, label=0, require_pure=pure_norm
    )
    X_pos, y_lpos, groups_pos, _ = load_one_dir(neg_dir, label=1, require_pure=False)
    X_others, y_others, groups_others, _ = load_one_dir(
        others_dir, label=2, require_pure=pure_others
    )

    X_flat = np.vstack([X_norm, X_pos, X_others])
    y = np.hstack([y_norm, y_lpos, y_others])
    groups = np.hstack([groups_norm, groups_pos, groups_others])

    # Reconstruct X_3d
    X_3d = X_flat.reshape(-1, beat_len, 12)
    return X_3d, X_flat, y, groups


def grouped_split_indices_multiclass(
    y, groups, test_size=0.2, val_size=0.2, random_state=42, max_tries=50
):
    """Split indices ensuring that train/val/test sets all contain three classes (grouped stratified)"""
    X_dummy = np.zeros(len(y))
    n_classes = len(np.unique(y))
    for offset in range(max_tries):
        rs = random_state + offset
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        train_val_idx, test_idx = next(gss_test.split(X_dummy, y, groups=groups))

        train_val_groups = groups[train_val_idx]
        train_val_y = y[train_val_idx]
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=val_size, random_state=rs + 1000
        )
        rel_train_idx, rel_val_idx = next(
            gss_val.split(X_dummy[train_val_idx], train_val_y, groups=train_val_groups)
        )

        train_idx = train_val_idx[rel_train_idx]
        val_idx = train_val_idx[rel_val_idx]

        # Check that each set contains all classes
        if (
            len(np.unique(y[train_idx])) == n_classes
            and len(np.unique(y[val_idx])) == n_classes
            and len(np.unique(y[test_idx])) == n_classes
        ):
            return train_idx, val_idx, test_idx
    raise RuntimeError("Failed to create splits with all three classes in each set")


def compute_multiclass_metrics(y_true, y_pred, y_prob):
    """Compute overall multiclass metrics (accuracy, macro/weighted f1, roc_auc, etc.)"""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }
    try:
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
        )
        metrics["roc_auc_ovr_weighted"] = float(
            roc_auc_score(y_true_bin, y_prob, average="weighted", multi_class="ovr")
        )
    except Exception:
        metrics["roc_auc_ovr_macro"] = float("nan")
        metrics["roc_auc_ovr_weighted"] = float("nan")
    return metrics


def compute_per_class_metrics_multiclass(y_true, y_pred, y_prob, class_names):
    """Compute detailed metrics per class (accuracy, precision, recall, F1, AUC, etc.)"""
    rows = []
    for i, name in enumerate(class_names):
        y_true_bin = (y_true == i).astype(int)
        y_pred_bin = (y_pred == i).astype(int)

        tp = ((y_true_bin == 1) & (y_pred_bin == 1)).sum()
        tn = ((y_true_bin == 0) & (y_pred_bin == 0)).sum()
        fp = ((y_true_bin == 0) & (y_pred_bin == 1)).sum()
        fn = ((y_true_bin == 1) & (y_pred_bin == 0)).sum()

        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        support = (y_true == i).sum()

        try:
            auc = roc_auc_score(y_true_bin, y_prob[:, i])
        except Exception:
            auc = float("nan")

        rows.append(
            {
                "class": name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": auc,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "support": support,
            }
        )
    return pd.DataFrame(rows)


def run_training_multiclass(
    pos_dir,
    neg_dir,
    others_dir,
    out_dir,
    pos_name,
    neg_name,
    others_name,
    pure_norm=True,
    test_size=DEFAULT_TEST_SIZE,
    val_size=DEFAULT_VAL_SIZE,
    rf_params: RandomForestParams = RANDOM_FOREST_PARAMS,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pos_dir = Path(pos_dir)
    neg_dir = Path(neg_dir)
    others_dir = Path(others_dir)

    print(
        f"Loading data from:\n  {pos_name}  : {pos_dir}\n  {neg_name}   : {neg_dir}\n  {others_name}: {others_dir}"
    )
    X_3d, X_flat, y, groups = load_data_multiclass(
        pos_dir, neg_dir, others_dir, pure_norm=pure_norm
    )

    class_names = [pos_name, neg_name, others_name]
    print(
        f"Loaded {len(y)} samples: {pos_name}={(y==0).sum()}, {neg_name}={(y==1).sum()}, {others_name}={(y==2).sum()}"
    )

    # Split dataset
    train_idx, val_idx, test_idx = grouped_split_indices_multiclass(
        y,
        groups,
        test_size=test_size,
        val_size=val_size,
        random_state=rf_params.random_state,
    )
    X_train, X_val, X_test = X_flat[train_idx], X_flat[val_idx], X_flat[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    clf = RandomForestClassifier(
        n_estimators=rf_params.n_estimators,
        bootstrap=rf_params.bootstrap,
        max_samples=rf_params.max_samples,
        oob_score=rf_params.oob_score,
        max_depth=rf_params.max_depth,
        min_samples_split=rf_params.min_samples_split,
        min_samples_leaf=rf_params.min_samples_leaf,
        max_features=rf_params.max_features,
        class_weight=rf_params.class_weight,
        random_state=rf_params.random_state,
        n_jobs=rf_params.n_jobs,
        criterion=rf_params.criterion,
        warm_start=rf_params.warm_start,
        verbose=rf_params.verbose,
    )
    clf.fit(X_train, y_train)

    val_prob = clf.predict_proba(X_val)
    test_prob = clf.predict_proba(X_test)
    test_pred = clf.predict(X_test)

    overall_metrics = compute_multiclass_metrics(y_test, test_pred, test_prob)
    per_class_df = compute_per_class_metrics_multiclass(
        y_test, test_pred, test_prob, class_names
    )

    split_info = {
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        f"n_train_{pos_name}": int((y_train == 0).sum()),
        f"n_train_{neg_name}": int((y_train == 1).sum()),
        f"n_train_{others_name}": int((y_train == 2).sum()),
        f"n_val_{pos_name}": int((y_val == 0).sum()),
        f"n_val_{neg_name}": int((y_val == 1).sum()),
        f"n_val_{others_name}": int((y_val == 2).sum()),
        f"n_test_{pos_name}": int((y_test == 0).sum()),
        f"n_test_{neg_name}": int((y_test == 1).sum()),
        f"n_test_{others_name}": int((y_test == 2).sum()),
        "oob_score": float(clf.oob_score_) if rf_params.oob_score else None,
    }

    beat_len = X_3d.shape[1]
    lead_importance = clf.feature_importances_.reshape(len(LEAD_NAMES), beat_len).sum(
        axis=1
    )
    lead_importance_df = pd.DataFrame(
        {"lead": LEAD_NAMES, "importance": lead_importance}
    ).sort_values("importance", ascending=False)

    joblib.dump(clf, out_dir / "model.joblib")
    np.save(out_dir / "X_test_flat.npy", X_test)
    np.save(out_dir / "y_test.npy", y_test)
    np.save(out_dir / "test_prob.npy", test_prob)
    pd.DataFrame([split_info]).to_csv(out_dir / "split_info.csv", index=False)
    pd.DataFrame([overall_metrics]).to_csv(out_dir / "test_metrics.csv", index=False)
    per_class_df.to_csv(out_dir / "per_class_metrics.csv", index=False)
    lead_importance_df.to_csv(out_dir / "lead_importance.csv", index=False)

    cm = confusion_matrix(y_test, test_pred, labels=[0, 1, 2])
    report = classification_report(
        y_test, test_pred, target_names=class_names, digits=4, zero_division=0
    )
    np.savetxt(out_dir / "confusion_matrix.txt", cm, fmt="%d")
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)

    print("\n=== Split info ===")
    for k, v in split_info.items():
        print(f"{k}: {v}")

    print("\n=== Overall test metrics ===")
    for k, v in overall_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n=== Per-class metrics ===")
    print(per_class_df.to_string(index=False))

    print(f"\nConfusion matrix (rows=true, cols=pred; order={class_names}):\n{cm}")

    print("\n=== Lead importance (sum over time) ===")
    for _, row in lead_importance_df.iterrows():
        print(f"{row['lead']}: {row['importance']:.6f}")

    print(f"\nModel and results saved to {out_dir.resolve()}")
    return clf, overall_metrics, per_class_df
