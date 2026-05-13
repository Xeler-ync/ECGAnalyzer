import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import ast
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils._config import PATH
from utils._data import Y

TARGET_THRESHOLD = 100
DEFAULT_OUT_DIR = Path("results/label_audit")
SUPERCLASS_ORDER = ["NORM", "HYP", "MI", "CD", "STTC"]
REPO_ROOT = Path(__file__).resolve().parent


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
    if scp_codes.get("NORM", 0) < threshold:
        return False
    others = [k for k, v in scp_codes.items() if k != "NORM" and v >= threshold]
    return not others


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


def build_label_audit(
    threshold: int = TARGET_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scp_df = load_scp_statements()
    superclass_lookup = build_superclass_lookup(scp_df)

    rows = []
    for idx in range(len(Y)):
        record = Y.iloc[idx]
        scp_codes = ensure_dict(record["scp_codes"])
        superclasses = map_superclasses(
            scp_codes, superclass_lookup, threshold=threshold
        )
        is_norm = int(scp_codes.get("NORM", 0) >= threshold)
        pure_norm = int(is_pure_norm(scp_codes, threshold=threshold))
        is_lvh = int(scp_codes.get("LVH", 0) >= threshold)
        n_codes = int(sum(v >= threshold for v in scp_codes.values()))
        single_superclass_only = int(len(superclasses) == 1)

        row = {
            "row_index": idx,
            "ecg_id": Y.index[idx],
            "patient_id": record.get("patient_id", None),
            "is_lvh": is_lvh,
            "is_norm": is_norm,
            "is_pure_norm": pure_norm,
            "keep_for_binary_lvh_norm": int(is_lvh == 1 or pure_norm == 1),
            "lvh_vs_norm_label": (
                (1 if is_lvh == 1 else 0) if (is_lvh == 1 or pure_norm == 1) else np.nan
            ),
            "n_positive_scp_codes": n_codes,
            "diagnostic_superclasses": ";".join(superclasses),
            "n_superclasses": len(superclasses),
            "single_superclass_only": single_superclass_only,
            "raw_scp_codes": json.dumps(scp_codes, ensure_ascii=False),
        }

        for sc in SUPERCLASS_ORDER:
            row[f"superclass_{sc.lower()}"] = int(sc in superclasses)

        if len(superclasses) == 1:
            row["primary_superclass"] = superclasses[0]
        else:
            row["primary_superclass"] = "MIXED" if superclasses else "UNKNOWN"

        rows.append(row)

    audit_df = pd.DataFrame(rows)

    summary_rows = [
        {"metric": "total_records", "value": len(audit_df)},
        {
            "metric": "binary_keep_lvh_or_pure_norm",
            "value": int(audit_df["keep_for_binary_lvh_norm"].sum()),
        },
    ]
    summary_rows.append(
        {"metric": "lvh_records", "value": int(audit_df["is_lvh"].sum())}
    )
    summary_rows.append(
        {"metric": "pure_norm_records", "value": int(audit_df["is_pure_norm"].sum())}
    )
    summary_rows.extend(
        {
            "metric": f"records_with_{sc}",
            "value": int(audit_df[f"superclass_{sc.lower()}"].sum()),
        }
        for sc in SUPERCLASS_ORDER
    )
    summary_rows.extend(
        {"metric": f"primary_superclass::{label_name}", "value": int(count)}
        for label_name, count in audit_df["primary_superclass"]
        .value_counts(dropna=False)
        .items()
    )
    summary_df = pd.DataFrame(summary_rows)
    return audit_df, summary_df


def main():
    parser = argparse.ArgumentParser(description="Build label audit table for PTB-XL")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--threshold", type=int, default=TARGET_THRESHOLD)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_df, summary_df = build_label_audit(threshold=args.threshold)
    audit_df.to_csv(out_dir / "label_audit.csv", index=False)
    summary_df.to_csv(out_dir / "label_audit_summary.csv", index=False)

    multiclass_df = audit_df[
        (audit_df["single_superclass_only"] == 1)
        & (audit_df["primary_superclass"].isin(SUPERCLASS_ORDER))
    ].copy()
    multiclass_df.to_csv(
        out_dir / "label_candidates_multiclass_superclass.csv", index=False
    )

    print("=" * 80)
    print("PTB-XL label audit completed")
    print("=" * 80)
    print(f"Total records                 : {len(audit_df)}")
    print(
        f"Binary keep (LVH or pure NORM): {int(audit_df['keep_for_binary_lvh_norm'].sum())}"
    )
    print(f"LVH records                  : {int(audit_df['is_lvh'].sum())}")
    print(f"Pure NORM records            : {int(audit_df['is_pure_norm'].sum())}")
    print(f"Single-superclass candidates : {len(multiclass_df)}")
    print(f"Saved outputs to             : {out_dir.resolve()}")


if __name__ == "__main__":
    main()
