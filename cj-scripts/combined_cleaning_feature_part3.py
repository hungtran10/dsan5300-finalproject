"""
Reads Encoded_Food_Inspections_v2.csv (the “full” feature table)
and writes Encoded_PreInspectionOnly_v2.csv that keeps **only**
attributes available *before an inspector walks through the door*.

The rules applied below are intentionally conservative:

• Drop the label column **IsPass**.

• Drop every column that
    – starts with   "violation_"             (post-inspection findings)
    – starts with   "first_violation_code"   (same)
    – contains      "critical_violation_flag"
    – contains      "violation_score"
    – contains      "risk_x_violations"      (multiplies risk × violation_count)

• Drop any of the one-hot dummies beginning with **Results_***.

Everything else (historical business stats, community metrics, schedule info,
rodent complaints per-100 k, etc.) is retained because it can be known or
estimated before the visit.

Usage
~~~~~
python cj-scripts/combined_cleaning_feature_part3.py --inputfile ./Datasets/Encoded_Food_Inspections_v2.csv --outputfile ./Datasets/Encoded_PreInspectionOnly_v2.csv
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import pandas as pd

# ─────────────────────────── configuration ──────────────────────────                    # label

PREFIX_DROPS = (
    "violation_",           # violation_count, violation_level_*, etc.
    "first_violation_code",
    "Results_",             # Results_Pass dummy columns (post-hoc only)
)

CONTAINS_DROPS = (
    "critical_violation_flag",
    "violation_score",
    "risk_x_violations",
)

# ─────────────────────────── helpers ────────────────────────────────
def select_preinspection(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DF with only pre-inspection-available columns."""
    keep_cols = []
    for col in df.columns:
        if col.startswith(PREFIX_DROPS):
            continue
        if any(token in col for token in CONTAINS_DROPS):
            continue
        keep_cols.append(col)
    return df[keep_cols]

# ─────────────────────────── main routine ───────────────────────────
def main(inputfile: Path, outputfile: Path) -> None:
    df_full = pd.read_csv(inputfile)
    print(f"Loaded {len(df_full):,} rows  {df_full.shape[1]} columns", file=sys.stderr)

    df_pre = select_preinspection(df_full)

    df_pre.to_csv(outputfile, index=False)
    print(
        f"Saved {outputfile.name}  "
        f"({df_pre.shape[1]} cols)  "
        f"{df_pre.memory_usage(deep=True).sum()/1_048_576:,.1f} MB",
        file=sys.stderr,
    )

# ───────────────────────── entry point ──────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Create Encoded_PreInspectionOnly_v2.csv from the v2 full table"
    )
    ap.add_argument("--inputfile", required=True, type=Path,
                    help="Path to Encoded_Food_Inspections_v2.csv")
    ap.add_argument("--outputfile", required=True, type=Path,
                    help="Destination CSV for the pre-inspection-only view")

    main(**vars(ap.parse_args()))
