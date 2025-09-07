
"""
----------------------------------
From Encoded_Food_Inspections.csv produce Encoded_Food_Inspections_v2.csv

Improvements over v1
────────────────────
• Drops location & other redundant fields
• Converts crime counts  → per-100 k population
• Converts income buckets → per-100 k population
• Converts age-sex & race counts → % of total population
• Attaches rodent-complaint data (per-100 k) by community area
• Replaces any space in a column name with “_”

Usage
~~~~~
python cj-scripts/combined_cleaning_feature_part2.py --inputfile  ./Datasets/Encoded_Food_Inspections.csv --outputfile ./Datasets/Encoded_Food_Inspections_v2.csv
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

# ─────────────────────────── column groups ──────────────────────────
DROP_TOP_LEVEL = [
    "Latitude", "Longitude", "Location",
    "total_crimes",                                   # granular crime counts stay
    "white_not_hispanic_or_latino", "hispanic_or_latino"   # duplicates
]

AGE_SEX_COLS = [
    "male_0_to_17","male_18_to_24","male_25_to_34","male_35_to_49",
    "male_50_to_64","male_65","female_0_to_17","female_18_to_24",
    "female_25_to_34","female_35_to_49","female_50_to_64","female_65",
]

RACE_COLS = [
    "white","black_or_african_american","american_indian_or_alaska",
    "asian","native_hawaiin_or_pacific","other_race","multiracial",
]

INCOME_BUCKET_COLS = [
    "under_25_000", "_25_000_to_49_999", "_50_000_to_74_999",
    "_75_000_to_125_000", "_125_000"
]

CRIME_PREFIX = "count_"          # any column starting with this

RODENT_ENRICH_PATH = Path("Datasets/rodent_enrichment.csv")

# ───────────────────────── helper transforms ────────────────────────
def counts_to_per100k(df: pd.DataFrame, cols: list[str],
                      pop: pd.Series, suffix="_per100k") -> None:
    """Divides *cols* by population and scales to 100 000."""
    safe_pop = pop.replace(0, np.nan)
    for col in cols:
        if col not in df.columns:
            continue
        df[f"{col}{suffix}"] = df[col] / safe_pop * 100_000
        df.drop(columns=col, inplace=True)

def counts_to_fraction(df: pd.DataFrame, cols: list[str],
                       pop: pd.Series, prefix="pct_") -> None:
    """Converts raw counts to 0-1 fractions of population."""
    safe_pop = pop.replace(0, np.nan)
    for col in cols:
        if col not in df.columns:
            continue
        df[f"{prefix}{col}"] = df[col] / safe_pop
        df.drop(columns=col, inplace=True)

# ─────────────────────────── main routine ───────────────────────────
def main(inputfile: Path, outputfile: Path) -> None:
    df = pd.read_csv(inputfile)
    print(f"Loaded {len(df):,} rows  {df.shape[1]} columns", file=sys.stderr)

    # Drop undesired top-level columns
    df.drop(columns=DROP_TOP_LEVEL, errors="ignore", inplace=True)

    # Population sanity check
    if "total_population" not in df.columns:
        raise KeyError("Column 'total_population' is required for rate conversions")
    pop = df["total_population"].astype(float)

    # Crime counts  → per-100 k
    crime_cols = [c for c in df.columns if c.startswith(CRIME_PREFIX)]
    counts_to_per100k(df, crime_cols, pop)

    # Income buckets → per-100 k
    counts_to_per100k(df, INCOME_BUCKET_COLS, pop)

    # 5 Age-sex & race counts → percentage of population
    counts_to_fraction(df, AGE_SEX_COLS, pop)
    counts_to_fraction(df, RACE_COLS, pop)

    # Attach rodent-complaint enrichment
    if not RODENT_ENRICH_PATH.exists():
        raise FileNotFoundError(f"{RODENT_ENRICH_PATH} not found. "
                                "Run get_rodent_enrichment.py first.")

    rodent = pd.read_csv(RODENT_ENRICH_PATH)


    # bring only the numeric key + the per-100 k column
    rodent_subset = rodent[["community_area", "rodent_complaints"]]

    # merge: area_number  (inspections)  ↔  community_area (rodent)
    df = df.merge(
        rodent_subset,
        left_on="area_number",
        right_on="community_area",
        how="left"
    )

    df["rodent_complaints_per100k"] = (
        df["rodent_complaints"] / pop.replace(0, np.nan) * 100_000
    )

    # drop the temp key column that came from enrichment
    df.drop(columns=["community_area"], inplace=True)

    # 7️Final tidy-up: replace spaces with underscores in remaining column names
    df.columns = [c.replace(" ", "_") for c in df.columns]

    # 8Write out
    df.to_csv(outputfile, index=False)
    mem_mb = df.memory_usage(deep=True).sum() / 1_048_576
    print(f"Saved {outputfile.name}  ({df.shape[1]} cols)  {mem_mb:,.1f} MB", file=sys.stderr)

# ───────────────────────── entry point ──────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Produce Encoded_Food_Inspections_v2.csv with rodent data.")
    ap.add_argument("--inputfile",  required=True, type=Path,
                    help="Path to Encoded_Food_Inspections.csv")
    ap.add_argument("--outputfile", required=True, type=Path,
                    help="Destination CSV file (v2)")
    main(**vars(ap.parse_args()))
