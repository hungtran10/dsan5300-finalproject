"""
Reusable PyTorch `Dataset` for the encoded Chicago food-inspection data.

Key logic
─────────
• drops text / ID columns that are meaningless to a model
• fills NaNs:
      pct_*        → 0 .0      (share)
      *_per100k    → 0 .0      (rate)
      everything else → column mean
• z-scores all *non-fraction* features
• stores tensors ready for DataLoader

A short smoke-test is executed when running this file directly:
    $ python inspections_dataset.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ───────────────────────── Dataset class ────────────────────────────
class InspectionsDataset(Dataset):
    """
    Parameters
    ----------
    csv_path : str | Path
        Path to *Encoded_Food_Inspections_v2.csv*  **or**
        *Encoded_PreInspectionOnly_v2.csv*.
    verbose : bool, default False
        If True, prints the final list of feature columns.
    """

    EXCLUDED_COLS = {
        # label
        "IsPass",
        # obvious IDs / strings
        "Inspection_ID", "Inspection_ID", "Inspection_Date", "Inspection_Date",
        "Violations", "Location", "Address", "DBA_Name", "AKA_Name",
        "License_Number", "License_Number", "geometry",
        # geo join keys (already baked into other features)
        "community", "community_name", "area_number",
        # lat/lon duplicates
        "Latitude", "Longitude",
        # post-inspection results dummies
        "Results_No_Entry", "Results_Not_Ready", "Results_Out_of_Business",
        "Results_Pass", "Results_Pass_w/_Conditions",
        # engineered helpers we don’t want the model to cheat with
        "first_inspection",
        # temporal one-hots we left in the table but don’t need here
        "zip_prefix", "inspection_month", "inspection_dayofweek",
        "inspection_quarter", "inspection_year",
        # the raw rodent count (keep *per100k* only)
        "rodent_complaints", "fail_rate_per_business"
    }

    # --------------------
    def __init__(self, csv_path: str | Path, *, verbose: bool = False) -> None:
        df = pd.read_csv(csv_path)

        # drop rows w/o community (rare parsing artefacts)
        if "community" in df.columns:
            df = df[df["community"].notna()]

        # ---------------- label -----------------
        y_np = df["IsPass"].astype(np.float32).to_numpy()  # (N,)

        # -------------- feature frame ----------
        X_df = (
            df.drop(columns=self.EXCLUDED_COLS, errors="ignore")
              .select_dtypes(include=["number", "bool"])   # keep numeric/bool only
              .astype(float)                               # cast bool→float
        )

        FRACTION_COLS = [c for c in X_df.columns if c.startswith("pct_")]
        PER100K_COLS  = [c for c in X_df.columns if c.endswith("_per100k")]

        # Fill missing values
        if FRACTION_COLS:
            X_df[FRACTION_COLS] = X_df[FRACTION_COLS].fillna(0.0)
        if PER100K_COLS:
            X_df[PER100K_COLS]  = X_df[PER100K_COLS].fillna(0.0)
        X_df = X_df.fillna(X_df.mean(numeric_only=True))

        # Standardise everything except the 0–1 shares
        scale_cols = X_df.columns.difference(FRACTION_COLS)
        means = X_df[scale_cols].mean()
        stds  = X_df[scale_cols].std().replace(0, 1.0)  # avoid /0
        X_df[scale_cols] = (X_df[scale_cols] - means) / stds

        # -------------- tensors ----------------
        self.X = torch.tensor(X_df.to_numpy(), dtype=torch.float32)
        self.y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

        self.feature_names = list(X_df.columns)

        if verbose:
            print("\nInspectionsDataset — final feature set")
            for col in self.feature_names:
                print(f"  • {col}")
            print(f"Total features: {len(self.feature_names)}\n")

    # --------------------
    def __len__(self) -> int:
        return len(self.X)

    # --------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
