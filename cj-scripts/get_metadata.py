"""

ONE-STOP script that generates **all external CSVs** needed for the
inspection-project pipeline:

    • Datasets/rodent_enrichment.csv
    • Datasets/crimes_last2years.csv
    • Datasets/community_boundaries.csv
    • Datasets/acs_by_community.csv

Run:

    python cj-scripts/get_city_datasets.py
"""

# ───────────────────────── imports ───────────────────────────
import os, sys, json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sodapy import Socrata

# ───────────────────────── constants ─────────────────────────
DATA_DIR = "Datasets"
Path(DATA_DIR).mkdir(exist_ok=True)

APP_TOKEN   = None        # INSERT your app-token (str) for higher rate limits
DOMAIN      = "data.cityofchicago.org"
CLIENT      = Socrata(DOMAIN, APP_TOKEN, timeout=30)

# 311 datasets / codes
SR_DATASET  = "v6vf-nfxy"     # 311 Service Requests (modern)
RODENT_CODE = "SGA"           # sr_short_code: Rodent Baiting / Rat Complaint

# Crime, boundaries, ACS
CRIME_DATASET     = "ijzp-q8t2"
BOUNDARY_DATASET  = "igwz-8jzy"
ACS_DATASET       = "t68z-cikk"

# Mapping {area_number:int → community name:str}
COMMUNITY_AREA_MAP = {         # full mapping—unchanged
    25:"AUSTIN",71:"AUBURN GRESHAM",50:"PULLMAN",34:"ARMOUR SQUARE",
    33:"NEAR SOUTH SIDE",14:"ALBANY PARK",32:"LOOP",4:"LINCOLN SQUARE",
    31:"LOWER WEST SIDE",35:"DOUGLAS",13:"NORTH PARK",44:"CHATHAM",
    16:"IRVING PARK",46:"SOUTH CHICAGO",8:"NEAR NORTH SIDE",23:"HUMBOLDT PARK",
    2:"WEST RIDGE",58:"BRIGHTON PARK",27:"EAST GARFIELD PARK",28:"NEAR WEST SIDE",
    6:"LAKE VIEW",7:"LINCOLN PARK",57:"ARCHER HEIGHTS",64:"CLEARING",
    24:"WEST TOWN",52:"EAST SIDE",63:"GAGE PARK",38:"GRAND BOULEVARD",
    21:"AVONDALE",66:"CHICAGO LAWN",70:"ASHBURN",68:"ENGLEWOOD",
    69:"GREATER GRAND CROSSING",75:"MORGAN PARK",43:"SOUTH SHORE",
    19:"BELMONT CRAGIN",1:"ROGERS PARK",77:"EDGEWATER",15:"PORTAGE PARK",
    62:"WEST ELSDON",74:"MOUNT GREENWOOD",60:"BRIDGEPORT",20:"HERMOSA",
    37:"FULLER PARK",11:"JEFFERSON PARK",61:"NEW CITY",56:"GARFIELD RIDGE",
    5:"NORTH CENTER",26:"WEST GARFIELD PARK",49:"ROSELAND",29:"NORTH LAWNDALE",
    76:"OHARE",0:"0",72:"BEVERLY",53:"WEST PULLMAN",12:"FOREST GLEN",
    3:"UPTOWN",22:"LOGAN SQUARE",54:"RIVERDALE",30:"SOUTH LAWNDALE",
    36:"OAKLAND",41:"HYDE PARK",39:"KENWOOD",65:"WEST LAWN",
    59:"MCKINLEY PARK",18:"MONTCLARE",51:"SOUTH DEERING",40:"WASHINGTON PARK",
    73:"WASHINGTON HEIGHTS",42:"WOODLAWN",17:"DUNNING",9:"EDISON PARK",
    45:"AVALON PARK",55:"HEGEWISCH",10:"NORWOOD PARK",67:"WEST ENGLEWOOD",
    48:"CALUMET HEIGHTS",47:"BURNSIDE"
}

# ───────────────────────── rodent part ───────────────────────
def build_rodent_enrichment():
    """
    Creates Datasets/rodent_enrichment.csv with:
        community_area, community_name, population,
        rodent_complaints, rodent_complaints_per1000
    """
    since = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%dT%H:%M:%S")
    where = (
        f"created_date >= '{since}' "
        "AND sr_short_code = 'SGA' "
        "AND community_area IS NOT NULL"
    )

    print("\n==== Rodent 311 extraction ====")
    rec, offset, limit = [], 0, 5_000
    while True:
        batch = CLIENT.get(SR_DATASET,
                           select="community_area",
                           where=where,
                           limit=limit,
                           offset=offset)
        if not batch:
            break
        rec.extend(batch)
        offset += limit
        print(f"   fetched {offset:,} rows", end="\r", flush=True)
        if offset >= 1_000_000:           # extra-safe cutoff
            break
    print(f"\n✔️  Total SGA rows pulled: {len(rec):,}")

    df = pd.DataFrame.from_records(rec)
    df["community_area"] = pd.to_numeric(df["community_area"], errors="coerce")
    rodent = (
        df.groupby("community_area").size()
          .rename("rodent_complaints")
          .reset_index()
    )
    print(f"✔️  {len(rodent)} community areas with ≥1 complaint")

    # Population from boundary dataset
    bd = CLIENT.get(BOUNDARY_DATASET)
    pop_df = pd.DataFrame.from_records(bd)
    pop_df["community_area"] = pd.to_numeric(pop_df["area_number"], errors="coerce")
    pop_df["population"]     = pd.to_numeric(pop_df["population"],   errors="coerce")
    pop_df = pop_df[["community_area","population"]]

    rodent = pop_df.merge(rodent, on="community_area", how="left")
    rodent["rodent_complaints"].fillna(0, inplace=True)
    rodent["rodent_complaints_per1000"] = (
        rodent["rodent_complaints"] / rodent["population"] * 1_000
    )
    rodent["community_name"] = rodent["community_area"].map(COMMUNITY_AREA_MAP)

    print("\nRodent enrichment preview:")
    print(rodent.head(), "\n")
    print(rodent.info(), "\n")

    out = Path(DATA_DIR, "rodent_enrichment.csv")
    rodent.to_csv(out, index=False)
    print(f"Wrote {len(rodent)} rows → {out}")

# ───────────────────────── crime / ACS part ──────────────────
def build_crime_boundary_acs():
    """
    Writes three CSVs:
        • crimes_last2years.csv
        • community_boundaries.csv
        • acs_by_community.csv
    """
    print("\n==== Crime, boundaries & ACS extraction ====")

    # ---- Crime (2 years) ----------------------------------
    twoyr = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%dT%H:%M:%S")
    crime_where = f"date >= '{twoyr}'"
    crime_limit, crime_offset, crime_rec = 10_000, 0, []

    print("Pulling crime data (this can take a while)…")
    while True:
        batch = CLIENT.get(CRIME_DATASET,
                           where=crime_where,
                           limit=crime_limit,
                           offset=crime_offset)
        if not batch:
            break
        crime_rec.extend(batch)
        crime_offset += crime_limit
        print(f"   fetched {crime_offset:,} crime rows", end="\r", flush=True)
    print(f"\nTotal crime rows: {len(crime_rec):,}")

    pd.DataFrame.from_records(crime_rec).to_csv(
        Path(DATA_DIR, "crimes_last2years.csv"), index=False
    )

    # ---- Boundaries ---------------------------------------
    print("Pulling community boundaries …")
    boundaries = CLIENT.get(BOUNDARY_DATASET)
    bd_df = pd.DataFrame.from_records(boundaries)
    bd_df.to_csv(Path(DATA_DIR, "community_boundaries.csv"), index=False)
    print(f"Boundaries rows: {len(bd_df):,}")

    # ---- ACS ----------------------------------------------
    print("Pulling ACS socio-economic data …")
    acs_rec, acs_offset, acs_limit = [], 0, 10_000
    while True:
        batch = CLIENT.get(ACS_DATASET, limit=acs_limit, offset=acs_offset)
        if not batch:
            break
        acs_rec.extend(batch); acs_offset += acs_limit
    print(f"✔️  ACS rows: {len(acs_rec):,}")

    pd.DataFrame.from_records(acs_rec).to_csv(
        Path(DATA_DIR, "acs_by_community.csv"), index=False
    )

# ───────────────────────── main entry ────────────────────────
def main():
    try:
        build_rodent_enrichment()
        build_crime_boundary_acs()
        print("\nAll datasets generated successfully.")
    finally:
        CLIENT.close()

if __name__ == "__main__":
    main()
