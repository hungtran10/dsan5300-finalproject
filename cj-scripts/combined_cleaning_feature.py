from __future__ import annotations
import argparse
import ast
import re
from pathlib import Path
from typing import Final

import pandas as pd
import geopandas as gpd
from shapely.geometry import shape

# ───────────────────────── CONSTANTS ─────────────────────────
RISK_MAPPING: Final = {
    "Risk 1 (High)": 3,
    "Risk 2 (Medium)": 2,
    "Risk 3 (Low)": 1,
    "All": 4,
}

PASS_SET: Final = {"pass", "pass w/ conditions", "pass with conditions"}
CORRECTIONS: Final = {"restuarant": "restaurant"}

HYBRID_STANDARDIZATION: Final = {
    r"(restaurant\s*[/&-]?\s*grocery\s*store|grocery\s*store\s*[/&-]?\s*restaurant)": "Restaurant/Grocery",
    r"(restaurant\s*[/&-]?\s*liquor|restaurant\s+and\s+liquor)": "Restaurant/Liquor",
}

GENERAL_CATEGORIES: Final = {
    r"(restaurant\s*[/&-]?\s*liquor|tavern\s*[/&-]?\s*restaurant)": "Restaurant/Bar",
    r"(alternative\s+school|charter\s+school|private\s+school|school\s+cafeteria|high\s+school\s+kitchen|university\s+cafeteria)": "School",
    r"(day\s*care|daycare.*)": "Daycare",
}

BUCKETS: Final[dict[str, list[str]]] = {
    "Special Event Venue": ["Banquet Hall", "Event Space", "Special Event"],
    "Gas Station": ["Gas Station/Grocery", "Gas Station/Mini Mart"],
    "Church": ["Church Kitchen", "Food Pantry/Church"],
    "Mobile Food": ["Mobile Food Vendor", "Mobile Food Truck"],
    "Culinary School": ["Cooking School", "Pastry School"],
    "Grocery Store": ["Grocery", "Grocery Store/Bakery", "Liquor/Grocery"],
    "Adult Care": ["Long Term Care Facility", "Assisted Living"],
    "Children Services Facility": ["Children'S Services Facility", "After School Program"],
    "Daycare": ["Day Care", "Day Care 1023"],
    "Pop Up Establishment": ["Pop/Up Establishment Host/Tier Ii"],
    "Cafe": ["Coffee Shop", "Cafe"],
    "Hybrid": ["Restaurant/Grocery", "Restaurant/Bar", "Grocery/Butcher"],
}

# ──────────────────────── CLEANING HELPERS ────────────────────────
def clean_city(city):
    if pd.isna(city): return city
    clean = str(city).strip().lower().title()
    return {"Cchicago": "Chicago", "Chcicago": "Chicago", "Chicagoc": "Chicago", "Ch": "Chicago"}.get(clean, clean)

def clean_facility_type(raw_type):
    if pd.isna(raw_type): return raw_type
    clean = str(raw_type).strip().lower()
    for typo, corr in CORRECTIONS.items(): clean = re.sub(rf"\\b{typo}\\b", corr, clean)
    for pat, repl in HYBRID_STANDARDIZATION.items():
        if re.search(pat, clean): clean = repl.lower()
    for pat, repl in GENERAL_CATEGORIES.items():
        if re.search(pat, clean): clean = repl.lower()
    clean = re.sub(r"\s*[/&-]\s*", "/", clean)
    clean = re.sub(r"\s+", " ", clean)
    return clean.title()

def bucket_facility_type(ftype):
    for canon, aliases in BUCKETS.items():
        if ftype in aliases: return canon
    return ftype

# ──────────────────────── BASIC CLEANING ────────────────────────
def basic_casts(df):
    df = df.rename(columns={"License #": "License Number"}).copy()
    df["License Number"] = df["License Number"].astype(str)
    df["Zip"] = df["Zip"].astype(str)
    df["Facility Type"] = df["Facility Type"].astype(str)
    df["Inspection Date"] = pd.to_datetime(df["Inspection Date"])
    return df

def drop_invalid(df):
    df = df[~df["City"].isin(["Los Angeles", "New York", "Inactive"])]
    df = df[df["State"] == "IL"]
    df = df[df["Results"] != "Business Not Located"]
    return df

def add_risk_numeric(df):
    df["Risk"] = df["Risk"].map(RISK_MAPPING).fillna(0).astype(int)
    return df

def value_count_exports(df, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    df["Inspection Type"].value_counts().to_frame().reset_index().to_csv(outdir / "inspection_type.csv", index=False)
    df["Facility Type"].value_counts().to_frame().reset_index().to_csv(outdir / "facility_type_raw.csv", index=False)

def filter_common_facilities(df, min_count=50):
    common = df["Facility Type"].value_counts().loc[lambda s: s >= min_count].index
    filt = df[df["Facility Type"].isin(common)].copy()
    filt["IsPass"] = filt["Results"].str.strip().str.lower().isin(PASS_SET)
    return filt

# ──────────────────────── SPATIAL HELPERS ────────────────────────
def to_point_gdf(df, lon_col, lat_col):
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")

def load_boundaries(path):
    raw = pd.read_csv(path)
    raw["geometry"] = [shape(ast.literal_eval(g)) for g in raw["the_geom"]]
    return gpd.GeoDataFrame(raw, geometry="geometry", crs="EPSG:4326")

def attach_community(gdf, boundaries):
    return gpd.sjoin(gdf, boundaries[["community", "area_numbe", "geometry"]], how="left", predicate="within") \
             .drop(columns="index_right") \
             .rename(columns={"community": "community", "area_numbe": "area_number"})

def crime_summary(crime_path, boundaries):
    crimes = pd.read_csv(crime_path)
    gdf = to_point_gdf(crimes, "longitude", "latitude")
    gdf = attach_community(gdf, boundaries)
    summary = gdf.groupby("area_number").agg(total_crimes=("id", "count"), arrest_rate=("arrest", lambda x: pd.to_numeric(x).mean())).reset_index()
    crime_types = gdf.groupby(["area_number", "primary_type"]).size().unstack(fill_value=0).add_prefix("count_").reset_index()
    return summary.merge(crime_types, on="area_number", how="left").rename(columns=str.lower)

# ──────────────────────── FEATURE ENGINEERING ────────────────────────
def run_feature_engineering(df, outdir):
    import numpy as np

    df['violation_count'] = df['Violations'].fillna('').apply(lambda x: len(re.findall(r'\d+\.', x)))

    critical_keywords = ['serious', 'critical', 'food borne', 'illness', 'vomit']
    df['critical_violation_flag'] = df['Violations'].apply(lambda text: int(any(keyword in str(text).lower() for keyword in critical_keywords)))
    df['first_violation_code'] = df['Violations'].str.extract(r'^(\d+)\.', expand=False)

    df['inspection_year'] = df['Inspection Date'].dt.year
    df['inspection_month'] = df['Inspection Date'].dt.month
    df['inspection_dayofweek'] = df['Inspection Date'].dt.dayofweek
    df['inspection_quarter'] = df['Inspection Date'].dt.quarter
    df['is_weekend'] = df['inspection_dayofweek'] >= 5

    df['inspections_per_business'] = df.groupby('License Number')['Inspection ID'].transform('count')
    fail_rate = df.groupby('License Number')['IsPass'].apply(lambda x: 1 - x.mean()).rename("fail_rate_per_business")
    df = df.merge(fail_rate, on='License Number', how='left')

    def violation_score(text):
        if pd.isnull(text): return 0
        text = text.lower()
        return 3 * text.count('critical') + 2 * text.count('serious') + text.count('violation')

    df['violation_score'] = df['Violations'].apply(violation_score)

    def bin_violations(count):
        if count == 0: return 'None'
        elif count <= 2: return 'Low'
        elif count <= 5: return 'Medium'
        else: return 'High'

    df['violation_level'] = df['violation_count'].apply(bin_violations)
    df = pd.get_dummies(df, columns=['violation_level'], drop_first=True)

    df['zip_prefix'] = df['Zip'].astype(str).str[:3]
    df['risk_x_violations'] = df['Risk'].astype(float) * df['violation_count']
    df['first_inspection'] = df.sort_values('Inspection Date').groupby('License Number')['Inspection Date'].transform('min')
    df['is_first_inspection'] = (df['Inspection Date'] == df['first_inspection']).astype(int)

    encode_cols = ['Facility Type', 'City', 'State', 'Risk', 'Zip', 'Results', 'first_violation_code']
    df[encode_cols] = df[encode_cols].fillna('Unknown')

    top_cities = df['City'].value_counts().nlargest(15).index
    df['City'] = df['City'].apply(lambda x: x if x in top_cities else 'Other')
    top_zips = df['Zip'].astype(str).value_counts().nlargest(15).index
    df['Zip'] = df['Zip'].astype(str).apply(lambda x: x if x in top_zips else 'Other')

    df_encoded = pd.get_dummies(df, columns=encode_cols, prefix=encode_cols, drop_first=True)
    df_encoded.to_csv(outdir / "Encoded_Food_Inspections.csv", index=False)

    pre_inspection_cols = ['Facility Type', 'City', 'State', 'Risk', 'Zip']
    df[pre_inspection_cols] = df[pre_inspection_cols].fillna('Unknown')
    df['City'] = df['City'].apply(lambda x: x if x in top_cities else 'Other')
    df['Zip'] = df['Zip'].astype(str).apply(lambda x: x if x in top_zips else 'Other')
    df_encoded_pre = pd.get_dummies(df, columns=pre_inspection_cols, prefix=pre_inspection_cols, drop_first=True)
    df_encoded_pre.to_csv(outdir / "Encoded_PreInspectionOnly.csv", index=False)

# ──────────────────────────── MAIN ────────────────────────────
def main(args):
    df = pd.read_csv(args.input)
    df = basic_casts(df)
    df['City'] = df['City'].apply(clean_city)
    df = drop_invalid(df)
    df = add_risk_numeric(df)
    df['Facility Type'] = df['Facility Type'].apply(clean_facility_type).apply(bucket_facility_type)
    value_count_exports(df, args.outdir)

    df_clean = filter_common_facilities(df)
    df_clean.to_csv(args.outdir / "Clean_Food_Inspections.csv", index=False)

    boundaries = load_boundaries(args.boundaries)
    gdf_inspect = attach_community(to_point_gdf(df_clean, "Longitude", "Latitude"), boundaries)

    crime_df = crime_summary(args.crimes, boundaries)
    acs_df = pd.read_csv(args.acs).drop(columns=["acs_year", "record_id"]).rename(columns={"community_area": "community"})
    final = gdf_inspect.merge(crime_df, on="area_number", how="left").merge(acs_df, on="community", how="left")
    final.to_csv(args.outdir / "Food_Inspections_Enriched.csv", index=False)

    if args.features:
        run_feature_engineering(final.copy(), args.outdir)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Clean & enrich Chicago food inspections dataset")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--boundaries", required=True, type=Path)
    p.add_argument("--crimes", required=True, type=Path)
    p.add_argument("--acs", required=True, type=Path)
    p.add_argument("--outdir", default=Path("../Datasets"), type=Path)
    p.add_argument("--features", action="store_true", help="Run feature engineering pipeline after cleaning")
    main(p.parse_args())
