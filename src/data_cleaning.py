"""
SmartFuel AI — Data Cleaning Pipeline
======================================
Loads the raw sales dataset, applies a comprehensive cleaning routine, and
applies the filling-station product mapping so that ONLY products appropriate
for a convenience store are retained in the processed dataset.

Academic note
-------------
The original dataset uses generic product IDs (P0001–P0020) with categories
such as Electronics, Clothing, Toys and Furniture that are NOT sold inside a
filling-station convenience store.  A transparent product-mapping layer
(data/product_mapping.csv) re-assigns each product ID a realistic filling-
station product name and category.  The original raw file is preserved
unchanged.  This mapping is clearly documented as a prototype assumption for
academic purposes.
"""

import os
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT, "data", "raw", "sales_data.csv")
MAPPING_PATH = os.path.join(ROOT, "data", "product_mapping.csv")
PROCESSED_PATH = os.path.join(ROOT, "data", "processed", "filling_station_sales.csv")

# ── Required columns ───────────────────────────────────────────────────────
REQUIRED_COLS = [
    "Date", "Store ID", "Product ID", "Category", "Region",
    "Inventory Level", "Units Sold", "Units Ordered", "Price",
    "Discount", "Weather Condition", "Promotion", "Competitor Pricing",
    "Seasonality", "Demand",
]

# ── Original categories that do NOT belong in a filling-station store ──────
NON_FS_CATEGORIES = {"Electronics", "Clothing", "Toys", "Furniture"}


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV and return a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw dataset not found at {path}.\n"
            "Please place sales_data.csv inside data/raw/."
        )
    df = pd.read_csv(path, low_memory=False)
    return df


def load_product_mapping(path: str = MAPPING_PATH) -> pd.DataFrame:
    """Load the filling-station product mapping table."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Product mapping not found at {path}.\n"
            "Please ensure data/product_mapping.csv exists."
        )
    mapping = pd.read_csv(path)
    # Normalise the Allowed column to boolean
    mapping["Allowed"] = mapping["Allowed"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(False)
    return mapping


def validate_columns(df: pd.DataFrame) -> list:
    """Return a list of missing required columns."""
    return [c for c in REQUIRED_COLS if c not in df.columns]


def clean(
    raw_path: str = RAW_PATH,
    mapping_path: str = MAPPING_PATH,
    save_path: str = PROCESSED_PATH,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the full cleaning pipeline.

    Returns
    -------
    df_clean : pd.DataFrame
        Processed, filling-station-only dataset ready for feature engineering.
    stats : dict
        Data-quality statistics for display in the dashboard.
    """
    stats: dict = {}

    # ── 1. Load raw data ───────────────────────────────────────────────────
    df = load_raw(raw_path)
    stats["raw_rows"] = len(df)
    stats["raw_columns"] = len(df.columns)
    if verbose:
        print(f"[Cleaning] Loaded {stats['raw_rows']:,} rows from {raw_path}")

    # ── 2. Validate required columns ───────────────────────────────────────
    missing_cols = validate_columns(df)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # ── 3. Parse dates ─────────────────────────────────────────────────────
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    bad_dates = df["Date"].isna().sum()
    stats["bad_dates_removed"] = int(bad_dates)
    df = df.dropna(subset=["Date"])

    # ── 4. Standardise string columns ─────────────────────────────────────
    for col in ["Store ID", "Product ID", "Category", "Region",
                "Weather Condition", "Seasonality"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # ── 5. Convert numeric columns ─────────────────────────────────────────
    numeric_cols = [
        "Inventory Level", "Units Sold", "Units Ordered",
        "Price", "Discount", "Competitor Pricing", "Demand",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 6. Drop rows missing Demand (the target) ───────────────────────────
    before = len(df)
    df = df.dropna(subset=["Demand"])
    stats["missing_demand_removed"] = int(before - len(df))

    # ── 7. Fill missing numeric features with sensible defaults ───────────
    fill_defaults = {
        "Inventory Level": 0,
        "Units Sold": 0,
        "Units Ordered": 0,
        "Discount": 0,
        "Competitor Pricing": df["Price"].median() if "Price" in df.columns else 0,
    }
    filled_count = 0
    for col, default in fill_defaults.items():
        if col in df.columns:
            n = df[col].isna().sum()
            filled_count += n
            df[col] = df[col].fillna(default)
    stats["missing_values_filled"] = int(filled_count)

    # Fill missing Price with median per product
    if "Price" in df.columns:
        med_price = df.groupby("Product ID")["Price"].transform("median")
        n_price = df["Price"].isna().sum()
        df["Price"] = df["Price"].fillna(med_price).fillna(df["Price"].median())
        stats["missing_values_filled"] = stats.get("missing_values_filled", 0) + int(n_price)

    # ── 8. Handle Promotion / Epidemic columns ─────────────────────────────
    for col in ["Promotion", "Epidemic"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # ── 9. Remove duplicate rows ───────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["Date", "Store ID", "Product ID"])
    stats["duplicates_removed"] = int(before - len(df))

    # ── 10. Remove impossible negative values ──────────────────────────────
    before = len(df)
    df = df[
        (df["Demand"] >= 0) &
        (df["Inventory Level"] >= 0) &
        (df["Price"] > 0)
    ]
    stats["invalid_records_removed"] = int(before - len(df))

    # ── 11. Apply product mapping — filling-station domain filter ──────────
    # The original dataset contains Electronics, Clothing, Toys, Furniture.
    # These categories do NOT exist in a filling-station convenience store.
    # Step A: flag rows whose ORIGINAL category is a non-FS category
    rows_before_mapping = len(df)
    df["_orig_category"] = df["Category"].copy()

    # Load mapping
    mapping = load_product_mapping(mapping_path)
    allowed_products = set(
        mapping.loc[mapping["Allowed"], "Product ID"].astype(str)
    )

    # Build lookup dictionaries from mapping
    pid_to_name = dict(zip(
        mapping["Product ID"].astype(str),
        mapping["Product Name"]
    ))
    pid_to_fs_category = dict(zip(
        mapping["Product ID"].astype(str),
        mapping["Filling Station Category"]
    ))
    pid_to_subcategory = dict(zip(
        mapping["Product ID"].astype(str),
        mapping["Subcategory"]
    ))

    # Step B: keep only rows whose Product ID is in the allowed set
    df = df[df["Product ID"].astype(str).isin(allowed_products)].copy()
    stats["non_fs_rows_removed"] = int(rows_before_mapping - len(df))

    # NOTE: In this dataset all 20 product IDs appear across all original categories
    # (Electronics, Clothing, Toys, Furniture, Groceries). The product mapping allowlist
    # covers all 20 IDs and re-assigns them to correct FS categories. No rows are dropped
    # by the ID filter because the mapping designates all products as Allowed=True.
    # The non-FS original categories (Electronics, Clothing, Toys, Furniture) are
    # REPLACED by the correct FS category — they do not reach the application.

    # Step C: replace Category with the filling-station category from mapping
    df["Category"] = df["Product ID"].map(pid_to_fs_category).fillna(df["Category"])
    df["Product Name"] = df["Product ID"].map(pid_to_name).fillna(df["Product ID"])
    df["Subcategory"] = df["Product ID"].map(pid_to_subcategory).fillna("General")

    # Drop the temporary original-category column
    df = df.drop(columns=["_orig_category"])

    # ── 12. Standardise Weather Condition values ───────────────────────────
    weather_map = {
        "sunny": "Sunny", "cloudy": "Cloudy",
        "rainy": "Rainy", "snowy": "Overcast",
        "rain": "Rainy", "snow": "Overcast",
        "overcast": "Overcast", "clear": "Sunny",
    }
    df["Weather Condition"] = (
        df["Weather Condition"]
        .str.lower()
        .map(lambda x: weather_map.get(x, "Sunny"))
    )

    # ── 13. Standardise Seasonality for South African context ──────────────
    # Dataset uses Northern-hemisphere seasons; we relabel for SA but keep
    # the relative meaning (high/low demand periods) intact.
    # Summer in the dataset corresponds to SA peak retail (Dec–Feb).
    # We document this as a prototype assumption.
    season_map = {
        "winter": "Winter", "summer": "Summer",
        "autumn": "Autumn", "spring": "Spring",
        "fall": "Autumn",
    }
    df["Seasonality"] = (
        df["Seasonality"]
        .str.lower()
        .map(lambda x: season_map.get(x, "Summer"))
    )

    # ── 14. Rename Region labels to SA-appropriate branch names ───────────
    region_map = {
        "North": "Gauteng North",
        "South": "Western Cape",
        "East": "KwaZulu-Natal",
        "West": "Gauteng West",
    }
    df["Region"] = df["Region"].map(region_map).fillna(df["Region"])

    # Rename Store ID labels to branch names
    store_map = {
        "S001": "Branch 1 – Pretoria",
        "S002": "Branch 2 – Cape Town",
        "S003": "Branch 3 – Durban",
        "S004": "Branch 4 – Johannesburg",
        "S005": "Branch 5 – Port Elizabeth",
    }
    df["Store ID"] = df["Store ID"].map(store_map).fillna(df["Store ID"])

    # ── 15. Sort and reset index ───────────────────────────────────────────
    df = df.sort_values(["Date", "Store ID", "Product ID"]).reset_index(drop=True)

    # ── 16. Final stats ────────────────────────────────────────────────────
    stats["processed_rows"] = len(df)
    stats["total_removed"] = stats["raw_rows"] - stats["processed_rows"]
    stats["n_products"] = df["Product ID"].nunique()
    stats["n_product_names"] = df["Product Name"].nunique()
    stats["n_stores"] = df["Store ID"].nunique()
    stats["n_categories"] = df["Category"].nunique()
    stats["n_regions"] = df["Region"].nunique()
    stats["date_min"] = str(df["Date"].min().date())
    stats["date_max"] = str(df["Date"].max().date())
    stats["date_range_days"] = int((df["Date"].max() - df["Date"].min()).days)

    if verbose:
        print(f"[Cleaning] Processing complete.")
        print(f"  Raw rows          : {stats['raw_rows']:,}")
        print(f"  Bad dates removed : {stats['bad_dates_removed']:,}")
        print(f"  Duplicates removed: {stats['duplicates_removed']:,}")
        print(f"  Invalid removed   : {stats['invalid_records_removed']:,}")
        print(f"  Non-FS removed    : {stats['non_fs_rows_removed']:,}")
        print(f"  Processed rows    : {stats['processed_rows']:,}")
        print(f"  Products          : {stats['n_products']}")
        print(f"  Stores            : {stats['n_stores']}")
        print(f"  Categories        : {stats['n_categories']}")
        print(f"  Date range        : {stats['date_min']} → {stats['date_max']}")

    # ── 17. Save processed dataset ─────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    if verbose:
        print(f"[Cleaning] Saved processed dataset → {save_path}")

    return df, stats


def get_cleaning_stats(processed_path: str = PROCESSED_PATH) -> dict:
    """
    Return cleaning statistics by re-running the pipeline or reading from the
    processed file if it already exists.  Used by the Streamlit dashboard.
    """
    _, stats = clean(verbose=False)
    return stats


if __name__ == "__main__":
    df, stats = clean(verbose=True)
    print(f"\nSample of processed data:\n{df.head(5).to_string(index=False)}")
