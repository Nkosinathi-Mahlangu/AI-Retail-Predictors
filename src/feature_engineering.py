"""
SmartFuel AI — Feature Engineering
=====================================
Builds a rich feature set from the cleaned filling-station sales dataset.

Features created
----------------
Date features   : day, day_of_week, week, month, quarter, year, is_weekend,
                  is_month_start, is_month_end
Lag features    : lag_1, lag_7, lag_14, lag_28
Rolling stats   : rolling_7_mean, rolling_14_mean, rolling_28_mean,
                  rolling_7_std, rolling_28_std
Business feats  : price, discount, promotion, inventory_level, units_ordered,
                  competitor_pricing
Context feats   : weather (encoded), seasonality (encoded), region (encoded)

All lag/rolling features are shifted by at least 1 to avoid data leakage.
"""

import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(ROOT, "data", "processed", "filling_station_sales.csv")


def build_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Take the cleaned DataFrame and return it with engineered features added.

    Parameters
    ----------
    df : pd.DataFrame
        Output of data_cleaning.clean()
    verbose : bool
        Print progress messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with all feature columns appended, rows with insufficient
        lag history dropped.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)

    if verbose:
        print(f"[Features] Building features for {len(df):,} rows …")

    # ── 1. Date features ───────────────────────────────────────────────────
    df["day"]           = df["Date"].dt.day
    df["day_of_week"]   = df["Date"].dt.dayofweek          # 0=Monday
    df["week"]          = df["Date"].dt.isocalendar().week.astype(int)
    df["month"]         = df["Date"].dt.month
    df["quarter"]       = df["Date"].dt.quarter
    df["year"]          = df["Date"].dt.year
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
    df["is_month_end"]   = df["Date"].dt.is_month_end.astype(int)

    # ── 2. Lag features (grouped by store + product to avoid cross-series)──
    grp = ["Store ID", "Product ID"]

    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df.groupby(grp)["Demand"].shift(lag)

    # ── 3. Rolling window statistics (shifted 1 to avoid leakage) ─────────
    def _rolling(series: pd.Series, window: int, stat: str = "mean",
                 min_periods: int = 2):
        shifted = series.shift(1)
        r = shifted.rolling(window=window, min_periods=min_periods)
        return r.mean() if stat == "mean" else r.std()

    for window in [7, 14, 28]:
        df[f"rolling_{window}_mean"] = df.groupby(grp)["Demand"].transform(
            lambda s, w=window: _rolling(s, w, "mean")
        )

    for window in [7, 28]:
        df[f"rolling_{window}_std"] = df.groupby(grp)["Demand"].transform(
            lambda s, w=window: _rolling(s, w, "std")
        )

    # ── 4. Numeric business features ───────────────────────────────────────
    for col in ["Price", "Discount", "Competitor Pricing",
                "Inventory Level", "Units Ordered", "Promotion"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Price-to-competitor ratio (useful signal)
    if "Price" in df.columns and "Competitor Pricing" in df.columns:
        df["price_comp_ratio"] = np.where(
            df["Competitor Pricing"] > 0,
            df["Price"] / df["Competitor Pricing"],
            1.0,
        )

    # Discount as a fraction
    df["discount_pct"] = df["Discount"] / 100.0

    # ── 5. Drop rows with insufficient lag history ─────────────────────────
    # We require at minimum lag_1 and lag_7 to be non-null for model training.
    before = len(df)
    df = df.dropna(subset=["lag_1", "lag_7", "rolling_7_mean"]).reset_index(drop=True)
    dropped = before - len(df)

    if verbose:
        print(f"[Features] Dropped {dropped:,} rows with insufficient lag history.")
        print(f"[Features] Feature engineering complete. Rows: {len(df):,}")

    return df


def get_feature_columns() -> dict:
    """
    Return the canonical feature-column lists used by the model.

    Returns
    -------
    dict with keys 'categorical', 'numeric', 'all_features'
    """
    categorical = [
        "Store ID", "Product ID", "Category", "Region",
        "Weather Condition", "Seasonality",
    ]
    numeric = [
        "Price", "Discount", "Competitor Pricing", "Inventory Level",
        "Units Ordered", "Promotion", "discount_pct", "price_comp_ratio",
        "day", "day_of_week", "week", "month", "quarter", "year",
        "is_weekend", "is_month_start", "is_month_end",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "rolling_7_mean", "rolling_14_mean", "rolling_28_mean",
        "rolling_7_std", "rolling_28_std",
    ]
    return {
        "categorical": categorical,
        "numeric": numeric,
        "all_features": categorical + numeric,
    }


if __name__ == "__main__":
    from src.data_cleaning import clean
    df_clean, _ = clean(verbose=False)
    df_feat = build_features(df_clean, verbose=True)
    cols = get_feature_columns()
    print(f"\nCategorical features : {cols['categorical']}")
    print(f"Numeric features     : {cols['numeric']}")
    print(f"\nSample:\n{df_feat[['Date', 'Product ID', 'Demand', 'lag_1', 'lag_7', 'rolling_7_mean']].head(10).to_string(index=False)}")
