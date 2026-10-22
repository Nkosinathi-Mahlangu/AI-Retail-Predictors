"""
SmartFuel AI — Forecasting Engine
====================================
Generates multi-horizon demand forecasts for every store/product combination
using the trained model bundle.

Supported horizons : 7, 14, 30 days (configurable)

The forecast is an iterative one-step-ahead prediction: the model predicts
day N+1 using the last known actual values as lag/rolling features.  Because
we are simulating future periods, later-period lag features are progressively
filled with the prior prediction.  This is clearly documented as an
approximation appropriate for a short-horizon (≤30 day) academic prototype.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR     = os.path.join(ROOT, "models")
PROCESSED_PATH = os.path.join(ROOT, "data", "processed", "filling_station_sales.csv")


def load_bundle(model_dir: str = MODEL_DIR) -> dict:
    path = os.path.join(model_dir, "model_bundle.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Model bundle not found. Run python train_model.py first."
        )
    return joblib.load(path)


def _get_latest_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent row for each store/product combination."""
    return (
        df.sort_values("Date")
        .groupby(["Store ID", "Product ID"], as_index=False)
        .last()
        .copy()
    )


def _build_forecast_row(
    base_row: pd.Series,
    forecast_date: pd.Timestamp,
    recent_demands: list,
    all_features: list,
) -> dict:
    """
    Construct a feature row for a single future date given recent demand history.

    Parameters
    ----------
    base_row      : last known row for this store/product
    forecast_date : the date we are predicting
    recent_demands: list of recent demands (most recent last), length ≥ 28 ideal
    all_features  : feature column names expected by the pipeline
    """
    row = base_row.to_dict()
    row["Date"] = forecast_date

    # Date features
    row["day"]            = forecast_date.day
    row["day_of_week"]    = forecast_date.dayofweek
    row["week"]           = int(forecast_date.isocalendar()[1])
    row["month"]          = forecast_date.month
    row["quarter"]        = (forecast_date.month - 1) // 3 + 1
    row["year"]           = forecast_date.year
    row["is_weekend"]     = int(forecast_date.dayofweek >= 5)
    row["is_month_start"] = int(forecast_date.day == 1)
    row["is_month_end"]   = int(
        forecast_date == forecast_date + pd.offsets.MonthEnd(0)
    )

    # Lag features (most recent demand = most recent element)
    n = len(recent_demands)
    row["lag_1"]  = recent_demands[-1]  if n >= 1  else np.nan
    row["lag_7"]  = recent_demands[-7]  if n >= 7  else (recent_demands[0] if n > 0 else np.nan)
    row["lag_14"] = recent_demands[-14] if n >= 14 else (recent_demands[0] if n > 0 else np.nan)
    row["lag_28"] = recent_demands[-28] if n >= 28 else (recent_demands[0] if n > 0 else np.nan)

    # Rolling means
    for w in [7, 14, 28]:
        vals = recent_demands[-w:] if len(recent_demands) >= w else recent_demands
        row[f"rolling_{w}_mean"] = float(np.mean(vals)) if vals else np.nan

    # Rolling stds
    for w in [7, 28]:
        vals = recent_demands[-w:] if len(recent_demands) >= w else recent_demands
        row[f"rolling_{w}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0

    # Derived business features
    price = row.get("Price", 1.0) or 1.0
    comp  = row.get("Competitor Pricing", price) or price
    row["price_comp_ratio"] = price / comp if comp > 0 else 1.0
    row["discount_pct"]     = (row.get("Discount", 0) or 0) / 100.0

    # Fill any remaining missing features with 0
    for f in all_features:
        if f not in row or (isinstance(row[f], float) and np.isnan(row[f])):
            row[f] = 0

    return row


def generate_forecast(
    df_processed: pd.DataFrame,
    horizon: int = 7,
    safety_stock_pct: float = 0.20,
    model_dir: str = MODEL_DIR,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate a rolling multi-day forecast for every store/product.

    Parameters
    ----------
    df_processed     : cleaned + feature-engineered DataFrame
    horizon          : number of days to forecast (7, 14, or 30)
    safety_stock_pct : fraction of forecast demand kept as safety buffer
    model_dir        : directory containing model_bundle.joblib
    verbose          : print progress

    Returns
    -------
    forecast_df : pd.DataFrame
        One row per (store, product, forecast_date) with demand prediction,
        inventory recommendation and stockout risk label.
    """
    bundle = load_bundle(model_dir)
    pipe         = bundle["pipeline"]
    all_features = bundle["features"]

    if verbose:
        print(f"[Forecast] Generating {horizon}-day forecast …")

    # Get the most recent actual values as a base
    latest = _get_latest_rows(df_processed)

    # Build a history buffer per store/product for lag computation
    history: dict = {}
    for _, grp in df_processed.groupby(["Store ID", "Product ID"]):
        key = (grp["Store ID"].iloc[0], grp["Product ID"].iloc[0])
        history[key] = list(grp.sort_values("Date")["Demand"].values)

    records = []
    last_date = df_processed["Date"].max()

    for _, base_row in latest.iterrows():
        key = (base_row["Store ID"], base_row["Product ID"])
        demands = list(history.get(key, [float(base_row.get("Demand", 50))]))

        for h in range(1, horizon + 1):
            forecast_date = last_date + pd.Timedelta(days=h)
            row_dict = _build_forecast_row(base_row, forecast_date, demands, all_features)

            # Predict
            X = pd.DataFrame([row_dict])[all_features]
            pred = float(np.maximum(pipe.predict(X)[0], 0))

            # Append prediction to running demand list for next iteration
            demands.append(pred)

            records.append({
                "Forecast Date":    forecast_date,
                "Store ID":         base_row["Store ID"],
                "Product ID":       base_row["Product ID"],
                "Product Name":     base_row.get("Product Name", base_row["Product ID"]),
                "Category":         base_row.get("Category", "Unknown"),
                "Subcategory":      base_row.get("Subcategory", ""),
                "Region":           base_row.get("Region", ""),
                "Predicted Demand": round(pred, 1),
                "Inventory Level":  float(base_row.get("Inventory Level", 0)),
                "Price":            float(base_row.get("Price", 0)),
                "Discount":         float(base_row.get("Discount", 0)),
                "Promotion":        int(base_row.get("Promotion", 0)),
            })

    forecast_df = pd.DataFrame(records)

    # ── Aggregate to total predicted demand per store/product over horizon ─
    agg = (
        forecast_df
        .groupby(
            ["Store ID", "Product ID", "Product Name", "Category",
             "Subcategory", "Region", "Inventory Level", "Price",
             "Discount", "Promotion"],
            as_index=False,
        )["Predicted Demand"]
        .sum()
        .rename(columns={"Predicted Demand": "Total Predicted Demand"})
    )

    agg["Forecast Horizon Days"] = horizon
    agg["Safety Stock"]          = np.ceil(agg["Total Predicted Demand"] * safety_stock_pct)
    agg["Reorder Point"]         = np.ceil(agg["Total Predicted Demand"] + agg["Safety Stock"])
    agg["Recommended Order"]     = np.maximum(
        np.ceil(agg["Reorder Point"] - agg["Inventory Level"]), 0
    )
    agg["Stockout Risk"] = np.where(
        agg["Inventory Level"] < agg["Total Predicted Demand"],
        "HIGH",
        np.where(
            agg["Inventory Level"] < agg["Reorder Point"],
            "MEDIUM",
            "LOW",
        ),
    )

    # Also keep the daily-level forecast for trend charts
    daily_path = os.path.join(model_dir, "forecast_daily.csv")
    forecast_df.to_csv(daily_path, index=False)

    # Save aggregate forecast
    agg_path = os.path.join(model_dir, "forecast.csv")
    agg.to_csv(agg_path, index=False)

    if verbose:
        high  = (agg["Stockout Risk"] == "HIGH").sum()
        total = agg["Recommended Order"].sum()
        print(f"[Forecast] Complete. High-risk items: {high}  |  Total reorder: {total:,.0f} units")

    return agg


def load_forecast(model_dir: str = MODEL_DIR) -> pd.DataFrame:
    """Load the saved aggregate forecast CSV."""
    path = os.path.join(model_dir, "forecast.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def load_daily_forecast(model_dir: str = MODEL_DIR) -> pd.DataFrame:
    """Load the saved daily forecast CSV."""
    path = os.path.join(model_dir, "forecast_daily.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "Forecast Date" in df.columns:
        df["Forecast Date"] = pd.to_datetime(df["Forecast Date"])
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ROOT)
    from src.data_cleaning import clean
    from src.feature_engineering import build_features

    df_clean, _ = clean(verbose=False)
    df_feat      = build_features(df_clean, verbose=False)

    for h in [7, 14, 30]:
        fc = generate_forecast(df_feat, horizon=h, verbose=True)
        print(fc[["Store ID", "Product Name", "Category",
                   "Total Predicted Demand", "Recommended Order",
                   "Stockout Risk"]].head(5).to_string(index=False))
        print()
