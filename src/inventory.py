"""
SmartFuel AI — Inventory Intelligence Module
==============================================
Calculates safety stock, reorder points, recommended order quantities and
stockout risk levels for all filling-station products.

Assumptions (clearly documented as prototype assumptions)
---------------------------------------------------------
Safety Stock     = Total Predicted Demand × safety_pct  (default 20%)
Reorder Point    = Total Predicted Demand + Safety Stock
Recommended Order= max(Reorder Point − Current Inventory, 0)
Stockout Risk    = HIGH   if Inventory < Total Predicted Demand
                 = MEDIUM if Inventory < Reorder Point
                 = LOW    otherwise

The safety-stock percentage is configurable by the store manager in the GUI.
In a real system this would be derived from demand variance and service-level
targets (Z-score method).  This simpler heuristic is appropriate for an
academic prototype.
"""

import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")


# ── Risk colour mapping ───────────────────────────────────────────────────
RISK_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
RISK_COLOURS = {
    "HIGH":   "#E74C3C",   # red
    "MEDIUM": "#F39C12",   # amber
    "LOW":    "#27AE60",   # green
}


def calculate_inventory(
    forecast_df: pd.DataFrame,
    safety_pct: float = 0.20,
) -> pd.DataFrame:
    """
    Enrich a forecast DataFrame with inventory intelligence columns.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Output of forecasting.generate_forecast() — must contain at least
        'Total Predicted Demand' and 'Inventory Level'.
    safety_pct : float
        Safety stock as a fraction of forecast demand (e.g. 0.20 = 20%).

    Returns
    -------
    pd.DataFrame with added columns:
        Safety Stock, Reorder Point, Recommended Order, Stockout Risk,
        Days Until Stockout (approx), Risk Colour
    """
    df = forecast_df.copy()
    horizon = int(df.get("Forecast Horizon Days", pd.Series([7])).iloc[0]) if "Forecast Horizon Days" in df.columns else 7

    demand_col = "Total Predicted Demand"
    if demand_col not in df.columns and "Predicted Demand" in df.columns:
        demand_col = "Predicted Demand"

    daily_demand = df[demand_col] / max(horizon, 1)

    df["Safety Stock"]      = np.ceil(df[demand_col] * safety_pct).astype(int)
    df["Reorder Point"]     = np.ceil(df[demand_col] + df["Safety Stock"]).astype(int)
    df["Recommended Order"] = np.maximum(
        np.ceil(df["Reorder Point"] - df["Inventory Level"]), 0
    ).astype(int)

    df["Stockout Risk"] = np.where(
        df["Inventory Level"] < df[demand_col],
        "HIGH",
        np.where(
            df["Inventory Level"] < df["Reorder Point"],
            "MEDIUM",
            "LOW",
        ),
    )

    # Approximate days until stockout (if daily demand > 0)
    df["Days Until Stockout"] = np.where(
        daily_demand > 0,
        np.floor(df["Inventory Level"] / daily_demand).astype(int),
        999,
    )
    df["Days Until Stockout"] = df["Days Until Stockout"].clip(upper=999)

    df["Risk Colour"] = df["Stockout Risk"].map(RISK_COLOURS)
    df["_risk_order"] = df["Stockout Risk"].map(RISK_ORDER)

    return df


def get_reorder_priorities(
    forecast_df: pd.DataFrame,
    safety_pct: float = 0.20,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Return the top-N reorder priorities sorted by risk then recommended order.
    """
    df = calculate_inventory(forecast_df, safety_pct)
    df = df.sort_values(
        ["_risk_order", "Recommended Order"],
        ascending=[True, False],
    ).drop(columns=["_risk_order", "Risk Colour"], errors="ignore")
    return df.head(top_n).reset_index(drop=True)


def get_stockout_summary(forecast_df: pd.DataFrame, safety_pct: float = 0.20) -> dict:
    """
    Return a summary dict of inventory statistics for the KPI dashboard.
    """
    df = calculate_inventory(forecast_df, safety_pct)
    return {
        "total_products":       len(df),
        "high_risk":            int((df["Stockout Risk"] == "HIGH").sum()),
        "medium_risk":          int((df["Stockout Risk"] == "MEDIUM").sum()),
        "low_risk":             int((df["Stockout Risk"] == "LOW").sum()),
        "products_to_reorder":  int((df["Recommended Order"] > 0).sum()),
        "total_reorder_units":  int(df["Recommended Order"].sum()),
        "avg_days_to_stockout": float(df["Days Until Stockout"].replace(999, np.nan).mean()),
    }


def generate_reorder_message(row: pd.Series, horizon: int = 7) -> str:
    """
    Generate a natural-language reorder recommendation for a single product row.
    """
    name    = row.get("Product Name", row.get("Product ID", "This product"))
    pred    = row.get("Total Predicted Demand", row.get("Predicted Demand", 0))
    inv     = row.get("Inventory Level", 0)
    order   = row.get("Recommended Order", 0)
    risk    = row.get("Stockout Risk", "UNKNOWN")
    ss      = row.get("Safety Stock", 0)

    if risk == "HIGH":
        urgency = "⚠️ URGENT"
    elif risk == "MEDIUM":
        urgency = "🟡 RECOMMENDED"
    else:
        urgency = "✅ SUFFICIENT"

    return (
        f"{urgency} — {name}: "
        f"Predicted demand over {horizon} days is {pred:.0f} units. "
        f"Current stock is {inv:.0f} units. "
        f"Safety stock buffer is {ss:.0f} units. "
        f"Recommended replenishment: {order:.0f} units."
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ROOT)
    from src.forecasting import load_forecast

    fc = load_forecast()
    if fc.empty:
        print("No forecast found. Run train_model.py first.")
    else:
        inv = calculate_inventory(fc)
        summary = get_stockout_summary(fc)
        print("Inventory Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print("\nTop 10 reorder priorities:")
        top = get_reorder_priorities(fc, top_n=10)
        print(top[["Product Name", "Category", "Inventory Level",
                    "Total Predicted Demand", "Recommended Order",
                    "Stockout Risk"]].to_string(index=False))
