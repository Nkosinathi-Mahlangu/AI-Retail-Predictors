"""
SmartFuel AI — Standalone Forecast Utility
============================================
Use this script to regenerate forecasts from an already-trained model without
rerunning the full training pipeline.

Usage
-----
    python forecast.py                   # 7-day forecast (default)
    python forecast.py --horizon 14      # 14-day forecast
    python forecast.py --horizon 30      # 30-day forecast
    python forecast.py --product P0001   # forecast for one product
    python forecast.py --store "Branch 1 – Pretoria"

The model bundle (models/model_bundle.joblib) must exist. Run train_model.py
first if it does not.
"""

import sys
import os
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data_cleaning       import clean
from src.feature_engineering  import build_features
from src.forecasting          import generate_forecast, load_forecast
from src.inventory            import calculate_inventory, generate_reorder_message


def main(horizon: int = 7, product_id: str = None, store_id: str = None,
         safety_pct: float = 0.20) -> None:

    print(f"SmartFuel AI — Generating {horizon}-day forecast …")

    # Load and prepare data
    df_clean, _ = clean(verbose=False)
    df_feat      = build_features(df_clean, verbose=False)

    # Generate forecast
    fc = generate_forecast(df_feat, horizon=horizon, safety_stock_pct=safety_pct, verbose=True)
    fc = calculate_inventory(fc, safety_pct=safety_pct)

    # Apply filters if requested
    if product_id:
        fc = fc[fc["Product ID"].str.upper() == product_id.upper()]
    if store_id:
        fc = fc[fc["Store ID"].str.lower().str.contains(store_id.lower())]

    if fc.empty:
        print("No records match the specified filters.")
        return

    # Display results
    print(f"\n{'='*70}")
    print(f"  {horizon}-Day Demand Forecast")
    print(f"{'='*70}")

    display_cols = [c for c in [
        "Store ID", "Product Name", "Category",
        "Inventory Level", "Total Predicted Demand",
        "Recommended Order", "Stockout Risk", "Days Until Stockout",
    ] if c in fc.columns]

    print(fc.sort_values(
        "Recommended Order", ascending=False
    )[display_cols].head(30).to_string(index=False))

    # Natural-language summary for top items
    print(f"\n{'='*70}")
    print("  Priority reorder messages:")
    print(f"{'='*70}")
    high_risk = fc[fc["Stockout Risk"] == "HIGH"].sort_values(
        "Recommended Order", ascending=False
    )
    if high_risk.empty:
        print("  ✅ No HIGH-risk items in current forecast.")
    else:
        for _, row in high_risk.head(5).iterrows():
            print(f"  {generate_reorder_message(row, horizon)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartFuel AI forecast utility")
    parser.add_argument("--horizon",  type=int,   default=7,    help="Forecast days (7/14/30)")
    parser.add_argument("--product",  type=str,   default=None, help="Product ID filter (e.g. P0001)")
    parser.add_argument("--store",    type=str,   default=None, help="Store name/ID filter")
    parser.add_argument("--safety",   type=float, default=0.20, help="Safety stock fraction")
    args = parser.parse_args()

    main(
        horizon=args.horizon,
        product_id=args.product,
        store_id=args.store,
        safety_pct=args.safety,
    )
