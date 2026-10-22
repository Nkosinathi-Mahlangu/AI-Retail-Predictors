"""
SmartFuel AI — Training Pipeline Orchestrator
===============================================
Run this script once to:
  1. Clean and preprocess the raw dataset
  2. Engineer features
  3. Train and compare four demand-forecasting models
  4. Generate the 7-day forecast
  5. Save all outputs to models/ and data/processed/

Usage
-----
    python train_model.py                  # default 7-day forecast
    python train_model.py --horizon 14     # 14-day forecast
    python train_model.py --horizon 30     # 30-day forecast
    python train_model.py --safety 0.25    # 25% safety stock
"""

import sys
import os
import argparse
import time

# ── Make src/ importable regardless of working directory ──────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data_cleaning      import clean
from src.feature_engineering import build_features, get_feature_columns
from src.model_training      import train_and_evaluate
from src.forecasting         import generate_forecast


def main(horizon: int = 7, safety_pct: float = 0.20) -> None:
    t0 = time.time()
    print("=" * 60)
    print("  SmartFuel AI — Training Pipeline")
    print("=" * 60)

    # ── Step 1: Data cleaning ──────────────────────────────────────────────
    print("\n[Step 1/4] Data cleaning & filling-station product filtering …")
    df_clean, stats = clean(verbose=True)
    print(f"  ✓ {stats['processed_rows']:,} rows ready for modelling "
          f"(removed {stats['total_removed']:,} rows)")

    # ── Step 2: Feature engineering ───────────────────────────────────────
    print("\n[Step 2/4] Feature engineering …")
    df_feat   = build_features(df_clean, verbose=True)
    feat_cols = get_feature_columns()

    # ── Step 3: Model training & evaluation ───────────────────────────────
    print("\n[Step 3/4] Training and evaluating models (chronological split) …")
    bundle, metrics_df, test_preds = train_and_evaluate(
        df_feat, feat_cols, train_frac=0.80, verbose=True
    )

    print("\n── Model comparison ─────────────────────────────────────────────")
    print(metrics_df[["Model", "MAE", "RMSE", "R2", "MAPE"]].to_string(index=False))
    print(f"\n  Best model : {bundle['best_model']}")
    best_row = metrics_df[metrics_df["Model"] == bundle["best_model"]].iloc[0]
    print(f"  MAE        : {best_row['MAE']:.4f}")
    print(f"  RMSE       : {best_row['RMSE']:.4f}")
    print(f"  R²         : {best_row['R2']:.4f}")
    print(f"  MAPE       : {best_row['MAPE']:.2f}%")

    # ── Step 4: Forecast generation ───────────────────────────────────────
    print(f"\n[Step 4/4] Generating {horizon}-day demand forecast …")
    fc = generate_forecast(
        df_feat,
        horizon=horizon,
        safety_stock_pct=safety_pct,
        verbose=True,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  Pipeline complete")
    print("=" * 60)
    print(f"  Time elapsed          : {elapsed:.1f}s")
    print(f"  Processed dataset     : data/processed/filling_station_sales.csv")
    print(f"  Model bundle          : models/model_bundle.joblib")
    print(f"  Metrics               : models/metrics.csv")
    print(f"  Test predictions      : models/test_predictions.csv")
    print(f"  Forecast ({horizon:2d} days)    : models/forecast.csv")
    print(f"  Daily forecast        : models/forecast_daily.csv")
    print(f"\n  Forecast summary:")
    print(f"    HIGH risk items     : {(fc['Stockout Risk']=='HIGH').sum()}")
    print(f"    Items to reorder    : {(fc['Recommended Order']>0).sum()}")
    print(f"    Total units to order: {fc['Recommended Order'].sum():,.0f}")
    print("\n  Launch the dashboard:  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartFuel AI training pipeline")
    parser.add_argument("--horizon", type=int,   default=7,    help="Forecast horizon in days (7/14/30)")
    parser.add_argument("--safety",  type=float, default=0.20, help="Safety stock percentage (0.0–1.0)")
    args = parser.parse_args()

    if args.horizon not in [7, 14, 30]:
        print(f"Warning: horizon {args.horizon} is non-standard. Supported: 7, 14, 30")
    if not (0.0 < args.safety < 1.0):
        print("Warning: safety should be between 0.0 and 1.0. Using 0.20.")
        args.safety = 0.20

    main(horizon=args.horizon, safety_pct=args.safety)
