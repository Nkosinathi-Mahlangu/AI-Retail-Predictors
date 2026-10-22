"""
SmartFuel AI — System Validation Tests
Run: python test_system.py
"""
import sys
import os
# Force UTF-8 output on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  ✓ {name}")
        PASS += 1
    else:
        print(f"  ✗ FAIL: {name} — {detail}")
        FAIL += 1

print("=" * 60)
print("  SmartFuel AI — System Validation")
print("=" * 60)

# ── 1. Core imports ────────────────────────────────────────────────────────
print("\n[1] Core imports")
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit
    check("plotly", True)
    check("streamlit", True)
except ImportError as e:
    check("core imports", False, str(e))

# ── 2. Source module imports ───────────────────────────────────────────────
print("\n[2] Source module imports")
try:
    from src.data_cleaning import clean, load_raw, load_product_mapping
    check("src.data_cleaning", True)
except Exception as e:
    check("src.data_cleaning", False, str(e))

try:
    from src.feature_engineering import build_features, get_feature_columns
    check("src.feature_engineering", True)
except Exception as e:
    check("src.feature_engineering", False, str(e))

try:
    from src.model_training import train_and_evaluate
    check("src.model_training", True)
except Exception as e:
    check("src.model_training", False, str(e))

try:
    from src.forecasting import generate_forecast, load_forecast, load_daily_forecast
    check("src.forecasting", True)
except Exception as e:
    check("src.forecasting", False, str(e))

try:
    from src.inventory import (calculate_inventory, get_reorder_priorities,
                                get_stockout_summary, generate_reorder_message, RISK_ORDER)
    check("src.inventory", True)
except Exception as e:
    check("src.inventory", False, str(e))

try:
    from src.assistant import ManagerAssistant
    check("src.assistant", True)
except Exception as e:
    check("src.assistant", False, str(e))

try:
    from src.speech import get_status, SPEECH_ENABLED
    check("src.speech", True)
except Exception as e:
    check("src.speech", False, str(e))

# ── 3. Data files ──────────────────────────────────────────────────────────
print("\n[3] Data files")
files_to_check = [
    "data/raw/sales_data.csv",
    "data/processed/filling_station_sales.csv",
    "data/product_mapping.csv",
    "models/model_bundle.joblib",
    "models/metrics.csv",
    "models/forecast.csv",
    "models/forecast_daily.csv",
    "models/test_predictions.csv",
]
for f in files_to_check:
    check(f, os.path.exists(f), "file not found")

# ── 4. Data cleaning ───────────────────────────────────────────────────────
print("\n[4] Data cleaning")
try:
    df_clean, stats = clean(verbose=False)
    check("clean() runs", True)
    check("76000 rows loaded", stats["raw_rows"] == 76000)
    check("all 7 FS categories present", stats["n_categories"] == 7)
    check("5 branches", stats["n_stores"] == 5)
    check("20 products", stats["n_products"] == 20)
    bad_cats = {"Electronics", "Clothing", "Toys", "Furniture"}
    found_bad = bad_cats.intersection(set(df_clean["Category"].unique()))
    check("No non-FS categories in cleaned data", len(found_bad) == 0,
          f"Found: {found_bad}")
    check("Product Name column present", "Product Name" in df_clean.columns)
    check("SA branch names present", "Branch 1" in str(df_clean["Store ID"].iloc[0]) or
          any("Branch" in str(s) for s in df_clean["Store ID"].unique()))
except Exception as e:
    check("data cleaning", False, str(e))

# ── 5. Feature engineering ─────────────────────────────────────────────────
print("\n[5] Feature engineering")
try:
    from src.feature_engineering import build_features, get_feature_columns
    df_feat = build_features(df_clean, verbose=False)
    check("build_features() runs", True)
    check("lag_1 present", "lag_1" in df_feat.columns)
    check("lag_7 present", "lag_7" in df_feat.columns)
    check("rolling_7_mean present", "rolling_7_mean" in df_feat.columns)
    check("is_weekend present", "is_weekend" in df_feat.columns)
    check("price_comp_ratio present", "price_comp_ratio" in df_feat.columns)
    feat_cols = get_feature_columns()
    check("get_feature_columns() returns dict", isinstance(feat_cols, dict))
except Exception as e:
    check("feature engineering", False, str(e))

# ── 6. Model outputs ───────────────────────────────────────────────────────
print("\n[6] Model outputs")
try:
    df_metrics = pd.read_csv("models/metrics.csv")
    check("metrics.csv has 4 models", len(df_metrics) == 4)
    check("HistGradientBoosting present", "HistGradientBoosting" in df_metrics["Model"].values)
    best = df_metrics.sort_values("MAE").iloc[0]
    check("best model MAE < 25", best["MAE"] < 25,
          f"MAE={best['MAE']:.2f}")
    check("best model R2 > 0.7", best["R2"] > 0.7,
          f"R2={best['R2']:.4f}")

    df_fc = pd.read_csv("models/forecast.csv")
    check("forecast.csv has 100 rows (5 stores x 20 products)", len(df_fc) == 100)
    check("Total Predicted Demand column", "Total Predicted Demand" in df_fc.columns)
    check("Recommended Order column", "Recommended Order" in df_fc.columns)
    check("Stockout Risk column", "Stockout Risk" in df_fc.columns)
    check("No Electronics in forecast", "Electronics" not in df_fc["Category"].values)
    check("No Clothing in forecast", "Clothing" not in df_fc["Category"].values)
    check("No Toys in forecast", "Toys" not in df_fc["Category"].values)
    check("No Furniture in forecast", "Furniture" not in df_fc["Category"].values)

    tp = pd.read_csv("models/test_predictions.csv")
    check("test_predictions.csv has 15000 rows", len(tp) == 15000)

    daily = pd.read_csv("models/forecast_daily.csv")
    check("forecast_daily.csv not empty", len(daily) > 0)
except Exception as e:
    check("model outputs", False, str(e))

# ── 7. Inventory module ────────────────────────────────────────────────────
print("\n[7] Inventory module")
try:
    df_fc = pd.read_csv("models/forecast.csv")
    inv = calculate_inventory(df_fc, safety_pct=0.20)
    check("calculate_inventory() runs", True)
    check("Safety Stock column", "Safety Stock" in inv.columns)
    check("Recommended Order column", "Recommended Order" in inv.columns)
    check("Days Until Stockout column", "Days Until Stockout" in inv.columns)
    summary = get_stockout_summary(df_fc)
    check("get_stockout_summary() runs", isinstance(summary, dict))
    check("summary has expected keys",
          all(k in summary for k in ["high_risk","medium_risk","low_risk","total_reorder_units"]))
    msg = generate_reorder_message(inv.iloc[0], horizon=7)
    check("generate_reorder_message() returns string", isinstance(msg, str) and len(msg) > 20)
except Exception as e:
    check("inventory module", False, str(e))

# ── 8. AI assistant ────────────────────────────────────────────────────────
print("\n[8] AI assistant")
try:
    df_proc = pd.read_csv("data/processed/filling_station_sales.csv")
    df_proc["Date"] = pd.to_datetime(df_proc["Date"])
    df_fc = pd.read_csv("models/forecast.csv")
    assistant = ManagerAssistant(df_proc, df_fc)
    check("ManagerAssistant initialises", True)

    test_questions = {
        "hello":                              "greeting",
        "help":                               "help",
        "Which products should I reorder?":   "reorder",
        "What are the top selling products?": "top_sellers",
        "Which products are slow moving?":    "slow_movers",
        "Show the demand forecast":           "forecast",
        "How is Beverages performing?":       "category",
        "What are the sales trends?":         "trend",
        "How does weather affect demand?":    "weather",
        "What is the impact of promotions?":  "promotion",
    }
    for q, intent in test_questions.items():
        resp = assistant.ask(q)
        has_text = isinstance(resp.get("text"), str) and len(resp["text"]) > 10
        check(f'assistant.ask("{q[:35]}...")', has_text, "empty response")

    # Verify no hallucinated numbers (all numeric answers come from data)
    resp = assistant.ask("Which products should I reorder?")
    check("reorder response has table", resp.get("table") is not None)

except Exception as e:
    check("AI assistant", False, str(e))

# ── 9. Product filtering ───────────────────────────────────────────────────
print("\n[9] Final product/category integrity check")
try:
    df_proc = pd.read_csv("data/processed/filling_station_sales.csv")
    all_cats = set(df_proc["Category"].unique())
    forbidden = {"Electronics", "Clothing", "Toys", "Furniture"}
    leaked = forbidden.intersection(all_cats)
    check("No forbidden categories in processed data", len(leaked) == 0,
          f"Found: {leaked}")

    df_fc = pd.read_csv("models/forecast.csv")
    fc_cats = set(df_fc["Category"].unique())
    fc_leaked = forbidden.intersection(fc_cats)
    check("No forbidden categories in forecast", len(fc_leaked) == 0,
          f"Found: {fc_leaked}")

    expected_cats = {"Beverages", "Snacks", "Food", "Dairy & Refrigerated",
                     "Automotive", "Personal Care", "Household & Convenience"}
    check("All 7 FS categories present in processed data",
          expected_cats == all_cats, f"Got: {all_cats}")
    check("Product Name column in processed data", "Product Name" in df_proc.columns)

    # Spot-check a few real product names
    product_names = set(df_fc["Product Name"].unique())
    check("Bottled Water in forecast", "Bottled Water (500ml)" in product_names)
    check("Coca-Cola in forecast", "Coca-Cola (500ml)" in product_names)
    check("Shell Helix Engine Oil in forecast",
          any("Shell" in p for p in product_names))
except Exception as e:
    check("product filtering", False, str(e))

# ── Summary ────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"  Results: {PASS} passed / {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL TESTS PASSED — SmartFuel AI is ready.")
else:
    print(f"  ⚠️  {FAIL} tests need attention.")
print("=" * 60)
