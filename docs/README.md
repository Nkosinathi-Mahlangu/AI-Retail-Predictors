# SmartFuel AI
## Intelligent Filling Station Sales, Demand Forecasting and Inventory Management System

**Academic prototype · South African filling-station convenience store**

---

## Project Overview

SmartFuel AI is an end-to-end AI/ML system that helps a filling-station convenience-store
manager understand historical sales, predict future product demand, identify stockout risks,
receive automated reorder recommendations, and interact with store data through a
natural-language AI assistant.

### Problem Statement
Filling-station convenience stores face unpredictable product demand, frequent stockouts,
and manual reorder processes. SmartFuel AI automates demand forecasting and inventory
intelligence using machine learning, reducing the risk of lost sales from stockouts and
over-ordering of slow-moving products.

### Target Users
- Filling-station convenience-store managers
- Branch supervisors and stock controllers
- Business intelligence analysts
- Academic demonstration of AI/ML development lifecycle

---

## Dataset

| Property | Value |
|----------|-------|
| Source file | `data/raw/sales_data.csv` |
| Original rows | 76,000 |
| Processed rows | 76,000 |
| Date range | 2022-01-01 to 2024-01-30 (759 days) |
| Stores | 5 branches |
| Products | 20 products |
| Original categories | Clothing, Electronics, Furniture, Groceries, Toys |

### ⚠️ Important Academic Note — Product Mapping

The original dataset uses **generic product IDs (P0001–P0020)** with categories that include
Electronics, Clothing, Toys and Furniture — none of which are sold inside a filling-station
convenience store. The same product ID appears under different category labels across stores
(this is a property of the synthetic dataset).

A transparent **product mapping layer** (`data/product_mapping.csv`) was created to:
1. Assign each product ID a realistic filling-station product name (e.g. "Bottled Water",
   "Coca-Cola", "Shell Helix Engine Oil")
2. Assign each product ID a correct filling-station category
3. Override all original category labels with the correct filling-station categories

**The original raw CSV is preserved unchanged.** The mapping is documented as a prototype
assumption and is clearly labelled in the Data Quality page of the dashboard.

### Filling-Station Categories (7)

| Category | Example Products |
|----------|-----------------|
| Beverages | Bottled Water, Coca-Cola, Red Bull, Fanta Orange, Lucozade Sport |
| Snacks | Lay's Chips, Pringles, Simba Chips, Cadbury Dairy Milk, Jungle Oats Bar |
| Food | Albany Bread, Indomie Noodles |
| Dairy & Refrigerated | Milk (Long Life), Yogurt |
| Automotive | Shell Helix Engine Oil, Texaco Coolant |
| Personal Care | Handy Andy Wipes, Dettol Hand Sanitiser |
| Household & Convenience | Energizer Batteries, Stimorol Chewing Gum |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (run once; takes ~70 seconds)
python train_model.py

# 3. Launch the dashboard
streamlit run app.py
```

**Or use the batch file (Windows):**
```
run_project.bat
```

The dashboard opens at: **http://localhost:8501**

---

## Data Cleaning Pipeline

1. **Load raw CSV** from `data/raw/sales_data.csv`
2. **Validate columns** — ensure all required fields are present
3. **Parse dates** — convert Date column; remove unparseable rows
4. **Numeric coercion** — convert Price, Demand, Inventory to numeric; fill defaults
5. **Duplicate removal** — drop exact (Date, Store, Product) duplicates
6. **Invalid value removal** — remove rows with Demand < 0, Inventory < 0, Price ≤ 0
7. **Product mapping** — apply `data/product_mapping.csv` to assign FS product names
   and categories; all non-FS original category labels are overridden
8. **SA context mapping** — rename regions to SA provinces; rename store IDs to branch names;
   standardise weather and seasonality labels
9. **Save** to `data/processed/filling_station_sales.csv`

---

## Feature Engineering

| Feature Type | Features |
|-------------|----------|
| Date | day, day_of_week, week, month, quarter, year, is_weekend, is_month_start, is_month_end |
| Lag | lag_1, lag_7, lag_14, lag_28 (demand of 1/7/14/28 days ago) |
| Rolling mean | rolling_7_mean, rolling_14_mean, rolling_28_mean |
| Rolling std | rolling_7_std, rolling_28_std |
| Business | Price, Discount, Competitor Pricing, Inventory Level, Units Ordered, Promotion |
| Derived | price_comp_ratio, discount_pct |
| Categorical | Store ID, Product ID, Category, Region, Weather Condition, Seasonality |

All lag/rolling features are shifted by ≥1 day to prevent data leakage.

---

## Machine Learning Models

### Models Compared

| Model | Type |
|-------|------|
| Naive Lag-1 Baseline | Predicts tomorrow = today (benchmark) |
| Ridge Regression | Linear model with L2 regularisation |
| Random Forest | Ensemble of 150 decision trees |
| HistGradientBoosting | Gradient-boosted trees (best performer) |

### Evaluation Method

**Chronological 80/20 train/test split** — the model is trained on dates up to 2 Sep 2023
and evaluated on the final ~15,000 observations (Sep 2023 – Jan 2024). This simulates
real-world forecasting where the model must predict demand it has never seen.

### Actual Results (from this dataset and holdout period)

| Model | MAE | RMSE | R² | MAPE |
|-------|-----|------|-----|------|
| Naive Lag-1 Baseline | 40.12 | 51.88 | −0.38 | 53.11% |
| Ridge Regression | 23.83 | 30.47 | 0.524 | 35.54% |
| Random Forest | 20.26 | 26.86 | 0.630 | 31.02% |
| **HistGradientBoosting** | **16.73** | **22.00** | **0.752** | **25.60%** |

**Best model: HistGradientBoosting** (selected by lowest MAE)

The MAPE of 25.60% is above the aspirational 15–20% target. This is reported honestly.
Potential improvements: holiday/event features, longer product histories, dedicated
time-series models (SARIMA, Prophet), hyperparameter tuning.

---

## Demand Forecasting

The trained HistGradientBoosting model generates iterative one-step-ahead predictions
for each store/product combination. Future-period lag features are progressively filled
with prior predictions — a short-horizon approximation documented as a prototype assumption.

**Supported horizons:** 7, 14, 30 days (configurable in the sidebar).

---

## Inventory Intelligence

| Metric | Formula | Assumption |
|--------|---------|-----------|
| Safety Stock | Forecast Demand × safety% | Default 20% (configurable) |
| Reorder Point | Forecast Demand + Safety Stock | |
| Recommended Order | max(Reorder Point − Inventory, 0) | |
| Days Until Stockout | Inventory ÷ Daily Demand | |

**Risk levels:** HIGH (inventory < forecast demand), MEDIUM (inventory < reorder point), LOW

---

## AI Manager Assistant

A controlled NLP query layer that answers questions using **only actual data** from the
cleaned dataset, forecasts and inventory calculations. No external LLM API is required —
the system works fully offline.

**Supported intents:** reorder, stockout risk, top sellers, slow movers, demand forecast,
inventory levels, category performance, revenue, comparison, trends, weather impact,
promotion impact, greeting, help.

Example questions:
- *"Which products should I reorder?"*
- *"What are the top selling products?"*
- *"How is the Beverages category performing?"*
- *"Compare Snacks and Beverages"*
- *"What is the expected demand for Bottled Water?"*
- *"What are my sales trends?"*

---

## Speech Recognition (Optional)

The AI Assistant includes an optional voice input button. It requires:
```bash
pip install SpeechRecognition pyaudio
```
If unavailable, the application works normally with text input. The feature is disabled
gracefully at runtime with a clear status message.

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Dashboard | KPI cards, sales trend, category donut, top 10 products, reorder priorities, quick questions |
| 📊 Sales Analytics | Sales over time, by category, weekend vs weekday, seasonal, promotion impact, weather impact, branch comparison |
| 🔮 Demand Forecast | Predicted demand charts, risk distribution, daily forecast trend, full forecast table |
| 📦 Inventory | Inventory vs forecast by category, risk by branch, full reorder priority table, urgent messages |
| 🏆 Products | Best sellers, slow movers, revenue leaders, category view (tabs) |
| 🤖 AI Assistant | Chat interface with history, quick-question buttons, optional voice input |
| 🔍 Data Quality | Pipeline diagram, cleaning statistics, product mapping table, sample processed data |
| 📈 Model Performance | Model comparison table/charts, actual vs predicted, error distribution, metric explanations |
| ℹ️ About | Project purpose, methodology, structure, limitations, how to run |

---

## Project Structure

```
AI_Sales_Demand_Prediction_Project/
├── app.py                          # Streamlit dashboard (9 pages)
├── train_model.py                  # Training pipeline orchestrator
├── forecast.py                     # Standalone forecast utility
├── requirements.txt
├── run_project.bat                 # One-click Windows launcher
├── test_system.py                  # 72-test validation suite
├── README.md
├── MODEL_RESULTS.md
│
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
│
├── data/
│   ├── raw/
│   │   └── sales_data.csv          # Original dataset (preserved)
│   ├── processed/
│   │   └── filling_station_sales.csv
│   └── product_mapping.csv         # FS product name + category mapping
│
├── models/
│   ├── model_bundle.joblib         # Trained best model
│   ├── metrics.csv                 # All model metrics
│   ├── forecast.csv                # Aggregate 7-day forecast
│   ├── forecast_daily.csv          # Daily forecast per product/store
│   └── test_predictions.csv        # Test-set actual vs predicted
│
└── src/
    ├── __init__.py
    ├── data_cleaning.py            # Full cleaning + FS product filtering
    ├── feature_engineering.py      # Lag/rolling/date features
    ├── model_training.py           # Multi-model training + evaluation
    ├── forecasting.py              # Multi-horizon forecast generation
    ├── inventory.py                # Safety stock, reorder intelligence
    ├── assistant.py                # AI manager assistant (NLP)
    └── speech.py                   # Optional speech recognition
```

---

## Ethical Considerations

- No personal customer data is collected or processed.
- All statistics and metrics are derived from actual data — no results are fabricated.
- Predictions are labelled as estimates throughout the application.
- The product mapping prototype assumption is clearly and explicitly documented.
- The MAPE result is reported honestly even though it exceeds the aspirational target.

---

## Limitations

1. Dataset uses generic product IDs — product names are assigned via prototype mapping.
2. Safety stock uses a heuristic percentage, not a statistically derived service-level factor.
3. The forecasting model does not include holiday/event data.
4. Forecasts beyond 30 days become increasingly unreliable due to lag-propagation drift.
5. Revenue estimates use Price × Demand and may not reflect actual net revenue.
6. Speech recognition requires an internet connection (Google Web Speech API).

---

## Future Improvements

- Integrate South African public holiday calendar as a feature
- Use Prophet or LSTM for improved time-series modelling
- Add real-time data ingestion from POS systems
- Implement variance-based safety stock (Z-score method)
- Add user authentication for branch-level access control
- Connect to a live ERP/inventory management system
- Add export functionality (PDF reports, Excel)

---

*Academic prototype — results describe this dataset and holdout period only.
Business targets require real-world deployment and post-implementation measurement.*
