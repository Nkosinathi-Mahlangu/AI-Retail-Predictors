# AI Retail Predictors

A Python-based demand forecasting and stockout detection system for
non-technical filling-station forecourt managers.

---

## Data Privacy

All data processed by this system is synthetic, sourced from the Kaggle
"Retail Store Inventory and Demand Forecasting" dataset. No real customer,
financial, or personally identifiable information is processed by this system.
All findings and outputs are illustrative only and do not represent real-world
outcomes.

---

## Project Overview

AI Retail Predictors helps filling-station managers make smarter daily
stock-ordering decisions. The system ingests historical sales and inventory
data, applies time-series decomposition, trains regression-based ML models
and a lightweight deep-learning model, then serves demand forecasts and
stockout alerts through a natural-language chatbot.

All models run on CPU-only student laptops within a Python 3.x environment.

---

## AI Components Addressed

| Component | Description | Location |
|---|---|---|
| Machine Learning | Regression-based demand forecasting models | [src/ml_models/](src/ml_models/) |
| Time Series Analysis | Decomposition and trend/seasonality analysis | [src/time_series/](src/time_series/) |
| NLP / Speech Processing | Natural-language query processing for the chatbot | [src/nlp/](src/nlp/) |
| Deep Learning | Neural network-based forecasting model | [src/deep_learning/](src/deep_learning/) |
| Chatbot / Softbot | Conversational interface for store managers | [src/chatbot/](src/chatbot/) |
| Data Preprocessing | Cleaning, relabelling, and feature engineering | [src/data_preprocessing/](src/data_preprocessing/) |

---

## Repository Structure

```
ai-retail-predictors/
├── README.md
├── requirements.txt
├── docs/
│   ├── requirements_spec.txt
│   ├── business_objectives.txt
│   └── final_report.docx
├── data/
│   ├── raw/                        # Original sales_data.csv lives here
│   └── processed/                  # Cleaned outputs written here
├── src/
│   ├── data_preprocessing/         # Req 1 & 2: cleaning + train/test split
│   ├── ml_models/                  # Req 4: Random Forest / Gradient Boosting
│   ├── time_series/                # Req 3: seasonal decomposition
│   ├── nlp/                        # Req 9: NLP query processor
│   ├── deep_learning/              # Req 5: LSTM / Dense neural network
│   └── chatbot/                    # Req 8: chatbot interface
├── config/
│   └── category_mapping.json       # Filling-station category relabelling rules
├── outputs/                        # Charts, reports, and model artefacts
├── logs/                           # Runtime log files
├── notebooks/                      # Jupyter notebooks for exploration
└── tests/                          # Unit and integration tests
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: Python 3.11.x, Windows. No GPU required.

---

## Running the Pipeline

```bash
python main.py
```

Or run the data cleaner in isolation:

```bash
python src/data_preprocessing/data_cleaner.py "data/raw/sales_data.csv"
```

---

## Limitations

- The dataset used is synthetic and has not been validated against operational
  data from a real filling-station.
- Model results and forecasts are not operationally validated and must not be
  used for real business decisions.
- The category-relabelling applied to the dataset is an approximation and may
  not reflect actual filling-station product categories.
