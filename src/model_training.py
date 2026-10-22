"""
SmartFuel AI — Model Training
================================
Trains and compares four demand-forecasting models on the engineered feature
set using a strict CHRONOLOGICAL train/test split (no data leakage).

Models compared
---------------
1. Naive Lag-1 Baseline      — predicts tomorrow's demand = today's demand
2. Ridge Regression           — linear model with L2 regularisation
3. Random Forest Regressor    — ensemble of decision trees
4. HistGradientBoosting (HGB) — gradient-boosted trees (best performer expected)

Evaluation metrics
------------------
MAE   — Mean Absolute Error (primary selection metric)
RMSE  — Root Mean Squared Error
R²    — Coefficient of determination
MAPE  — Mean Absolute Percentage Error (zero-demand rows handled explicitly)

The best model is selected by lowest MAE and saved to models/model_bundle.joblib.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")


# ── Helper: safe MAPE ─────────────────────────────────────────────────────
def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAPE that safely handles zero-demand observations.
    Rows where |y_true| < 1 are excluded from the MAPE calculation to avoid
    division-by-zero; the exclusion is documented in Model Performance.
    """
    mask = np.abs(y_true) >= 1.0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ── Build sklearn pipeline for a given estimator ──────────────────────────
def _build_pipeline(
    estimator,
    cat_cols: list,
    num_cols: list,
    scale_numeric: bool = False,
    impute_numeric: bool = True,
) -> Pipeline:
    """
    Build a full sklearn Pipeline with:
      - OneHotEncoder for categorical columns
      - SimpleImputer (median) + optional StandardScaler for numeric columns
    """
    # Numeric transformer: always impute; optionally scale
    if scale_numeric:
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
    else:
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    transformers = [
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False), cat_cols),
        ("num", num_transformer, num_cols),
    ]

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: dict,
    train_frac: float = 0.80,
    verbose: bool = True,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Train all four models, evaluate on the chronological holdout, return the
    best pipeline and the metrics/predictions DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Output of feature_engineering.build_features()
    feature_cols : dict
        Output of feature_engineering.get_feature_columns()
    train_frac : float
        Fraction of the chronological data used for training (default 0.80).
    verbose : bool

    Returns
    -------
    bundle : dict
        {'pipeline': best_pipeline, 'best_model': name, 'features': ...,
         'categorical': ..., 'numeric': ..., 'train_cutoff': date}
    metrics_df : pd.DataFrame
    test_preds_df : pd.DataFrame
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    cat_cols = feature_cols["categorical"]
    num_cols = feature_cols["numeric"]

    # Only keep columns that actually exist in df
    existing_cat = [c for c in cat_cols if c in df.columns]
    existing_num = [c for c in num_cols if c in df.columns]
    all_features = existing_cat + existing_num

    # Chronological split
    df_sorted = df.sort_values("Date").reset_index(drop=True)
    cutoff_idx = int(len(df_sorted) * train_frac)
    cutoff_date = df_sorted.iloc[cutoff_idx]["Date"]

    train = df_sorted[df_sorted["Date"] <= cutoff_date].copy()
    test  = df_sorted[df_sorted["Date"]  > cutoff_date].copy()

    if verbose:
        print(f"[Training] Train rows : {len(train):,}  (up to {cutoff_date.date()})")
        print(f"[Training] Test rows  : {len(test):,}")

    X_train = train[all_features]
    y_train = train["Demand"].to_numpy()
    X_test  = test[all_features]
    y_test  = test["Demand"].to_numpy()

    # ── Model definitions ─────────────────────────────────────────────────
    model_defs = {
        "Ridge Regression": _build_pipeline(
            Ridge(alpha=1.0), existing_cat, existing_num,
            scale_numeric=True, impute_numeric=True,
        ),
        "Random Forest": _build_pipeline(
            RandomForestRegressor(
                n_estimators=150, max_depth=12, min_samples_leaf=5,
                n_jobs=-1, random_state=42
            ),
            existing_cat, existing_num,
            impute_numeric=True,
        ),
        "HistGradientBoosting": _build_pipeline(
            HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.07, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=42,
            ),
            existing_cat, existing_num,
            impute_numeric=True,
        ),
    }

    results = []
    pipelines = {}

    # ── Naive baseline (lag-1) ─────────────────────────────────────────────
    if "lag_1" in test.columns:
        naive_pred = np.maximum(test["lag_1"].fillna(test["Demand"].mean()).to_numpy(), 0)
        results.append({
            "Model": "Naive Lag-1 Baseline",
            "MAE":   round(mean_absolute_error(y_test, naive_pred), 4),
            "RMSE":  round(np.sqrt(mean_squared_error(y_test, naive_pred)), 4),
            "R2":    round(r2_score(y_test, naive_pred), 4),
            "MAPE":  round(_mape(y_test, naive_pred), 4),
            "Train Rows": len(train),
            "Test Rows":  len(test),
        })
        if verbose:
            print(f"[Training] Naive Lag-1  MAE={results[-1]['MAE']:.4f}  RMSE={results[-1]['RMSE']:.4f}")

    # ── ML models ─────────────────────────────────────────────────────────
    for name, pipe in model_defs.items():
        if verbose:
            print(f"[Training] Fitting {name} …")
        pipe.fit(X_train, y_train)
        pred = np.maximum(pipe.predict(X_test), 0)
        mae  = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2   = r2_score(y_test, pred)
        mape = _mape(y_test, pred)
        results.append({
            "Model": name,
            "MAE":   round(mae, 4),
            "RMSE":  round(rmse, 4),
            "R2":    round(r2, 4),
            "MAPE":  round(mape, 4),
            "Train Rows": len(train),
            "Test Rows":  len(test),
        })
        pipelines[name] = (pipe, pred)
        if verbose:
            print(f"[Training]   MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  MAPE={mape:.2f}%")

    # ── Select best model by MAE ───────────────────────────────────────────
    ml_results = [r for r in results if r["Model"] != "Naive Lag-1 Baseline"]
    best_row   = min(ml_results, key=lambda r: r["MAE"])
    best_name  = best_row["Model"]
    best_pipe, best_pred = pipelines[best_name]

    if verbose:
        print(f"\n[Training] ✓ Best model: {best_name}  (MAE={best_row['MAE']})")

    # ── Test predictions DataFrame ─────────────────────────────────────────
    test_preds_df = test[["Date", "Store ID", "Product ID", "Product Name",
                           "Category", "Region", "Demand",
                           "Inventory Level", "Units Ordered"]].copy()
    test_preds_df["Predicted Demand"] = np.round(best_pred, 2)
    test_preds_df["Error"]             = np.round(test_preds_df["Demand"] - test_preds_df["Predicted Demand"], 2)
    test_preds_df["Abs Error"]         = test_preds_df["Error"].abs()

    # ── Metrics DataFrame ─────────────────────────────────────────────────
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(os.path.join(MODEL_DIR, "metrics.csv"), index=False)

    # ── Save test predictions ─────────────────────────────────────────────
    test_preds_df.to_csv(os.path.join(MODEL_DIR, "test_predictions.csv"), index=False)

    # ── Bundle ────────────────────────────────────────────────────────────
    bundle = {
        "pipeline":       best_pipe,
        "best_model":     best_name,
        "features":       all_features,
        "categorical":    existing_cat,
        "numeric":        existing_num,
        "train_cutoff":   str(cutoff_date.date()),
        "train_rows":     len(train),
        "test_rows":      len(test),
    }
    joblib.dump(bundle, os.path.join(MODEL_DIR, "model_bundle.joblib"))
    if verbose:
        print(f"[Training] Model bundle saved → {MODEL_DIR}/model_bundle.joblib")

    return bundle, metrics_df, test_preds_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ROOT)
    from src.data_cleaning import clean
    from src.feature_engineering import build_features, get_feature_columns

    df_clean, _ = clean(verbose=True)
    df_feat = build_features(df_clean, verbose=True)
    feat_cols = get_feature_columns()
    bundle, metrics, preds = train_and_evaluate(df_feat, feat_cols, verbose=True)
    print(f"\nModel comparison:\n{metrics[['Model','MAE','RMSE','R2','MAPE']].to_string(index=False)}")
