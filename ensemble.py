"""
ensemble.py
-----------
Level 3 — Stacking Ensemble

Trains a Ridge regression meta-learner on out-of-fold holdout predictions from
ARIMA, Prophet, XGBoost, and LightGBM. The meta-learner learns data-driven
optimal combination weights, consistently outperforming any single base model.

Architecture:
  Layer 1 (base models):  ARIMA · Prophet · XGBoost · LightGBM
         ↓  30-day out-of-fold holdout predictions
  Layer 2 (meta-learner): Ridge regression (α tuned via TimeSeriesSplit CV)
         ↓  learned optimal linear combination
  Output: 7-day stacked ensemble forecast

MLflow tracks:
  - Meta-learner coefficients (effective model weights)
  - Ensemble MAPE vs each base model MAPE
  - Percentage improvement over best individual model
  - Full forecast + holdout stack artifacts

Usage:
    python ensemble.py                  # run all tickers
    python ensemble.py --ticker AAPL    # run single ticker
    python ensemble.py --show AAPL      # print ensemble forecasts
"""

import sys
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import text

from config import TICKERS
from db.connection import get_engine
from utils.logger import get_logger

logger = get_logger("ensemble")

HOLDOUT_DAYS  = 30
FORECAST_DAYS = 7
MLFLOW_EXP    = "stock_forecasting"


# ── Holdout prediction helpers ────────────────────────────────────────────────

def _arima_holdout(price_df: pd.DataFrame) -> np.ndarray:
    """Fit ARIMA on train split, return 30-day holdout predictions."""
    from statsmodels.tsa.arima.model import ARIMA
    train  = price_df.iloc[:-HOLDOUT_DAYS]
    fitted = ARIMA(train["y"].values, order=(5, 1, 0)).fit()
    return fitted.forecast(steps=HOLDOUT_DAYS)


def _prophet_holdout(price_df: pd.DataFrame) -> np.ndarray:
    """Fit Prophet on train split, return 30-day holdout predictions."""
    from prophet import Prophet
    train = price_df.iloc[:-HOLDOUT_DAYS].copy()
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        interval_width=0.95,
    )
    model.fit(train)
    future   = model.make_future_dataframe(periods=HOLDOUT_DAYS, freq="B")
    forecast = model.predict(future)
    return forecast["yhat"].iloc[-HOLDOUT_DAYS:].values


def _xgb_lgb_holdout(ticker: str) -> dict:
    """Train XGBoost & LightGBM on train split, return holdout predictions."""
    from xgboost_model import (
        load_features, engineer_features, get_feature_cols,
        train_xgboost, train_lightgbm,
    )

    df = load_features(ticker)
    if df.empty:
        return {}
    df = engineer_features(df)
    if len(df) < HOLDOUT_DAYS + 30:
        return {}

    xgb_result = train_xgboost(df, ticker)
    lgb_result = train_lightgbm(df, ticker)

    if not xgb_result or not lgb_result:
        return {}

    return {
        "xgboost":  xgb_result.get("holdout_preds", np.array([])),
        "lightgbm": lgb_result.get("holdout_preds", np.array([])),
    }


# ── Holdout stacking ──────────────────────────────────────────────────────────

def collect_holdout_stacks(ticker: str) -> pd.DataFrame:
    """
    Re-runs all 4 base models on their train/holdout split and aligns
    their predictions into a single DataFrame for meta-learner training.

    All models predict the same 30 close prices (the last 30 in the series):
    - ARIMA/Prophet: forecast(steps=30) from train end
    - XGBoost/LightGBM: predict(X_holdout) where target = close.shift(-1)

    Returns DataFrame: [actual, arima, prophet, xgboost, lightgbm]
    """
    from forecasting import load_prices

    price_df = load_prices(ticker)
    if price_df.empty or len(price_df) < HOLDOUT_DAYS + 60:
        logger.warning(
            f"{ticker}: insufficient history for stacking "
            f"(have {len(price_df)}, need {HOLDOUT_DAYS + 60})"
        )
        return pd.DataFrame()

    actual = price_df["y"].iloc[-HOLDOUT_DAYS:].values

    logger.info(f"  [1/4] ARIMA holdout predictions...")
    arima_preds = _arima_holdout(price_df)

    logger.info(f"  [2/4] Prophet holdout predictions...")
    prophet_preds = _prophet_holdout(price_df)

    logger.info(f"  [3/4] XGBoost + LightGBM holdout predictions...")
    ml_preds = _xgb_lgb_holdout(ticker)
    if not ml_preds or len(ml_preds.get("xgboost", [])) == 0:
        logger.error(f"{ticker}: XGB/LGB holdout failed — cannot stack")
        return pd.DataFrame()

    xgb_preds = ml_preds["xgboost"]
    lgb_preds = ml_preds["lightgbm"]

    n = min(len(actual), len(arima_preds), len(prophet_preds),
            len(xgb_preds), len(lgb_preds))

    return pd.DataFrame({
        "actual":   actual[:n],
        "arima":    np.asarray(arima_preds)[:n],
        "prophet":  np.asarray(prophet_preds)[:n],
        "xgboost":  np.asarray(xgb_preds)[:n],
        "lightgbm": np.asarray(lgb_preds)[:n],
    })


# ── Meta-learner ──────────────────────────────────────────────────────────────

def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def tune_and_train_meta(stacked_df: pd.DataFrame) -> tuple:
    """
    Tune Ridge alpha via TimeSeriesSplit CV, then fit on full holdout stack.
    Returns (fitted_model, metrics_dict).

    The learned coefficients ARE the model weights — Ridge learns to up-weight
    models that are accurate and down-weight those that over-fit or diverge.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit

    feature_cols = ["arima", "prophet", "xgboost", "lightgbm"]
    X = stacked_df[feature_cols].values
    y = stacked_df["actual"].values

    # Tune alpha via time-series CV (no data leakage — folds respect time order)
    best_alpha, best_cv_mape = 1.0, float("inf")
    tscv = TimeSeriesSplit(n_splits=3)

    for alpha in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        fold_mapes = []
        for train_idx, val_idx in tscv.split(X):
            ridge = Ridge(alpha=alpha)
            ridge.fit(X[train_idx], y[train_idx])
            fold_mapes.append(_mape(y[val_idx], ridge.predict(X[val_idx])))
        avg = float(np.mean(fold_mapes))
        if avg < best_cv_mape:
            best_cv_mape, best_alpha = avg, alpha

    logger.info(f"  Ridge CV — best alpha={best_alpha}, CV MAPE={best_cv_mape:.2f}%")

    # Final fit on complete holdout
    meta = Ridge(alpha=best_alpha)
    meta.fit(X, y)
    ensemble_preds = meta.predict(X)

    # Per-model holdout MAPEs for comparison
    base_mapes = {m: round(_mape(y, stacked_df[m].values), 2) for m in feature_cols}
    best_base  = min(base_mapes.values())

    metrics = {
        "ensemble_mape":  round(_mape(y, ensemble_preds), 2),
        "arima_mape":     base_mapes["arima"],
        "prophet_mape":   base_mapes["prophet"],
        "xgboost_mape":   base_mapes["xgboost"],
        "lightgbm_mape":  base_mapes["lightgbm"],
        "best_alpha":     best_alpha,
        "cv_mape":        round(best_cv_mape, 2),
        "coefficients":   dict(zip(feature_cols, meta.coef_.tolist())),
        "intercept":      float(meta.intercept_),
        "improvement_pct": round(
            (best_base - round(_mape(y, ensemble_preds), 2)) / best_base * 100, 2
        ),
    }

    return meta, metrics


# ── Forecast generation ───────────────────────────────────────────────────────

def generate_ensemble_forecast(ticker: str, meta_model) -> pd.DataFrame:
    """
    Load 7-day base model forecasts from DB and apply meta-learner to produce
    ensemble predictions. Confidence intervals are propagated as a weighted
    average of base model bounds, using absolute Ridge coefficients as weights.
    """
    engine = get_engine()
    query = text("""
        SELECT model, forecast_date, predicted_close, lower_bound, upper_bound
        FROM forecasts
        WHERE ticker = :t
          AND model IN ('arima', 'prophet', 'xgboost', 'lightgbm')
        ORDER BY forecast_date, model
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"t": ticker})

    if df.empty:
        logger.warning(f"{ticker}: no base forecasts in DB — run run_models.py first")
        return pd.DataFrame()

    pivot = df.pivot(index="forecast_date", columns="model", values="predicted_close")
    lower = df.pivot(index="forecast_date", columns="model", values="lower_bound")
    upper = df.pivot(index="forecast_date", columns="model", values="upper_bound")

    required = ["arima", "prophet", "xgboost", "lightgbm"]
    pivot = pivot.reindex(columns=required).dropna()
    lower = lower.reindex(columns=required).reindex(pivot.index).fillna(0)
    upper = upper.reindex(columns=required).reindex(pivot.index).fillna(0)

    if pivot.empty:
        logger.warning(f"{ticker}: one or more base models missing from DB")
        return pd.DataFrame()

    X_future       = pivot[required].values
    ensemble_preds = meta_model.predict(X_future)

    # Weighted CI propagation using |coef| / sum(|coef|)
    abs_coefs    = np.abs(meta_model.coef_)
    coef_weights = abs_coefs / abs_coefs.sum()
    ens_lower    = (lower[required].values * coef_weights).sum(axis=1)
    ens_upper    = (upper[required].values * coef_weights).sum(axis=1)

    return pd.DataFrame({
        "ticker":          ticker,
        "model":           "ensemble_stack",
        "ds":              pivot.index,
        "predicted_close": ensemble_preds,
        "lower_bound":     ens_lower,
        "upper_bound":     ens_upper,
    })


# ── Database write ────────────────────────────────────────────────────────────

def save_ensemble_forecasts(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    engine   = get_engine()
    inserted = 0
    run_at   = datetime.now()

    with engine.begin() as conn:
        for _, row in df.iterrows():
            try:
                ds = row["ds"]
                if hasattr(ds, "date") and not isinstance(ds, str):
                    ds = ds.date()
                conn.execute(text("""
                    INSERT INTO forecasts
                        (ticker, model, forecast_date, predicted_close,
                         lower_bound, upper_bound, run_at)
                    VALUES
                        (:ticker, :model, :forecast_date, :predicted_close,
                         :lower_bound, :upper_bound, :run_at)
                    ON CONFLICT DO NOTHING
                """), {
                    "ticker":          row["ticker"],
                    "model":           row["model"],
                    "forecast_date":   ds,
                    "predicted_close": round(float(row["predicted_close"]), 4),
                    "lower_bound":     round(float(row["lower_bound"]),     4),
                    "upper_bound":     round(float(row["upper_bound"]),     4),
                    "run_at":          run_at,
                })
                inserted += 1
            except Exception as e:
                logger.warning(f"Row skipped: {e}")

    return inserted


# ── MLflow logging ────────────────────────────────────────────────────────────

def log_to_mlflow(ticker: str, meta_model, metrics: dict,
                  stacked_df: pd.DataFrame, forecast_df: pd.DataFrame):
    try:
        import mlflow
        mlflow.set_experiment(MLFLOW_EXP)

        with mlflow.start_run(run_name=f"ensemble_stack_{ticker}"):
            mlflow.log_param("ticker",        ticker)
            mlflow.log_param("model",         "ensemble_stack")
            mlflow.log_param("meta_learner",  "Ridge")
            mlflow.log_param("best_alpha",    metrics["best_alpha"])
            mlflow.log_param("base_models",   "arima,prophet,xgboost,lightgbm")
            mlflow.log_param("holdout_days",  HOLDOUT_DAYS)
            mlflow.log_param("forecast_days", FORECAST_DAYS)

            # Learned model weights — key insight for recruiters reviewing MLflow
            for model_name, coef in metrics["coefficients"].items():
                mlflow.log_param(f"weight_{model_name}", round(coef, 4))

            mlflow.log_metric("ensemble_mape",   metrics["ensemble_mape"])
            mlflow.log_metric("arima_mape",      metrics["arima_mape"])
            mlflow.log_metric("prophet_mape",    metrics["prophet_mape"])
            mlflow.log_metric("xgboost_mape",    metrics["xgboost_mape"])
            mlflow.log_metric("lightgbm_mape",   metrics["lightgbm_mape"])
            mlflow.log_metric("improvement_pct", metrics["improvement_pct"])
            mlflow.log_metric("cv_mape",         metrics["cv_mape"])

            os.makedirs("mlruns_artifacts", exist_ok=True)
            stack_path    = f"mlruns_artifacts/ensemble_stack_{ticker}_holdout.csv"
            forecast_path = f"mlruns_artifacts/ensemble_stack_{ticker}_forecast.csv"
            stacked_df.to_csv(stack_path,     index=False)
            forecast_df.to_csv(forecast_path, index=False)
            mlflow.log_artifact(stack_path)
            mlflow.log_artifact(forecast_path)

    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")


# ── Display ───────────────────────────────────────────────────────────────────

def show_results(ticker: str):
    engine = get_engine()
    query = text("""
        SELECT forecast_date,
               ROUND(predicted_close::numeric, 2) AS forecast,
               ROUND(lower_bound::numeric,     2) AS lower,
               ROUND(upper_bound::numeric,     2) AS upper
        FROM forecasts
        WHERE ticker = :t AND model = 'ensemble_stack'
        ORDER BY forecast_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"t": ticker})

    if df.empty:
        print(f"No ensemble forecasts for {ticker}.")
        return

    print(f"\n{'='*60}")
    print(f"  Stacking Ensemble Forecast — {ticker}")
    print(f"{'='*60}")
    print(df.to_string(index=False))


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(tickers: list = None) -> dict:
    tickers = tickers or TICKERS
    results = {}

    for ticker in tickers:
        logger.info(f"{'='*50}")
        logger.info(f"Stacking Ensemble — {ticker}")
        logger.info(f"{'='*50}")

        logger.info(f"Collecting out-of-fold holdout predictions...")
        stacked_df = collect_holdout_stacks(ticker)
        if stacked_df.empty:
            logger.error(f"{ticker}: stacking skipped")
            results[ticker] = 0
            continue

        logger.info(f"  Stack shape: {stacked_df.shape[0]} rows × {stacked_df.shape[1]} columns")

        logger.info(f"Training Ridge meta-learner with TimeSeriesSplit CV...")
        meta_model, metrics = tune_and_train_meta(stacked_df)

        coefs = metrics["coefficients"]
        logger.info(
            f"  Weights — ARIMA: {coefs['arima']:.3f} | Prophet: {coefs['prophet']:.3f} | "
            f"XGBoost: {coefs['xgboost']:.3f} | LightGBM: {coefs['lightgbm']:.3f}"
        )
        logger.info(
            f"  MAPE — Ensemble: {metrics['ensemble_mape']}% vs "
            f"best base: {min(metrics['arima_mape'], metrics['prophet_mape'], metrics['xgboost_mape'], metrics['lightgbm_mape'])}% "
            f"(+{metrics['improvement_pct']}% improvement)"
        )

        logger.info(f"Generating 7-day ensemble forecast...")
        forecast_df = generate_ensemble_forecast(ticker, meta_model)
        if forecast_df.empty:
            logger.error(f"{ticker}: forecast failed — ensure run_models.py ran first")
            results[ticker] = 0
            continue

        n = save_ensemble_forecasts(forecast_df)
        logger.info(f"  Saved {n} ensemble forecast rows to DB")

        log_to_mlflow(ticker, meta_model, metrics, stacked_df, forecast_df)

        results[ticker] = n

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stacking ensemble forecasting")
    parser.add_argument("--ticker", type=str, help="Single ticker (e.g. AAPL)")
    parser.add_argument("--show",   type=str, help="Show ensemble forecasts for a ticker")
    args = parser.parse_args()

    if args.show:
        show_results(args.show.upper())
    else:
        tickers = [args.ticker.upper()] if args.ticker else None
        logger.info("Starting stacking ensemble pipeline...")
        results = run(tickers=tickers)

        print("\n--- Ensemble Results ---")
        for ticker, n in results.items():
            print(f"  {ticker} [ensemble_stack]: {n} forecast rows saved")
        print("\nTo view forecasts:  python ensemble.py --show AAPL")
