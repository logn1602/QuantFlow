"""XGBoost and LightGBM forecasting.

Trains one regressor per ticker on the feature matrix from
quantflow.features.engineering, then rolls a 7-day forecast forward
recursively.

Note on what the reported metric measures: the holdout MAPE logged here is
one-step-ahead. Each holdout row carries the *actual* lagged close, rolling
statistics and indicators for its day, so the model is predicting tomorrow
given a true today. The 7-day forecast published to the dashboard is a
different task — generate_forecast feeds each prediction back in as the next
day's lag, so errors compound. The holdout number does not describe it.
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sqlalchemy import text

from quantflow.config import TICKERS
from quantflow.db.connection import get_engine
from quantflow.db.forecasts import save_forecasts
from quantflow.db.metrics import save_model_metrics
from quantflow.features.engineering import (
    engineer_features,
    get_feature_cols,
    load_features,
)
from quantflow.utils.logger import get_logger

warnings.filterwarnings("ignore")

logger = get_logger("boosting")

FORECAST_DAYS = 7
HOLDOUT_DAYS = 30
MLFLOW_EXP = "stock_forecasting"


def train_xgboost(df: pd.DataFrame, ticker: str) -> dict:
    """
    Train XGBoost regressor for price prediction.
    Uses walk-forward validation on holdout set.
    Returns dict with model, metrics, feature importance.
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("XGBoost not installed. Run: pip install xgboost")
        return {}

    feature_cols = get_feature_cols()
    available = [c for c in feature_cols if c in df.columns]

    # Fill any remaining NaNs
    df[available] = df[available].fillna(0)

    # Train/holdout split
    train = df.iloc[:-HOLDOUT_DAYS].copy()
    holdout = df.iloc[-HOLDOUT_DAYS:].copy()

    X_train = train[available]
    y_train = train["target_price"]
    X_hold = holdout[available]
    y_hold = holdout["target_price"]

    # XGBoost regressor
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_hold, y_hold)], verbose=False)

    # Evaluate on holdout
    preds = model.predict(X_hold)
    rmse = float(np.sqrt(np.mean((y_hold.values - preds) ** 2)))
    mae = float(np.mean(np.abs(y_hold.values - preds)))
    mape = float(np.mean(np.abs((y_hold.values - preds) / y_hold.values)) * 100)

    logger.info(
        f"  XGBoost {ticker} holdout — RMSE: {round(rmse, 4)} | MAE: {round(mae, 4)} | MAPE: {round(mape, 2)}%"
    )

    # Feature importance
    importance = pd.DataFrame(
        {
            "feature": available,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return {
        "model": model,
        "metrics": {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 2),
        },
        "importance": importance,
        "features": available,
        "X_last": df[available].iloc[-1:],
        "last_close": float(df["close"].iloc[-1]),
        "last_date": df["date"].iloc[-1],
        "holdout_preds": preds,
    }


def train_lightgbm(df: pd.DataFrame, ticker: str) -> dict:
    """
    Train LightGBM regressor — faster than XGBoost, often similar accuracy.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("LightGBM not installed. Run: pip install lightgbm")
        return {}

    feature_cols = get_feature_cols()
    available = [c for c in feature_cols if c in df.columns]
    df[available] = df[available].fillna(0)

    train = df.iloc[:-HOLDOUT_DAYS].copy()
    holdout = df.iloc[-HOLDOUT_DAYS:].copy()

    X_train = train[available]
    y_train = train["target_price"]
    X_hold = holdout[available]
    y_hold = holdout["target_price"]

    model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_hold)
    rmse = float(np.sqrt(np.mean((y_hold.values - preds) ** 2)))
    mae = float(np.mean(np.abs(y_hold.values - preds)))
    mape = float(np.mean(np.abs((y_hold.values - preds) / y_hold.values)) * 100)

    logger.info(
        f"  LightGBM {ticker} holdout — RMSE: {round(rmse, 4)} | MAE: {round(mae, 4)} | MAPE: {round(mape, 2)}%"
    )

    importance = pd.DataFrame(
        {
            "feature": available,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return {
        "model": model,
        "metrics": {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 2),
        },
        "importance": importance,
        "features": available,
        "X_last": df[available].iloc[-1:],
        "last_close": float(df["close"].iloc[-1]),
        "last_date": df["date"].iloc[-1],
        "holdout_preds": preds,
    }


# ── Forecasting ───────────────────────────────────────────────────────────────


def generate_forecast(
    result: dict, model_name: str, ticker: str, price_history: list | None = None
) -> pd.DataFrame:
    """
    Generate 7-day forecast using the trained model.

    Each step rolls the full feature vector forward using the predicted price:
      - Lag features (1/2/3/5/10) cascade from the rolling price buffer
      - Returns (1d/5d) recomputed from the buffer
      - Rolling mean/std (5/10/20) recomputed from the buffer
      - Bollinger position and width derived from updated rolling stats
      - Exogenous features (RSI, MACD, sentiment, anomaly, volume) stay frozen
        at their last known values — we cannot extrapolate them forward.

    price_history: full list of historical closes used to seed lag/rolling calcs.
    """
    if not result:
        return pd.DataFrame()

    model = result["model"]
    current_X = result["X_last"].copy()
    last_date = result["last_date"]

    # Seed the rolling buffer with the last 30 actual closes so all windows
    # (up to rolling_mean_20 + lag_10) have enough history from day one.
    buf = list(price_history[-30:]) if price_history else [result["last_close"]]

    forecasts = []
    future_dates = pd.bdate_range(
        start=pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=FORECAST_DAYS
    )

    for forecast_date in future_dates:
        pred = float(model.predict(current_X)[0])

        forecasts.append(
            {
                "ticker": ticker,
                "model": model_name,
                "ds": forecast_date,
                "predicted_close": pred,
                "lower_bound": pred * 0.98,
                "upper_bound": pred * 1.02,
            }
        )

        # Append prediction to buffer, then recompute all price-derived features
        buf.append(pred)
        prices = np.array(buf)

        # Lag features — each lag reads the correct historical position
        for lag in [1, 2, 3, 5, 10]:
            col = f"close_lag_{lag}"
            if col in current_X.columns and len(prices) > lag:
                current_X[col] = prices[-(lag + 1)]

        # Returns
        if "return_1d" in current_X.columns and len(prices) >= 2:
            current_X["return_1d"] = (prices[-1] - prices[-2]) / prices[-2]
        if "return_5d" in current_X.columns and len(prices) >= 6:
            current_X["return_5d"] = (prices[-1] - prices[-6]) / prices[-6]

        # Rolling mean and std
        for window in [5, 10, 20]:
            if len(prices) >= window:
                w_prices = prices[-window:]
                mean_col = f"rolling_mean_{window}"
                std_col = f"rolling_std_{window}"
                if mean_col in current_X.columns:
                    current_X[mean_col] = float(np.mean(w_prices))
                if std_col in current_X.columns:
                    current_X[std_col] = (
                        float(np.std(w_prices, ddof=1)) if len(w_prices) > 1 else 0.0
                    )

        # Bollinger Band position and width (derived from updated rolling_mean/std_20)
        if (
            "rolling_mean_20" in current_X.columns
            and "rolling_std_20" in current_X.columns
        ):
            mean20 = float(current_X["rolling_mean_20"].values[0])
            std20 = float(current_X["rolling_std_20"].values[0])
            bb_upper = mean20 + 2 * std20
            bb_lower = mean20 - 2 * std20
            bb_range = bb_upper - bb_lower
            if "bb_position" in current_X.columns and bb_range > 0:
                current_X["bb_position"] = float(
                    np.clip((pred - bb_lower) / bb_range, 0, 1)
                )
            if "bb_width" in current_X.columns and mean20 > 0:
                current_X["bb_width"] = bb_range / mean20

        # RSI, MACD, volume, sentiment, anomaly stay frozen — exogenous signals
        # that cannot be extrapolated from price alone.

    return pd.DataFrame(forecasts)


# ── MLflow logging ────────────────────────────────────────────────────────────


def log_mlflow(
    ticker: str,
    model_name: str,
    metrics: dict,
    importance: pd.DataFrame,
    forecast_df: pd.DataFrame,
):
    """Log experiment to MLflow."""
    try:
        import mlflow

        mlflow.set_experiment(MLFLOW_EXP)

        with mlflow.start_run(run_name=f"{model_name}_{ticker}"):
            mlflow.log_param("ticker", ticker)
            mlflow.log_param("model", model_name)
            mlflow.log_param("features_used", len(importance))
            mlflow.log_param("forecast_days", FORECAST_DAYS)

            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("mae", metrics["mae"])
            mlflow.log_metric("mape", metrics["mape"])

            # Top 5 features as params
            for i, row in importance.head(5).iterrows():
                mlflow.log_param(
                    f"top_feature_{list(importance.index).index(i) + 1}", row["feature"]
                )

            os.makedirs("mlruns_artifacts", exist_ok=True)
            imp_path = f"mlruns_artifacts/{model_name}_{ticker}_importance.csv"
            importance.to_csv(imp_path, index=False)
            mlflow.log_artifact(imp_path)

    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")


# ── Display ───────────────────────────────────────────────────────────────────


def show_results(ticker: str):
    """Print forecasts and feature importance for a ticker."""
    engine = get_engine()

    # Forecasts
    query = text("""
        SELECT model, forecast_date,
               ROUND(predicted_close::numeric, 2) AS forecast,
               ROUND(lower_bound::numeric, 2)     AS lower,
               ROUND(upper_bound::numeric, 2)     AS upper
        FROM forecasts
        WHERE ticker = :t AND model IN ('xgboost', 'lightgbm')
        ORDER BY model, forecast_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"t": ticker})

    if df.empty:
        print(f"No XGBoost/LightGBM forecasts for {ticker}.")
        return

    print(f"\n{'=' * 65}")
    print(f"  XGBoost + LightGBM Forecasts — {ticker}")
    print(f"{'=' * 65}")
    print(df.to_string(index=False))


def compare_all_models(ticker: str):
    """Compare ARIMA, Prophet, XGBoost, LightGBM for a ticker."""
    engine = get_engine()
    query = text("""
        SELECT
            model,
            forecast_date,
            ROUND(predicted_close::numeric, 2) AS forecast
        FROM forecasts
        WHERE ticker = :t
        ORDER BY model, forecast_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"t": ticker})

    if df.empty:
        print(f"No forecasts found for {ticker}")
        return

    pivot = df.pivot(index="forecast_date", columns="model", values="forecast")
    print(f"\n{'=' * 75}")
    print(f"  All Models Forecast Comparison — {ticker}")
    print(f"{'=' * 75}")
    print(pivot.to_string())
    print()


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run(tickers: list[str] | None = None) -> dict:
    tickers = tickers or TICKERS
    results = {}

    for ticker in tickers:
        logger.info(f"Training XGBoost + LightGBM for {ticker}...")

        df = load_features(ticker)
        if df.empty:
            results[ticker] = {"xgboost": 0, "lightgbm": 0}
            continue

        df = engineer_features(df)
        if len(df) < 60:
            logger.warning(f"{ticker}: Not enough rows after feature engineering")
            results[ticker] = {"xgboost": 0, "lightgbm": 0}
            continue

        logger.info(
            f"  Feature matrix: {len(df)} rows × {len(get_feature_cols())} features"  # noqa: RUF001
        )

        ticker_results = {}

        price_buf = list(df["close"].values)

        # XGBoost
        xgb_result = train_xgboost(df, ticker)
        if xgb_result:
            xgb_forecast = generate_forecast(xgb_result, "xgboost", ticker, price_buf)
            n = save_forecasts(xgb_forecast)
            log_mlflow(
                ticker,
                "xgboost",
                xgb_result["metrics"],
                xgb_result["importance"],
                xgb_forecast,
            )
            save_model_metrics(ticker, "xgboost", xgb_result["metrics"], HOLDOUT_DAYS)
            ticker_results["xgboost"] = n
            logger.info(f"  XGBoost: {n} forecast rows saved")

            top5 = xgb_result["importance"].head(5)
            logger.info(f"  Top features: {', '.join(top5['feature'].tolist())}")

        # LightGBM
        lgb_result = train_lightgbm(df, ticker)
        if lgb_result:
            lgb_forecast = generate_forecast(lgb_result, "lightgbm", ticker, price_buf)
            n = save_forecasts(lgb_forecast)
            log_mlflow(
                ticker,
                "lightgbm",
                lgb_result["metrics"],
                lgb_result["importance"],
                lgb_forecast,
            )
            save_model_metrics(ticker, "lightgbm", lgb_result["metrics"], HOLDOUT_DAYS)
            ticker_results["lightgbm"] = n
            logger.info(f"  LightGBM: {n} forecast rows saved")

        results[ticker] = ticker_results

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost + LightGBM forecasting")
    parser.add_argument("--ticker", type=str, help="Single ticker (e.g. AAPL)")
    parser.add_argument("--show", type=str, help="Show forecasts for ticker")
    parser.add_argument("--compare", type=str, help="Compare all models for ticker")
    args = parser.parse_args()

    if args.show:
        show_results(args.show.upper())
    elif args.compare:
        compare_all_models(args.compare.upper())
    else:
        tickers = [args.ticker.upper()] if args.ticker else None
        logger.info("Starting XGBoost + LightGBM training pipeline...")
        results = run(tickers=tickers)
        print("\n--- Results ---")
        for ticker, model_results in results.items():
            for model, n in model_results.items():
                print(f"  {ticker} [{model}]: {n} forecast rows saved")
        print("\nTo view forecasts:   python xgboost_model.py --show AAPL")
        print("To compare models:   python xgboost_model.py --compare AAPL")
