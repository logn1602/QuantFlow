"""XGBoost and LightGBM forecasting.

Trains two regressors per ticker and family on the feature matrix from
quantflow.features.engineering, and rolls a 7-day forecast forward
recursively from the newer one. They have different jobs:
  - the scored model is fitted on the train split and evaluated on the
    holdout; its holdout predictions also feed the ensemble stack;
  - the live model has the same settings, is fitted on every labelled row,
    and produces the published forecast, starting from the newest trading
    day. ARIMA and Prophet refit on the full series before forecasting too.

Note on what the reported metric measures: the holdout MAPE logged here is
one-step-ahead. Each holdout row carries the *actual* lagged close, rolling
statistics and indicators for its day, so the model is predicting tomorrow
given a true today. The 7-day forecast published to the dashboard is a
different task: generate_forecast feeds each prediction back in as the next
day's lag, so errors compound. The holdout number does not describe it.
"""

import argparse
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from quantflow.config import TICKERS
from quantflow.db.connection import get_engine
from quantflow.db.forecasts import save_forecasts
from quantflow.db.metrics import save_model_metrics
from quantflow.evaluation.metrics import regression_metrics
from quantflow.evaluation.splits import HOLDOUT_DAYS, train_holdout_split
from quantflow.features.engineering import (
    engineer_features,
    get_feature_cols,
    load_features,
)
from quantflow.models import FORECAST_DAYS
from quantflow.utils.logger import get_logger
from quantflow.utils.trading_calendar import next_trading_days

warnings.filterwarnings("ignore")

logger = get_logger("boosting")

MLFLOW_EXP = "stock_forecasting"

# One set of settings per family, shared by the scored and the live model, so
# the model that is evaluated is the model that forecasts.
MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": 0,
    },
    "lightgbm": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    },
}


def _new_regressor(model_name: str):
    """An unfitted regressor of the family; None if its package is missing.

    Raises ValueError for a family MODEL_PARAMS does not define, which would
    otherwise fall through to LightGBM and be stored under the wrong name.
    """
    if model_name not in MODEL_PARAMS:
        raise ValueError(
            f"unknown model family {model_name!r}; expected one of "
            f"{', '.join(sorted(MODEL_PARAMS))}"
        )
    try:
        if model_name == "xgboost":
            import xgboost as xgb

            return xgb.XGBRegressor(**MODEL_PARAMS["xgboost"])
        import lightgbm as lgb

        return lgb.LGBMRegressor(**MODEL_PARAMS["lightgbm"])
    except ImportError:
        logger.error(f"{model_name} not installed. Run: pip install {model_name}")
        return None


def _train_scored(model_name: str, df: pd.DataFrame, ticker: str) -> dict:
    """Fit on the train split, score on the holdout.

    Returns dict with the fitted model, holdout metrics and predictions,
    feature importance and the feature list; {} if the package is missing.
    """
    model = _new_regressor(model_name)
    if model is None:
        return {}

    feature_cols = get_feature_cols()
    available = [c for c in feature_cols if c in df.columns]

    # Fill any remaining NaNs
    df[available] = df[available].fillna(0)

    # Train/holdout split
    train, holdout = train_holdout_split(df, copy=True)

    X_train = train[available]
    y_train = train["target_price"]
    X_hold = holdout[available]
    y_hold = holdout["target_price"]

    if model_name == "xgboost":
        model.fit(X_train, y_train, eval_set=[(X_hold, y_hold)], verbose=False)
    else:
        model.fit(X_train, y_train)

    # Evaluate on holdout
    preds = model.predict(X_hold)
    scores = regression_metrics(y_hold, preds)

    label = "XGBoost" if model_name == "xgboost" else "LightGBM"
    logger.info(
        f"  {label} {ticker} holdout: RMSE: {scores['rmse']} | MAE: {scores['mae']} | MAPE: {scores['mape']}%"
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
        "metrics": scores,
        "importance": importance,
        "features": available,
        "holdout_preds": preds,
    }


def train_xgboost(df: pd.DataFrame, ticker: str) -> dict:
    """
    Train XGBoost regressor for price prediction on the train split and
    score it on the holdout. df must hold labelled rows only.
    Returns dict with model, metrics, feature importance.
    """
    return _train_scored("xgboost", df, ticker)


def train_lightgbm(df: pd.DataFrame, ticker: str) -> dict:
    """
    Train LightGBM regressor, faster than XGBoost and often similar accuracy,
    on the train split and score it on the holdout. df must hold labelled rows.
    """
    return _train_scored("lightgbm", df, ticker)


def fit_live_model(model_name: str, labelled: pd.DataFrame, features: list[str]):
    """A fresh model with the scored model's settings, fitted on every
    labelled row, including the 30 the scored model held out."""
    model = _new_regressor(model_name)
    if model is None:
        return None
    model.fit(labelled[features].fillna(0), labelled["target_price"])
    return model


# ── Forecasting ───────────────────────────────────────────────────────────────


def generate_forecast(
    model,
    X_last: pd.DataFrame,
    last_date,
    price_history: list,
    model_name: str,
    ticker: str,
) -> pd.DataFrame:
    """
    Generate a 7-day forecast from the newest trading day.

    X_last is that day's feature row and last_date its date, so the first
    step predicts the next trading day's close. price_history holds the
    actual closes up to and including last_date.

    Each step rolls the full feature vector forward using the predicted price:
      - Lag features (1/2/3/5/10) cascade from the rolling price buffer
      - Returns (1d/5d) recomputed from the buffer
      - Rolling mean/std (5/10/20) recomputed from the buffer
      - Bollinger position and width derived from updated rolling stats
      - Exogenous features (RSI, MACD, sentiment, anomaly, volume) stay frozen
        at their last known values, since we cannot extrapolate them forward.
    """
    current_X = X_last.copy()

    # Seed the rolling buffer with the last 30 actual closes so all windows
    # (up to rolling_mean_20 + lag_10) have enough history from day one.
    buf = list(price_history[-30:])

    forecasts = []
    # Trading days, not weekdays: a forecast must not promise a close on a
    # market holiday.
    future_dates = next_trading_days(last_date, FORECAST_DAYS)

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

        # Lag features: each lag reads the correct historical position
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

        # RSI, MACD, volume, sentiment, anomaly stay frozen: exogenous signals
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

        raw = load_features(ticker)
        if raw.empty:
            results[ticker] = {"xgboost": 0, "lightgbm": 0}
            continue

        # Keep the newest row: it has no target, but the live forecast starts
        # from its features. Training and scoring use the labelled rows only.
        features = engineer_features(raw, drop_unlabeled=False)
        labelled = features.dropna(subset=["target_price"]).copy()
        if len(labelled) < 60:
            logger.warning(f"{ticker}: Not enough rows after feature engineering")
            results[ticker] = {"xgboost": 0, "lightgbm": 0}
            continue

        logger.info(
            f"  Feature matrix: {len(labelled)} rows × {len(get_feature_cols())} features"  # noqa: RUF001
        )

        newest = features.iloc[[-1]]
        last_date = newest["date"].iloc[0]
        price_history = list(features["close"].to_numpy())

        ticker_results = {}

        for model_name, train in (
            ("xgboost", train_xgboost),
            ("lightgbm", train_lightgbm),
        ):
            result = train(labelled, ticker)
            if not result:
                continue

            live_model = fit_live_model(model_name, labelled, result["features"])
            forecast = generate_forecast(
                live_model,
                newest[result["features"]].fillna(0),
                last_date,
                price_history,
                model_name,
                ticker,
            )
            n = save_forecasts(forecast)
            log_mlflow(
                ticker, model_name, result["metrics"], result["importance"], forecast
            )
            save_model_metrics(ticker, model_name, result["metrics"], HOLDOUT_DAYS)
            ticker_results[model_name] = n
            logger.info(f"  {model_name}: {n} forecast rows saved")

            if model_name == "xgboost":
                top5 = result["importance"].head(5)
                logger.info(f"  Top features: {', '.join(top5['feature'].tolist())}")

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
