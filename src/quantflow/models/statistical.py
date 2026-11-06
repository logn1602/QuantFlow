"""
forecasting.py
--------------
Phase 4 — Forecasting Engine

Trains two models per ticker and saves 7-day price forecasts:
  1. ARIMA      : Classical time series baseline (statsmodels)
  2. Prophet    : Meta's forecasting library — handles weekends,
                  seasonality, and trend changes automatically

MLflow tracks every experiment run:
  - Model type, ticker, training date
  - Metrics: RMSE, MAE on last 30 days (holdout)
  - Forecast artifacts saved as CSV

Forecasts are written to the forecasts table.

Usage:
    python forecasting.py                   # forecast all tickers, both models
    python forecasting.py --ticker AAPL     # one ticker, both models
    python forecasting.py --model prophet   # all tickers, prophet only
    python forecasting.py --show AAPL       # print saved forecasts for AAPL
    python forecasting.py --compare AAPL    # compare ARIMA vs Prophet for AAPL
"""

import argparse
import os
import warnings

import pandas as pd
from sqlalchemy import text

from quantflow.config import TICKERS
from quantflow.db.connection import get_engine
from quantflow.db.forecasts import save_forecasts
from quantflow.db.metrics import save_model_metrics
from quantflow.db.prices import load_daily_close as load_prices
from quantflow.evaluation.metrics import regression_metrics
from quantflow.evaluation.splits import HOLDOUT_DAYS, train_holdout_split
from quantflow.utils.logger import get_logger
from quantflow.utils.trading_calendar import next_trading_days

warnings.filterwarnings("ignore")

logger = get_logger("forecasting")

FORECAST_DAYS = 7
ARIMA_ORDER = (5, 1, 0)
MLFLOW_EXP = "stock_forecasting"


def prophet_settings(history: pd.DataFrame) -> dict:
    """Constructor arguments for every Prophet fit: the holdout fit, the live
    fit and the ensemble's holdout fit. One definition, so the model that is
    scored and the model that forecasts cannot drift apart."""
    return {
        "daily_seasonality": False,
        "weekly_seasonality": True,
        "yearly_seasonality": True,
        "changepoint_prior_scale": 0.05,
        "interval_width": 0.95,
    }


# ── Data loading ──────────────────────────────────────────────────────────────


# ── Metrics ───────────────────────────────────────────────────────────────────


compute_metrics = regression_metrics


# ── ARIMA ─────────────────────────────────────────────────────────────────────


def run_arima(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, dict]:
    """
    Fit ARIMA and generate 7-day forecast with confidence intervals.
    Returns (forecast_df, holdout_metrics). Metrics are {} on failure.
    """
    from statsmodels.tsa.arima.model import ARIMA

    if len(df) < HOLDOUT_DAYS + 30:
        logger.warning(f"{ticker}: Not enough data for ARIMA")
        return pd.DataFrame(), {}

    train, holdout = train_holdout_split(df)

    try:
        model = ARIMA(train["y"].values, order=ARIMA_ORDER)
        fitted = model.fit()
        holdout_preds = fitted.forecast(steps=HOLDOUT_DAYS)
        metrics = compute_metrics(holdout["y"], pd.Series(holdout_preds))
        logger.info(
            f"  ARIMA {ticker} holdout — "
            f"RMSE: {metrics['rmse']} | MAE: {metrics['mae']} | MAPE: {metrics['mape']}%"
        )

        full_model = ARIMA(df["y"].values, order=ARIMA_ORDER)
        full_fitted = full_model.fit()

        forecast_result = full_fitted.get_forecast(steps=FORECAST_DAYS)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.05)

        # Trading days, not weekdays: a forecast must not promise a close on
        # a market holiday.
        future_dates = next_trading_days(df["ds"].iloc[-1], FORECAST_DAYS)

        forecast_df = pd.DataFrame(
            {
                "ds": [d.date() for d in future_dates],
                "predicted_close": forecast_mean,
                "lower_bound": conf_int[:, 0],
                "upper_bound": conf_int[:, 1],
                "model": "arima",
                "ticker": ticker,
            }
        )

        _log_mlflow(ticker, "arima", metrics, forecast_df)
        return forecast_df, metrics

    except Exception as e:
        logger.error(f"ARIMA failed for {ticker}: {e}")
        return pd.DataFrame(), {}


# ── Prophet ───────────────────────────────────────────────────────────────────


def run_prophet(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, dict]:
    """
    Fit Prophet and generate 7-day forecast with uncertainty intervals.
    Returns (forecast_df, holdout_metrics). Metrics are {} on failure.
    """
    try:
        from prophet import Prophet
    except ImportError:
        logger.error("Prophet not installed. Run: pip install prophet")
        return pd.DataFrame(), {}

    if len(df) < HOLDOUT_DAYS + 30:
        logger.warning(f"{ticker}: Not enough data for Prophet")
        return pd.DataFrame(), {}

    train, holdout = train_holdout_split(df, copy=True)

    try:
        model = Prophet(**prophet_settings(train))
        model.fit(train)

        # Predict on the holdout's own dates. make_future_dataframe(freq="B")
        # stepped through every weekday, so after a market holiday inside the
        # window each prediction was scored against the next trading day's
        # close, and the final close was never predicted at all.
        holdout_preds = model.predict(holdout[["ds"]])["yhat"]
        metrics = compute_metrics(holdout["y"].to_numpy(), holdout_preds.to_numpy())
        logger.info(
            f"  Prophet {ticker} holdout: "
            f"RMSE: {metrics['rmse']} | MAE: {metrics['mae']} | MAPE: {metrics['mape']}%"
        )

        full_model = Prophet(**prophet_settings(df))
        full_model.fit(df)

        future = pd.DataFrame(
            {"ds": next_trading_days(df["ds"].iloc[-1], FORECAST_DAYS)}
        )
        forecast_future = full_model.predict(future)

        forecast_df = pd.DataFrame(
            {
                "ds": [pd.Timestamp(d).date() for d in forecast_future["ds"]],
                "predicted_close": forecast_future["yhat"].values,
                "lower_bound": forecast_future["yhat_lower"].values,
                "upper_bound": forecast_future["yhat_upper"].values,
                "model": "prophet",
                "ticker": ticker,
            }
        )

        _log_mlflow(ticker, "prophet", metrics, forecast_df)
        return forecast_df, metrics

    except Exception as e:
        logger.error(f"Prophet failed for {ticker}: {e}")
        return pd.DataFrame(), {}


# ── MLflow logging ────────────────────────────────────────────────────────────


def _log_mlflow(ticker: str, model_name: str, metrics: dict, forecast_df: pd.DataFrame):
    try:
        import mlflow

        mlflow.set_experiment(MLFLOW_EXP)

        with mlflow.start_run(run_name=f"{model_name}_{ticker}"):
            mlflow.log_param("ticker", ticker)
            mlflow.log_param("model", model_name)
            mlflow.log_param("forecast_days", FORECAST_DAYS)
            mlflow.log_param("holdout_days", HOLDOUT_DAYS)
            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("mae", metrics["mae"])
            mlflow.log_metric("mape", metrics["mape"])

            os.makedirs("mlruns_artifacts", exist_ok=True)
            artifact_path = f"mlruns_artifacts/{model_name}_{ticker}_forecast.csv"
            forecast_df.to_csv(artifact_path, index=False)
            mlflow.log_artifact(artifact_path)

    except Exception as e:
        logger.warning(f"MLflow logging failed (non-critical): {e}")


# ── Database write ────────────────────────────────────────────────────────────


# ── Display helpers ───────────────────────────────────────────────────────────


def show_forecasts(ticker: str):
    engine = get_engine()
    query = text("""
        SELECT
            model,
            forecast_date::date                AS date,
            ROUND(predicted_close::numeric, 2) AS forecast,
            ROUND(lower_bound::numeric, 2)     AS lower,
            ROUND(upper_bound::numeric, 2)     AS upper,
            DATE(run_at)                       AS run_date
        FROM forecasts
        WHERE ticker = :ticker
        ORDER BY model, forecast_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker})

    if df.empty:
        print(f"No forecasts found for {ticker}.")
        return

    print(f"\n{'=' * 70}")
    print(f"  7-Day Forecasts for {ticker}")
    print(f"{'=' * 70}")
    print(df.to_string(index=False))


def compare_models(ticker: str):
    engine = get_engine()
    query = text("""
        SELECT
            forecast_date::date AS date,
            MAX(CASE WHEN model='arima'   THEN ROUND(predicted_close::numeric,2) END) AS arima,
            MAX(CASE WHEN model='prophet' THEN ROUND(predicted_close::numeric,2) END) AS prophet
        FROM forecasts
        WHERE ticker = :ticker
        GROUP BY forecast_date
        ORDER BY forecast_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker})

    if df.empty:
        print(f"No forecasts found for {ticker}.")
        return

    print(f"\n{'=' * 55}")
    print(f"  ARIMA vs Prophet — {ticker}")
    print(f"{'=' * 55}")
    print(df.to_string(index=False))
    print()


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run(tickers: list[str] | None = None, models: list[str] | None = None) -> dict:
    tickers = tickers or TICKERS
    models = models or ["arima", "prophet"]
    results = {}

    for ticker in tickers:
        logger.info(f"Forecasting {ticker}...")
        df = load_prices(ticker)
        if df.empty:
            results[ticker] = {"arima": 0, "prophet": 0}
            continue

        ticker_results = {}

        if "arima" in models:
            logger.info(f"  Fitting ARIMA for {ticker}...")
            arima_forecast, arima_metrics = run_arima(df, ticker)
            n = save_forecasts(arima_forecast)
            save_model_metrics(ticker, "arima", arima_metrics, HOLDOUT_DAYS)
            ticker_results["arima"] = n
            logger.info(f"  ARIMA: {n} forecast rows saved")

        if "prophet" in models:
            logger.info(f"  Fitting Prophet for {ticker}...")
            prophet_forecast, prophet_metrics = run_prophet(df, ticker)
            n = save_forecasts(prophet_forecast)
            save_model_metrics(ticker, "prophet", prophet_metrics, HOLDOUT_DAYS)
            ticker_results["prophet"] = n
            logger.info(f"  Prophet: {n} forecast rows saved")

        results[ticker] = ticker_results

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock price forecasting engine")
    parser.add_argument("--ticker", type=str, help="Single ticker (e.g. AAPL)")
    parser.add_argument("--model", type=str, choices=["arima", "prophet"])
    parser.add_argument("--show", type=str, help="Print forecasts for a ticker")
    parser.add_argument("--compare", type=str, help="Compare ARIMA vs Prophet")
    args = parser.parse_args()

    if args.show:
        show_forecasts(args.show.upper())
    elif args.compare:
        compare_models(args.compare.upper())
    else:
        tickers = [args.ticker.upper()] if args.ticker else None
        models = [args.model] if args.model else None

        logger.info("Starting forecasting pipeline...")
        results = run(tickers=tickers, models=models)

        print("\n--- Results ---")
        for ticker, model_results in results.items():
            for model, n in model_results.items():
                print(f"  {ticker} [{model}]: {n} forecast rows saved")

        print("\nTo view forecasts:  python forecasting.py --show AAPL")
        print("To compare models:  python forecasting.py --compare AAPL")
        print("To open MLflow UI:  mlflow ui  (then open http://localhost:5000)")
