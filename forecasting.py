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

import sys
import os
import argparse
import warnings
warnings.filterwarnings("ignore")   # suppress statsmodels convergence warnings

import pandas as pd
import numpy as np
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(__file__))
from config import TICKERS
from db.connection import get_engine
from utils.logger import get_logger

logger = get_logger("forecasting")

# ── Config ────────────────────────────────────────────────────────────────────
FORECAST_DAYS  = 7     # how many trading days ahead to forecast
HOLDOUT_DAYS   = 30    # days held out for model evaluation (RMSE/MAE)
ARIMA_ORDER    = (5, 1, 0)   # (p, d, q) — standard starting point
MLFLOW_EXP     = "stock_forecasting"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_prices(ticker: str, source: str = "yfinance") -> pd.DataFrame:
    """Load daily close prices, return clean time series sorted oldest → newest."""
    engine = get_engine()
    query = text("""
        SELECT ts::date AS ds, close AS y
        FROM raw_prices
        WHERE ticker = :ticker
          AND source = :source
        ORDER BY ts ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker, "source": source})

    if df.empty:
        logger.warning(f"No price data for {ticker}")
        return df

    df["ds"] = pd.to_datetime(df["ds"])
    df["y"]  = df["y"].astype(float)
    df = df.drop_duplicates(subset="ds").sort_values("ds").reset_index(drop=True)
    return df


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """Compute RMSE and MAE between actual and predicted values."""
    actual    = actual.values
    predicted = predicted.values
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae  = float(np.mean(np.abs(actual - predicted)))
    mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100)
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "mape": round(mape, 2)}


# ── ARIMA ─────────────────────────────────────────────────────────────────────

def run_arima(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Fit ARIMA model and generate 7-day forecast.

    Uses last 30 days as holdout to evaluate accuracy before saving.
    Returns forecast DataFrame with columns: ds, predicted_close, lower, upper
    """
    from statsmodels.tsa.arima.model import ARIMA

    if len(df) < HOLDOUT_DAYS + 30:
        logger.warning(f"{ticker}: Not enough data for ARIMA")
        return pd.DataFrame()

    # Split train / holdout
    train = df.iloc[:-HOLDOUT_DAYS]
    holdout = df.iloc[-HOLDOUT_DAYS:]

    try:
        # Fit on training data
        model = ARIMA(train["y"].values, order=ARIMA_ORDER)
        fitted = model.fit()

        # Evaluate on holdout
        holdout_preds = fitted.forecast(steps=HOLDOUT_DAYS)
        metrics = compute_metrics(holdout["y"], pd.Series(holdout_preds))
        logger.info(f"  ARIMA {ticker} holdout — RMSE: {metrics['rmse']} | MAE: {metrics['mae']} | MAPE: {metrics['mape']}%")

        # Refit on full data for actual forecast
        full_model = ARIMA(df["y"].values, order=ARIMA_ORDER)
        full_fitted = full_model.fit()

        # Forecast next FORECAST_DAYS trading days
        forecast_result = full_fitted.get_forecast(steps=FORECAST_DAYS)
        forecast_mean   = forecast_result.predicted_mean
        conf_int        = forecast_result.conf_int(alpha=0.05)

        # Generate future dates (skip weekends)
        last_date = df["ds"].iloc[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS)

        forecast_df = pd.DataFrame({
            "ds":              future_dates,
            "predicted_close": forecast_mean,
            "lower_bound":     conf_int[:, 0],
            "upper_bound":     conf_int[:, 1],
            "model":           "arima",
            "ticker":          ticker,
        })

        # Log to MLflow
        _log_mlflow(ticker, "arima", metrics, forecast_df)

        return forecast_df

    except Exception as e:
        logger.error(f"ARIMA failed for {ticker}: {e}")
        return pd.DataFrame()


# ── Prophet ───────────────────────────────────────────────────────────────────

def run_prophet(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Fit Prophet model and generate 7-day forecast.

    Prophet handles:
      - Weekly seasonality (market days)
      - Yearly seasonality (quarterly earnings cycles)
      - Trend changepoints automatically

    Returns forecast DataFrame with columns: ds, predicted_close, lower, upper
    """
    try:
        from prophet import Prophet
    except ImportError:
        logger.error("Prophet not installed. Run: pip install prophet")
        return pd.DataFrame()

    if len(df) < HOLDOUT_DAYS + 30:
        logger.warning(f"{ticker}: Not enough data for Prophet")
        return pd.DataFrame()

    # Split train / holdout
    train   = df.iloc[:-HOLDOUT_DAYS].copy()
    holdout = df.iloc[-HOLDOUT_DAYS:].copy()

    try:
        # Prophet requires columns named 'ds' and 'y' — already done
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,   # controls trend flexibility
            interval_width=0.95,             # 95% confidence interval
        )
        model.fit(train)

        # Evaluate on holdout
        holdout_future = model.make_future_dataframe(
            periods=HOLDOUT_DAYS, freq="B"   # 'B' = business days
        )
        holdout_forecast = model.predict(holdout_future)
        holdout_preds    = holdout_forecast["yhat"].iloc[-HOLDOUT_DAYS:]
        metrics = compute_metrics(holdout["y"].reset_index(drop=True),
                                  holdout_preds.reset_index(drop=True))
        logger.info(f"  Prophet {ticker} holdout — RMSE: {metrics['rmse']} | MAE: {metrics['mae']} | MAPE: {metrics['mape']}%")

        # Refit on full data
        full_model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            interval_width=0.95,
        )
        full_model.fit(df)

        # Forecast next FORECAST_DAYS business days
        future = full_model.make_future_dataframe(periods=FORECAST_DAYS, freq="B")
        forecast = full_model.predict(future)

        # Extract only the future rows
        forecast_future = forecast[forecast["ds"] > df["ds"].max()].head(FORECAST_DAYS)

        forecast_df = pd.DataFrame({
            "ds":              pd.to_datetime(forecast_future["ds"]),
            "predicted_close": forecast_future["yhat"].values,
            "lower_bound":     forecast_future["yhat_lower"].values,
            "upper_bound":     forecast_future["yhat_upper"].values,
            "model":           "prophet",
            "ticker":          ticker,
        })

        # Log to MLflow
        _log_mlflow(ticker, "prophet", metrics, forecast_df)

        return forecast_df

    except Exception as e:
        logger.error(f"Prophet failed for {ticker}: {e}")
        return pd.DataFrame()


# ── MLflow logging ────────────────────────────────────────────────────────────

def _log_mlflow(ticker: str, model_name: str, metrics: dict, forecast_df: pd.DataFrame):
    """Log a model run to MLflow tracking."""
    try:
        import mlflow
        mlflow.set_experiment(MLFLOW_EXP)

        with mlflow.start_run(run_name=f"{model_name}_{ticker}"):
            mlflow.log_param("ticker",   ticker)
            mlflow.log_param("model",    model_name)
            mlflow.log_param("forecast_days", FORECAST_DAYS)
            mlflow.log_param("holdout_days",  HOLDOUT_DAYS)

            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("mae",  metrics["mae"])
            mlflow.log_metric("mape", metrics["mape"])

            # Save forecast as CSV artifact
            os.makedirs("mlruns_artifacts", exist_ok=True)
            artifact_path = f"mlruns_artifacts/{model_name}_{ticker}_forecast.csv"
            forecast_df.to_csv(artifact_path, index=False)
            mlflow.log_artifact(artifact_path)

    except Exception as e:
        logger.warning(f"MLflow logging failed (non-critical): {e}")


# ── Database write ────────────────────────────────────────────────────────────

def save_forecasts(df: pd.DataFrame) -> int:
    """
    Save forecast rows to the forecasts table.
    Returns number of rows inserted.
    """
    if df.empty:
        return 0

    engine = get_engine()
    inserted = 0
    run_at = pd.Timestamp.now()

    with engine.begin() as conn:
        for _, row in df.iterrows():
            try:
                conn.execute(
                    text("""
                        INSERT INTO forecasts
                            (ticker, model, forecast_date, predicted_close,
                             lower_bound, upper_bound, run_at)
                        VALUES
                            (:ticker, :model, :forecast_date, :predicted_close,
                             :lower_bound, :upper_bound, :run_at)
                        ON CONFLICT DO NOTHING
                    """),
                    {
                        "ticker":          row["ticker"],
                        "model":           row["model"],
                        "forecast_date":   row["ds"].date() if hasattr(row["ds"], "date") else row["ds"],
                        "predicted_close": round(float(row["predicted_close"]), 4),
                        "lower_bound":     round(float(row["lower_bound"]),     4),
                        "upper_bound":     round(float(row["upper_bound"]),     4),
                        "run_at":          run_at,
                    }
                )
                inserted += 1
            except Exception as e:
                logger.warning(f"Row skipped: {e}")

    return inserted


# ── Display helpers ───────────────────────────────────────────────────────────

def show_forecasts(ticker: str):
    """Print saved forecasts for a ticker."""
    engine = get_engine()
    query = text("""
        SELECT
            model,
            forecast_date,
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

    print(f"\n{'='*70}")
    print(f"  7-Day Forecasts for {ticker}")
    print(f"{'='*70}")
    print(df.to_string(index=False))


def compare_models(ticker: str):
    """Side-by-side comparison of ARIMA vs Prophet for a ticker."""
    engine = get_engine()
    query = text("""
        SELECT
            forecast_date,
            MAX(CASE WHEN model='arima'   THEN ROUND(predicted_close::numeric,2) END) AS arima,
            MAX(CASE WHEN model='prophet' THEN ROUND(predicted_close::numeric,2) END) AS prophet,
            MAX(CASE WHEN model='arima'   THEN ROUND(lower_bound::numeric,2) END)    AS arima_low,
            MAX(CASE WHEN model='prophet' THEN ROUND(lower_bound::numeric,2) END)    AS prophet_low,
            MAX(CASE WHEN model='arima'   THEN ROUND(upper_bound::numeric,2) END)    AS arima_high,
            MAX(CASE WHEN model='prophet' THEN ROUND(upper_bound::numeric,2) END)    AS prophet_high
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

    print(f"\n{'='*75}")
    print(f"  ARIMA vs Prophet — {ticker} — 7-Day Forecast Comparison")
    print(f"{'='*75}")
    print(df.to_string(index=False))
    print()
    print("  Tip: Wider confidence interval = model is less certain about that day")
    print()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(tickers: list[str] = None, models: list[str] = None) -> dict:
    """
    Full pipeline: load prices → fit ARIMA → fit Prophet → save forecasts.
    Returns dict of {ticker: {model: rows_inserted}}.
    """
    tickers = tickers or TICKERS
    models  = models  or ["arima", "prophet"]
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
            arima_forecast = run_arima(df, ticker)
            n = save_forecasts(arima_forecast)
            ticker_results["arima"] = n
            logger.info(f"  ARIMA: {n} forecast rows saved")

        if "prophet" in models:
            logger.info(f"  Fitting Prophet for {ticker}...")
            prophet_forecast = run_prophet(df, ticker)
            n = save_forecasts(prophet_forecast)
            ticker_results["prophet"] = n
            logger.info(f"  Prophet: {n} forecast rows saved")

        results[ticker] = ticker_results

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock price forecasting engine")
    parser.add_argument("--ticker",  type=str, help="Single ticker (e.g. AAPL)")
    parser.add_argument("--model",   type=str, choices=["arima", "prophet"], help="Run one model only")
    parser.add_argument("--show",    type=str, help="Print forecasts for a ticker")
    parser.add_argument("--compare", type=str, help="Compare ARIMA vs Prophet for a ticker")
    args = parser.parse_args()

    if args.show:
        show_forecasts(args.show.upper())
    elif args.compare:
        compare_models(args.compare.upper())
    else:
        tickers = [args.ticker.upper()] if args.ticker else None
        models  = [args.model] if args.model else None

        logger.info("Starting forecasting pipeline...")
        results = run(tickers=tickers, models=models)

        print("\n--- Results ---")
        for ticker, model_results in results.items():
            for model, n in model_results.items():
                print(f"  {ticker} [{model}]: {n} forecast rows saved")

        print("\nTo view forecasts:  python forecasting.py --show AAPL")
        print("To compare models:  python forecasting.py --compare AAPL")
        print("To open MLflow UI:  mlflow ui  (then open http://localhost:5000)")
        