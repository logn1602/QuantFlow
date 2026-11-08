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

import numpy as np
import pandas as pd
from sqlalchemy import text

from quantflow.config import TICKERS
from quantflow.db.connection import get_engine
from quantflow.db.forecasts import save_forecasts
from quantflow.db.metrics import save_model_metrics
from quantflow.db.prices import load_daily_close as load_prices
from quantflow.evaluation.metrics import regression_metrics
from quantflow.evaluation.splits import HOLDOUT_DAYS, train_holdout_split
from quantflow.models import FORECAST_DAYS
from quantflow.utils.logger import get_logger
from quantflow.utils.trading_calendar import next_trading_days

warnings.filterwarnings("ignore")

logger = get_logger("forecasting")

# ARIMA's order (Phase 3, E4): None chooses it for each fit, by AIC on that
# fit's own closes, over ARIMA_ORDER_GRID; a tuple fixes it (it was
# (5, 1, 0)). Every order in the grid differences once and has no trend, so
# their AICs compare like with like; (0, 1, 0) is the naive forecast. On the
# validation dates choosing by AIC was lower on only 3 of 8 tickers (it chose
# (0, 1, 0) in 139 of 336 fits): dropped, the order stays fixed
# (docs/evaluation.md, section 16).
ARIMA_ORDER: tuple[int, int, int] | None = (5, 1, 0)
ARIMA_ORDER_GRID = [(p, 1, q) for p in range(6) for q in range(3)]
MLFLOW_EXP = "stock_forecasting"
# The seed Prophet's interval is simulated from (prophet_forecast).
PROPHET_INTERVAL_SEED = 42


# Prophet's own 'auto' setting turns yearly seasonality off below this span
# (prophet/forecaster.py). Forcing it on with less fits a yearly curve to a
# single season, which then repeats in every forecast.
MIN_YEARLY_SPAN = pd.Timedelta(days=730)


def prophet_settings(history: pd.DataFrame) -> dict:
    """Constructor arguments for every Prophet fit: the holdout fit, the live
    fit and the ensemble's holdout fit. One definition, so the model that is
    scored and the model that forecasts cannot drift apart.

    Yearly seasonality depends on how much history this particular fit sees,
    so a ticker near the two-year mark can have it in the live fit and not in
    the holdout fit, which trains on 30 fewer days.
    """
    span = history["ds"].max() - history["ds"].min()
    return {
        "daily_seasonality": False,
        "weekly_seasonality": True,
        "yearly_seasonality": bool(span >= MIN_YEARLY_SPAN),
        "changepoint_prior_scale": 0.05,
        "interval_width": 0.95,
    }


# ── Forecasts ─────────────────────────────────────────────────────────────────
#
# One function per model makes every forecast that model produces: the
# holdout score, the ensemble's holdout stack and the live 7-day forecast.
# The ensemble used to fit its own copies, with the ARIMA order written out
# as (5, 1, 0), so a change to ARIMA_ORDER would have reached the scored and
# live ARIMA but not the column the meta-learner weights.
#
# Both return the same columns, yhat, yhat_lower and yhat_upper (a 95%
# interval), so a caller can treat either model's forecast the same way.


def fit_arima(y):
    """ARIMA fitted on the closes y, and its order: ARIMA_ORDER if set,
    else the order in ARIMA_ORDER_GRID with the lowest AIC. An order that
    cannot be fitted is skipped; RuntimeError if none can."""
    from statsmodels.tsa.arima.model import ARIMA

    y = np.asarray(y, dtype=float)
    if ARIMA_ORDER is not None:
        return ARIMA(y, order=ARIMA_ORDER).fit(), tuple(ARIMA_ORDER)
    best, best_order = None, None
    for order in ARIMA_ORDER_GRID:
        try:
            fitted = ARIMA(y, order=order).fit()
        except (np.linalg.LinAlgError, ValueError):
            continue
        # Strictly lower: of equal AICs the first, the simpler, order wins.
        if np.isfinite(fitted.aic) and (best is None or fitted.aic < best.aic):
            best, best_order = fitted, order
    if best is None:
        raise RuntimeError("no ARIMA order in the grid could be fitted")
    return best, best_order


def arima_forecast(y, steps: int) -> pd.DataFrame:
    """ARIMA fitted on the closes y (fit_arima), forecast `steps` ahead,
    with the order it used.

    get_forecast's mean is what forecast() returns, so the holdout score,
    which used forecast(), is unchanged by going through here.
    """
    fitted, order = fit_arima(y)
    result = fitted.get_forecast(steps=steps)
    interval = np.asarray(result.conf_int(alpha=0.05))
    return pd.DataFrame(
        {
            "yhat": np.asarray(result.predicted_mean, dtype=float),
            "yhat_lower": interval[:, 0],
            "yhat_upper": interval[:, 1],
            "arima_order": str(tuple(order)),
        }
    )


def prophet_forecast(history: pd.DataFrame, dates) -> pd.DataFrame:
    """Prophet fitted on history (ds, y), predicted on the given dates.

    Predicting on actual dates rather than make_future_dataframe(freq="B")
    keeps market holidays out: weekday stepping scored each holdout
    prediction after a holiday against the next trading day's close, and
    promised live closes on holidays.
    """
    from prophet import Prophet

    model = Prophet(**prophet_settings(history))
    model.fit(history.copy())
    future = pd.DataFrame({"ds": pd.DatetimeIndex(pd.to_datetime(dates))})
    # Prophet simulates its interval from numpy's global random state, so
    # the same fit gave a different band on every run. A fixed seed makes
    # the band reproducible (in the evaluation's worker processes too); the
    # caller's random state is put back afterwards.
    state = np.random.get_state()
    np.random.seed(PROPHET_INTERVAL_SEED)
    try:
        forecast = model.predict(future)
    finally:
        np.random.set_state(state)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def arima_holdout(price_df: pd.DataFrame) -> np.ndarray:
    """ARIMA fitted on the train split, its predictions for the holdout."""
    train, _ = train_holdout_split(price_df)
    return arima_forecast(train["y"], HOLDOUT_DAYS)["yhat"].to_numpy()


def prophet_holdout(price_df: pd.DataFrame) -> np.ndarray:
    """Prophet fitted on the train split, predicted on the holdout's own
    dates, so the result lines up row for row with the holdout's closes."""
    train, holdout = train_holdout_split(price_df, copy=True)
    return prophet_forecast(train, holdout["ds"])["yhat"].to_numpy()


# ── Metrics ───────────────────────────────────────────────────────────────────


compute_metrics = regression_metrics


# ── ARIMA ─────────────────────────────────────────────────────────────────────


def run_arima(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, dict]:
    """
    Fit ARIMA and generate 7-day forecast with confidence intervals.
    Returns (forecast_df, holdout_metrics). Metrics are {} on failure.
    """
    if len(df) < HOLDOUT_DAYS + 30:
        logger.warning(f"{ticker}: Not enough data for ARIMA")
        return pd.DataFrame(), {}

    _, holdout = train_holdout_split(df)

    try:
        metrics = compute_metrics(holdout["y"].to_numpy(), arima_holdout(df))
        logger.info(
            f"  ARIMA {ticker} holdout — "
            f"RMSE: {metrics['rmse']} | MAE: {metrics['mae']} | MAPE: {metrics['mape']}%"
        )

        forecast = arima_forecast(df["y"], FORECAST_DAYS)

        # Trading days, not weekdays: a forecast must not promise a close on
        # a market holiday.
        future_dates = next_trading_days(df["ds"].iloc[-1], FORECAST_DAYS)

        forecast_df = pd.DataFrame(
            {
                "ds": [d.date() for d in future_dates],
                "predicted_close": forecast["yhat"].to_numpy(),
                "lower_bound": forecast["yhat_lower"].to_numpy(),
                "upper_bound": forecast["yhat_upper"].to_numpy(),
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
    # Only checks that Prophet is installed; prophet_forecast imports it.
    try:
        import prophet  # noqa: F401
    except ImportError:
        logger.error("Prophet not installed. Run: pip install prophet")
        return pd.DataFrame(), {}

    if len(df) < HOLDOUT_DAYS + 30:
        logger.warning(f"{ticker}: Not enough data for Prophet")
        return pd.DataFrame(), {}

    _, holdout = train_holdout_split(df)

    try:
        metrics = compute_metrics(holdout["y"].to_numpy(), prophet_holdout(df))
        logger.info(
            f"  Prophet {ticker} holdout: "
            f"RMSE: {metrics['rmse']} | MAE: {metrics['mae']} | MAPE: {metrics['mape']}%"
        )

        forecast_future = prophet_forecast(
            df, next_trading_days(df["ds"].iloc[-1], FORECAST_DAYS)
        )

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
