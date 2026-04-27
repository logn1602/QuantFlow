"""
xgboost_model.py
----------------
Level 2 — XGBoost/LightGBM Forecasting Engine

Unlike ARIMA/Prophet which only use price history,
this model uses ALL available signals as features:

  Price features:
    - Lagged closes (1, 2, 3, 5, 10 days)
    - Rolling mean/std (5, 10, 20 days)
    - Daily return, 5-day return

  Technical indicators:
    - RSI (14)
    - MACD, MACD signal, MACD histogram
    - Bollinger Band position (where price sits in the band)
    - BB width (volatility measure)

  Anomaly features:
    - Z-score of recent price movements
    - Binary anomaly flag

  Sentiment features:
    - Daily average compound score
    - Rolling 3-day sentiment
    - Positive/negative article counts

Target: Next day's closing price (regression)
Also trains a direction classifier: will price go UP or DOWN tomorrow?

MLflow tracks every experiment with full feature importance.

Usage:
    python xgboost_model.py                   # train all tickers
    python xgboost_model.py --ticker AAPL     # train one ticker
    python xgboost_model.py --show AAPL       # show forecast + feature importance
    python xgboost_model.py --compare AAPL    # compare XGB vs ARIMA vs Prophet
"""

import sys
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from sqlalchemy import text
from datetime import datetime, timedelta

from config import TICKERS
from db.connection import get_engine
from utils.logger import get_logger

logger = get_logger("xgboost_model")

FORECAST_DAYS  = 7
HOLDOUT_DAYS   = 30
MLFLOW_EXP     = "stock_forecasting"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_features(ticker: str) -> pd.DataFrame:
    """
    Load and merge all available signals for a ticker:
    prices + technical indicators + anomalies + sentiment
    """
    engine = get_engine()

    # Base prices
    prices = pd.read_sql(text("""
        SELECT ts::date AS date, open, high, low, close, volume
        FROM raw_prices
        WHERE ticker = :t AND source = 'yfinance'
        ORDER BY ts ASC
    """), engine, params={"t": ticker})

    if prices.empty:
        logger.warning(f"No price data for {ticker}")
        return pd.DataFrame()

    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    # Technical indicators
    indicators = pd.read_sql(text("""
        SELECT ts::date AS date, rsi_14, macd, macd_signal, macd_hist,
               bb_upper, bb_middle, bb_lower
        FROM technical_indicators
        WHERE ticker = :t
        ORDER BY ts ASC
    """), engine, params={"t": ticker})
    indicators["date"] = pd.to_datetime(indicators["date"])
    indicators = indicators.drop_duplicates("date")

    # Anomalies
    anomalies = pd.read_sql(text("""
        SELECT ts::date AS date, zscore
        FROM anomalies
        WHERE ticker = :t
        ORDER BY ts ASC
    """), engine, params={"t": ticker})
    anomalies["date"] = pd.to_datetime(anomalies["date"])
    anomalies = anomalies.drop_duplicates("date")

    # Sentiment — daily average
    sentiment = pd.read_sql(text("""
        SELECT
            published_at::date                  AS date,
            AVG(compound)                       AS sentiment_compound,
            COUNT(*) FILTER (WHERE sentiment='positive') AS pos_count,
            COUNT(*) FILTER (WHERE sentiment='negative') AS neg_count,
            COUNT(*)                            AS article_count
        FROM news_sentiment
        WHERE ticker = :t
        GROUP BY published_at::date
        ORDER BY date ASC
    """), engine, params={"t": ticker})
    sentiment["date"] = pd.to_datetime(sentiment["date"])

    # Merge everything
    df = prices.merge(indicators, on="date", how="left")
    df = df.merge(anomalies,  on="date", how="left")
    df = df.merge(sentiment,  on="date", how="left")

    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix from raw merged data.
    All features are derived from past data only — no future leakage.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # ── Price-based features ─────────────────────────────────────────────────
    # Lagged closes
    for lag in [1, 2, 3, 5, 10]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)

    # Daily return
    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)

    # Rolling statistics
    for window in [5, 10, 20]:
        df[f"rolling_mean_{window}"] = df["close"].rolling(window).mean()
        df[f"rolling_std_{window}"]  = df["close"].rolling(window).std()

    # High-Low range
    df["hl_range"] = df["high"] - df["low"]

    # Volume change
    df["volume_change"] = df["volume"].pct_change(1)

    # ── Bollinger Band features ──────────────────────────────────────────────
    # BB position: where is price relative to the bands? (0=lower, 1=upper)
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / bb_range.replace(0, np.nan)

    # BB width: measures volatility
    df["bb_width"] = bb_range / df["bb_middle"].replace(0, np.nan)

    # ── Anomaly features ─────────────────────────────────────────────────────
    df["zscore"]       = df["zscore"].fillna(0)
    df["is_anomaly"]   = (df["zscore"].abs() >= 2.0).astype(int)

    # ── Sentiment features ───────────────────────────────────────────────────
    df["sentiment_compound"] = df["sentiment_compound"].fillna(0)
    df["pos_count"]          = df["pos_count"].fillna(0)
    df["neg_count"]          = df["neg_count"].fillna(0)
    df["article_count"]      = df["article_count"].fillna(0)

    # Rolling sentiment (3-day average)
    df["sentiment_3d"] = df["sentiment_compound"].rolling(3).mean().fillna(0)

    # Sentiment momentum
    df["sentiment_change"] = df["sentiment_compound"].diff().fillna(0)

    # ── Target variables ─────────────────────────────────────────────────────
    df["target_price"]     = df["close"].shift(-1)   # next day's price
    df["target_direction"] = (df["target_price"] > df["close"]).astype(int)  # 1=up, 0=down

    # Drop rows with NaN targets or insufficient history
    df = df.dropna(subset=["target_price", "close_lag_10", "rolling_mean_20"])

    return df


def get_feature_cols() -> list[str]:
    """Return the list of feature column names used for training."""
    return [
        # Price lags
        "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_5", "close_lag_10",
        # Returns
        "return_1d", "return_5d",
        # Rolling stats
        "rolling_mean_5", "rolling_mean_10", "rolling_mean_20",
        "rolling_std_5",  "rolling_std_10",  "rolling_std_20",
        # OHLCV
        "open", "high", "low", "volume", "hl_range", "volume_change",
        # Technical indicators
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_position", "bb_width",
        # Anomaly
        "zscore", "is_anomaly",
        # Sentiment
        "sentiment_compound", "sentiment_3d", "sentiment_change",
        "pos_count", "neg_count", "article_count",
    ]


# ── Model training ────────────────────────────────────────────────────────────

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
    available    = [c for c in feature_cols if c in df.columns]

    # Fill any remaining NaNs
    df[available] = df[available].fillna(0)

    # Train/holdout split
    train = df.iloc[:-HOLDOUT_DAYS].copy()
    holdout = df.iloc[-HOLDOUT_DAYS:].copy()

    X_train = train[available]
    y_train = train["target_price"]
    X_hold  = holdout[available]
    y_hold  = holdout["target_price"]

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
    model.fit(X_train, y_train,
              eval_set=[(X_hold, y_hold)],
              verbose=False)

    # Evaluate on holdout
    preds  = model.predict(X_hold)
    rmse   = float(np.sqrt(np.mean((y_hold.values - preds) ** 2)))
    mae    = float(np.mean(np.abs(y_hold.values - preds)))
    mape   = float(np.mean(np.abs((y_hold.values - preds) / y_hold.values)) * 100)

    logger.info(f"  XGBoost {ticker} holdout — RMSE: {round(rmse,4)} | MAE: {round(mae,4)} | MAPE: {round(mape,2)}%")

    # Feature importance
    importance = pd.DataFrame({
        "feature":    available,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "model":         model,
        "metrics":       {"rmse": round(rmse,4), "mae": round(mae,4), "mape": round(mape,2)},
        "importance":    importance,
        "features":      available,
        "X_last":        df[available].iloc[-1:],
        "last_close":    float(df["close"].iloc[-1]),
        "last_date":     df["date"].iloc[-1],
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
    available    = [c for c in feature_cols if c in df.columns]
    df[available] = df[available].fillna(0)

    train   = df.iloc[:-HOLDOUT_DAYS].copy()
    holdout = df.iloc[-HOLDOUT_DAYS:].copy()

    X_train = train[available]
    y_train = train["target_price"]
    X_hold  = holdout[available]
    y_hold  = holdout["target_price"]

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
    rmse  = float(np.sqrt(np.mean((y_hold.values - preds) ** 2)))
    mae   = float(np.mean(np.abs(y_hold.values - preds)))
    mape  = float(np.mean(np.abs((y_hold.values - preds) / y_hold.values)) * 100)

    logger.info(f"  LightGBM {ticker} holdout — RMSE: {round(rmse,4)} | MAE: {round(mae,4)} | MAPE: {round(mape,2)}%")

    importance = pd.DataFrame({
        "feature":    available,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "model":         model,
        "metrics":       {"rmse": round(rmse,4), "mae": round(mae,4), "mape": round(mape,2)},
        "importance":    importance,
        "features":      available,
        "X_last":        df[available].iloc[-1:],
        "last_close":    float(df["close"].iloc[-1]),
        "last_date":     df["date"].iloc[-1],
        "holdout_preds": preds,
    }


# ── Forecasting ───────────────────────────────────────────────────────────────

def generate_forecast(result: dict, model_name: str, ticker: str) -> pd.DataFrame:
    """
    Generate 7-day forecast using the trained model.
    Uses last known feature vector, rolling forward each day.
    """
    if not result:
        return pd.DataFrame()

    model     = result["model"]
    X_last    = result["X_last"].copy()
    last_date = result["last_date"]
    last_close = result["last_close"]

    forecasts = []
    current_close = last_close
    current_X     = X_last.copy()

    future_dates = pd.bdate_range(
        start=pd.Timestamp(last_date) + pd.Timedelta(days=1),
        periods=FORECAST_DAYS
    )

    for forecast_date in future_dates:
        pred = float(model.predict(current_X)[0])

        # Simple confidence interval: ± 2% for XGB (wider than ARIMA)
        forecasts.append({
            "ticker":          ticker,
            "model":           model_name,
            "ds":              forecast_date,
            "predicted_close": pred,
            "lower_bound":     pred * 0.98,
            "upper_bound":     pred * 1.02,
        })

        # Roll forward: update lag features for next prediction
        if "close_lag_1" in current_X.columns:
            for lag in [10, 5, 3, 2]:
                col_curr = f"close_lag_{lag}"
                col_prev = f"close_lag_{lag-1}" if lag > 1 else None
                if col_curr in current_X.columns:
                    if col_prev and col_prev in current_X.columns:
                        current_X[col_curr] = current_X[col_prev].values
                    else:
                        current_X[col_curr] = current_close
            current_X["close_lag_1"] = current_close
            current_close = pred

    return pd.DataFrame(forecasts)


# ── Database write ────────────────────────────────────────────────────────────

def save_forecasts(df: pd.DataFrame) -> int:
    """Save XGBoost/LightGBM forecasts to the forecasts table."""
    if df.empty:
        return 0

    engine = get_engine()
    inserted = 0
    run_at = datetime.now()

    with engine.begin() as conn:
        for _, row in df.iterrows():
            try:
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
                    "forecast_date":   row["ds"].date(),
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

def log_mlflow(ticker: str, model_name: str, metrics: dict,
               importance: pd.DataFrame, forecast_df: pd.DataFrame):
    """Log experiment to MLflow."""
    try:
        import mlflow
        mlflow.set_experiment(MLFLOW_EXP)

        with mlflow.start_run(run_name=f"{model_name}_{ticker}"):
            mlflow.log_param("ticker",        ticker)
            mlflow.log_param("model",         model_name)
            mlflow.log_param("features_used", len(importance))
            mlflow.log_param("forecast_days", FORECAST_DAYS)

            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("mae",  metrics["mae"])
            mlflow.log_metric("mape", metrics["mape"])

            # Top 5 features as params
            for i, row in importance.head(5).iterrows():
                mlflow.log_param(f"top_feature_{list(importance.index).index(i)+1}",
                                 row["feature"])

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

    print(f"\n{'='*65}")
    print(f"  XGBoost + LightGBM Forecasts — {ticker}")
    print(f"{'='*65}")
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
    print(f"\n{'='*75}")
    print(f"  All Models Forecast Comparison — {ticker}")
    print(f"{'='*75}")
    print(pivot.to_string())
    print()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(tickers: list[str] = None) -> dict:
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

        logger.info(f"  Feature matrix: {len(df)} rows × {len(get_feature_cols())} features")

        ticker_results = {}

        # XGBoost
        xgb_result = train_xgboost(df, ticker)
        if xgb_result:
            xgb_forecast = generate_forecast(xgb_result, "xgboost", ticker)
            n = save_forecasts(xgb_forecast)
            log_mlflow(ticker, "xgboost", xgb_result["metrics"],
                      xgb_result["importance"], xgb_forecast)
            ticker_results["xgboost"] = n
            logger.info(f"  XGBoost: {n} forecast rows saved")

            # Print top 5 features
            top5 = xgb_result["importance"].head(5)
            logger.info(f"  Top features: {', '.join(top5['feature'].tolist())}")

        # LightGBM
        lgb_result = train_lightgbm(df, ticker)
        if lgb_result:
            lgb_forecast = generate_forecast(lgb_result, "lightgbm", ticker)
            n = save_forecasts(lgb_forecast)
            log_mlflow(ticker, "lightgbm", lgb_result["metrics"],
                      lgb_result["importance"], lgb_forecast)
            ticker_results["lightgbm"] = n
            logger.info(f"  LightGBM: {n} forecast rows saved")

        results[ticker] = ticker_results

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost + LightGBM forecasting")
    parser.add_argument("--ticker",  type=str, help="Single ticker (e.g. AAPL)")
    parser.add_argument("--show",    type=str, help="Show forecasts for ticker")
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