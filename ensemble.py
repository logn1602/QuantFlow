"""
ensemble.py
-----------
Level 3 — Stacking Ensemble

Trains a non-negative least squares (NNLS) meta-learner on out-of-fold holdout
predictions from ARIMA, Prophet, XGBoost, and LightGBM. The meta-learner learns
data-driven combination weights per ticker.

NNLS replaced Ridge because Ridge can assign negative coefficients to base
models and lean on the intercept to compensate — fine in-sample, but it
extrapolates badly once live predictions drift outside the training price
range. NNLS constrains weights to w_i >= 0 summing to 1, so the ensemble is
always a convex combination and can never predict outside the range of its
base models. See the NNLSMeta docstring for the full rationale.

Whether the ensemble beats the best single base model is measured, not
assumed — see out_of_fold_meta_predictions. It does not always win.

Architecture:
  Layer 1 (base models):  ARIMA · Prophet · XGBoost · LightGBM
         ↓  30-day out-of-fold holdout predictions
  Layer 2 (meta-learner): NNLS convex combination, evaluated with an
                          expanding-window out-of-fold loop
         ↓  learned non-negative weights summing to 1
  Output: 7-day stacked ensemble forecast

MLflow tracks:
  - weight_* params (the learned per-model weights)
  - oof_mape — honest out-of-fold ensemble MAPE
  - eval_days — length of the out-of-fold evaluation window
  - Per-base-model MAPE over that same window
  - improvement_pct over the best base model (negative when the ensemble loses)
  - Full forecast + holdout stack artifacts

Usage:
    python ensemble.py                  # run all tickers
    python ensemble.py --ticker AAPL    # run single ticker
    python ensemble.py --show AAPL      # print ensemble forecasts
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

# The imports below sit after the sys.path bootstrap above and therefore trip
# E402. Each carries a targeted suppression rather than a config-wide ignore.
# The src/ layout removes the bootstrap, and these all come out with it.
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402

from db.connection import get_engine  # noqa: E402
from db.metrics import save_model_metrics  # noqa: E402
from quantflow.config import TICKERS  # noqa: E402
from quantflow.utils.logger import get_logger  # noqa: E402

logger = get_logger("ensemble")

HOLDOUT_DAYS = 30
FORECAST_DAYS = 7
META_WARMUP_DAYS = 10  # days used to fit the first meta-learner
MLFLOW_EXP = "stock_forecasting"


# ── Holdout prediction helpers ────────────────────────────────────────────────


def _arima_holdout(price_df: pd.DataFrame) -> np.ndarray:
    """Fit ARIMA on train split, return 30-day holdout predictions."""
    from statsmodels.tsa.arima.model import ARIMA

    train = price_df.iloc[:-HOLDOUT_DAYS]
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
    future = model.make_future_dataframe(periods=HOLDOUT_DAYS, freq="B")
    forecast = model.predict(future)
    return forecast["yhat"].iloc[-HOLDOUT_DAYS:].values


def _xgb_lgb_holdout(ticker: str) -> dict:
    """Train XGBoost & LightGBM on train split, return holdout predictions."""
    from xgboost_model import (
        engineer_features,
        load_features,
        train_lightgbm,
        train_xgboost,
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
        "xgboost": xgb_result.get("holdout_preds", np.array([])),
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

    logger.info("  [1/4] ARIMA holdout predictions...")
    arima_preds = _arima_holdout(price_df)

    logger.info("  [2/4] Prophet holdout predictions...")
    prophet_preds = _prophet_holdout(price_df)

    logger.info("  [3/4] XGBoost + LightGBM holdout predictions...")
    ml_preds = _xgb_lgb_holdout(ticker)
    if not ml_preds or len(ml_preds.get("xgboost", [])) == 0:
        logger.error(f"{ticker}: XGB/LGB holdout failed — cannot stack")
        return pd.DataFrame()

    xgb_preds = ml_preds["xgboost"]
    lgb_preds = ml_preds["lightgbm"]

    n = min(
        len(actual),
        len(arima_preds),
        len(prophet_preds),
        len(xgb_preds),
        len(lgb_preds),
    )

    return pd.DataFrame(
        {
            "actual": actual[:n],
            "arima": np.asarray(arima_preds)[:n],
            "prophet": np.asarray(prophet_preds)[:n],
            "xgboost": np.asarray(xgb_preds)[:n],
            "lightgbm": np.asarray(lgb_preds)[:n],
        }
    )


# ── Meta-learner ──────────────────────────────────────────────────────────────


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


class NNLSMeta:
    """
    Non-negative least squares meta-learner.

    Replaces Ridge because Ridge can assign negative coefficients to base models,
    which lets the intercept compensate and causes wild extrapolation when
    production base-model predictions drift outside the training price range.

    NNLS constraints: w_i >= 0 and sum(w_i) == 1, so the ensemble is always
    a proper convex combination of the base models — it can never predict
    outside [min(base_preds), max(base_preds)] for a given day.
    """

    def __init__(self, weights: np.ndarray):
        self.coef_ = weights  # non-negative, sums to 1

    def predict(self, X) -> np.ndarray:
        if hasattr(X, "values"):
            X = X.values
        return X @ self.coef_


def _fit_nnls(X, y) -> NNLSMeta:
    """
    Solve min ||Xw - y||^2 s.t. w >= 0, then normalise to a convex combination.

    Shared by the out-of-fold loop and the final production fit so both use
    one implementation of the meta-learner.
    """
    from scipy.optimize import nnls

    if hasattr(X, "values"):
        X = X.values
    if hasattr(y, "values"):
        y = y.values

    n_features = X.shape[1]
    raw_weights, _ = nnls(X, y)

    weight_sum = raw_weights.sum()
    weights = (
        raw_weights / weight_sum if weight_sum > 0 else np.ones(n_features) / n_features
    )
    return NNLSMeta(weights)


def out_of_fold_meta_predictions(
    stacked_df: pd.DataFrame, warmup: int = META_WARMUP_DAYS
) -> tuple:
    """
    Expanding-window out-of-fold ensemble predictions.

    For each day i in [warmup, len(stacked_df)):
      fit NNLS on days [0, i)  ->  predict day i
    The prediction for day i never sees day i's actual, so the resulting
    MAPE is an honest estimate of ensemble performance.

    Returns (oof_preds, actuals) — both length len(stacked_df) - warmup.
    """
    feature_cols = ["arima", "prophet", "xgboost", "lightgbm"]
    X = stacked_df[feature_cols].values
    y = stacked_df["actual"].values

    if len(stacked_df) <= warmup:
        logger.warning(
            f"  Stack has {len(stacked_df)} rows, warmup is {warmup} — "
            f"no out-of-fold days available"
        )
        return np.array([]), np.array([])

    oof_preds = np.empty(len(stacked_df) - warmup, dtype=float)
    for j, i in enumerate(range(warmup, len(stacked_df))):
        meta = _fit_nnls(X[:i], y[:i])
        oof_preds[j] = float(meta.predict(X[i : i + 1])[0])

    return oof_preds, y[warmup:]


def tune_and_train_meta(stacked_df: pd.DataFrame) -> tuple:
    """
    Fit a non-negative least squares meta-learner on the 30-day holdout stack.
    Returns (NNLSMeta, metrics_dict).

    Why NNLS instead of Ridge:
      Ridge with an intercept can learn negative weights for base models and
      use the intercept to compensate. This works in-sample but extrapolates
      badly when current prices differ from the training range. NNLS forces
      all weights >= 0 and we normalise to sum=1, making the ensemble a safe
      convex combination that cannot diverge from the base models.

    Reported metrics are out-of-fold: each evaluation day is predicted by a
    meta-learner fitted only on days before it. Base-model MAPEs are scored
    over that same window so the comparison is like-for-like, and
    improvement_pct is free to go negative when the ensemble genuinely loses.

    The returned model is fitted on ALL holdout days — that is the model used
    to generate the forward 7-day forecast.
    """
    feature_cols = ["arima", "prophet", "xgboost", "lightgbm"]
    X = stacked_df[feature_cols].values
    y = stacked_df["actual"].values

    # ── Honest evaluation: out-of-fold predictions ────────────────────────────
    oof_preds, oof_actuals = out_of_fold_meta_predictions(stacked_df)
    eval_days = len(oof_actuals)
    warmup = len(stacked_df) - eval_days

    if eval_days > 0:
        # Base models MUST be scored on the same window as the OOF ensemble.
        base_mapes = {
            m: round(_mape(oof_actuals, stacked_df[m].values[warmup:]), 2)
            for m in feature_cols
        }
        best_base = min(base_mapes.values())
        ens_mape = round(_mape(oof_actuals, oof_preds), 2)
        improvement = (
            round((best_base - ens_mape) / best_base * 100, 2) if best_base > 0 else 0.0
        )
    else:
        base_mapes = dict.fromkeys(feature_cols)
        best_base = None
        ens_mape = None
        improvement = None

    # ── Production fit: all holdout days, used for the forward forecast ───────
    meta = _fit_nnls(X, y)
    weights = meta.coef_

    logger.info(
        "  NNLS weights — "
        + " | ".join(
            f"{m}: {w:.3f}" for m, w in zip(feature_cols, weights, strict=False)
        )
    )
    if eval_days > 0:
        verdict = "beats" if improvement > 0 else "does NOT beat"
        logger.info(
            f"  Out-of-fold over {eval_days} days: ensemble {ens_mape}% "
            f"{verdict} best base {best_base}% ({improvement:+.2f}%)"
        )

    metrics = {
        "ensemble_mape": ens_mape,
        "arima_mape": base_mapes["arima"],
        "prophet_mape": base_mapes["prophet"],
        "xgboost_mape": base_mapes["xgboost"],
        "lightgbm_mape": base_mapes["lightgbm"],
        "meta_learner": "nnls_convex",
        "oof_mape": ens_mape,
        "eval_days": eval_days,
        "warmup_days": warmup,
        "coefficients": dict(zip(feature_cols, weights.tolist(), strict=False)),
        "intercept": 0.0,
        "improvement_pct": improvement,
    }

    return meta, metrics


# ── Forecast generation ───────────────────────────────────────────────────────


def generate_ensemble_forecast(ticker: str, meta_model) -> pd.DataFrame:
    """
    Load 7-day base model forecasts from DB and apply meta-learner to produce
    ensemble predictions. Confidence intervals are propagated as a weighted
    average of base model bounds, using the NNLS weights. Those weights are
    already non-negative and sum to 1, so normalising |w| is a no-op here —
    it is kept because it makes the CI propagation independent of how the
    meta-learner was fitted.
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

    X_future = pivot[required].values
    ensemble_preds = meta_model.predict(X_future)

    # Weighted CI propagation using |coef| / sum(|coef|)
    abs_coefs = np.abs(meta_model.coef_)
    coef_weights = abs_coefs / abs_coefs.sum()
    ens_lower = (lower[required].values * coef_weights).sum(axis=1)
    ens_upper = (upper[required].values * coef_weights).sum(axis=1)

    return pd.DataFrame(
        {
            "ticker": ticker,
            "model": "ensemble_stack",
            "ds": pivot.index,
            "predicted_close": ensemble_preds,
            "lower_bound": ens_lower,
            "upper_bound": ens_upper,
        }
    )


# ── Database write ────────────────────────────────────────────────────────────


def save_ensemble_forecasts(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    engine = get_engine()
    inserted = 0
    run_at = datetime.now()

    with engine.begin() as conn:
        for _, row in df.iterrows():
            try:
                ds = row["ds"]
                if hasattr(ds, "date") and not isinstance(ds, str):
                    ds = ds.date()
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
                        "ticker": row["ticker"],
                        "model": row["model"],
                        "forecast_date": ds,
                        "predicted_close": round(float(row["predicted_close"]), 4),
                        "lower_bound": round(float(row["lower_bound"]), 4),
                        "upper_bound": round(float(row["upper_bound"]), 4),
                        "run_at": run_at,
                    },
                )
                inserted += 1
            except Exception as e:
                logger.warning(f"Row skipped: {e}")

    return inserted


# ── MLflow logging ────────────────────────────────────────────────────────────


def log_to_mlflow(
    ticker: str,
    meta_model,
    metrics: dict,
    stacked_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
):
    try:
        import mlflow

        mlflow.set_experiment(MLFLOW_EXP)

        with mlflow.start_run(run_name=f"ensemble_stack_{ticker}"):
            mlflow.log_param("ticker", ticker)
            mlflow.log_param("model", "ensemble_stack")
            mlflow.log_param("meta_learner", metrics["meta_learner"])
            mlflow.log_param("base_models", "arima,prophet,xgboost,lightgbm")
            mlflow.log_param("holdout_days", HOLDOUT_DAYS)
            mlflow.log_param("warmup_days", metrics["warmup_days"])
            mlflow.log_param("forecast_days", FORECAST_DAYS)

            # Learned model weights — key insight for recruiters reviewing MLflow
            for model_name, coef in metrics["coefficients"].items():
                mlflow.log_param(f"weight_{model_name}", round(coef, 4))

            # eval_days is always meaningful; the MAPEs are None if the stack
            # was too short to leave any out-of-fold days.
            mlflow.log_metric("eval_days", metrics["eval_days"])
            for key in (
                "ensemble_mape",
                "oof_mape",
                "arima_mape",
                "prophet_mape",
                "xgboost_mape",
                "lightgbm_mape",
                "improvement_pct",
            ):
                if metrics[key] is not None:
                    mlflow.log_metric(key, metrics[key])

            os.makedirs("mlruns_artifacts", exist_ok=True)
            stack_path = f"mlruns_artifacts/ensemble_stack_{ticker}_holdout.csv"
            forecast_path = f"mlruns_artifacts/ensemble_stack_{ticker}_forecast.csv"
            stacked_df.to_csv(stack_path, index=False)
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

    print(f"\n{'=' * 60}")
    print(f"  Stacking Ensemble Forecast — {ticker}")
    print(f"{'=' * 60}")
    print(df.to_string(index=False))


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run(tickers: list | None = None) -> dict:
    tickers = tickers or TICKERS
    results = {}

    for ticker in tickers:
        logger.info(f"{'=' * 50}")
        logger.info(f"Stacking Ensemble — {ticker}")
        logger.info(f"{'=' * 50}")

        logger.info("Collecting out-of-fold holdout predictions...")
        stacked_df = collect_holdout_stacks(ticker)
        if stacked_df.empty:
            logger.error(f"{ticker}: stacking skipped")
            results[ticker] = 0
            continue

        logger.info(
            f"  Stack shape: {stacked_df.shape[0]} rows × {stacked_df.shape[1]} columns"  # noqa: RUF001
        )

        logger.info("Training NNLS meta-learner (non-negative convex combination)...")
        meta_model, metrics = tune_and_train_meta(stacked_df)

        coefs = metrics["coefficients"]
        logger.info(
            f"  Weights — ARIMA: {coefs['arima']:.3f} | Prophet: {coefs['prophet']:.3f} | "
            f"XGBoost: {coefs['xgboost']:.3f} | LightGBM: {coefs['lightgbm']:.3f}"
        )
        if metrics["eval_days"] > 0:
            base_best = min(
                metrics["arima_mape"],
                metrics["prophet_mape"],
                metrics["xgboost_mape"],
                metrics["lightgbm_mape"],
            )
            logger.info(
                f"  MAPE (out-of-fold, {metrics['eval_days']} days) — "
                f"Ensemble: {metrics['ensemble_mape']}% vs "
                f"best base: {base_best}% "
                f"({metrics['improvement_pct']:+.2f}% vs best base)"
            )
        else:
            logger.warning(f"  {ticker}: no out-of-fold days — MAPE not measured")

        # Store the out-of-fold MAPE against the evaluation window it was
        # measured on, not the full 30-day holdout.
        save_model_metrics(
            ticker,
            "ensemble_stack",
            {"mape": metrics["ensemble_mape"]},
            metrics["eval_days"],
        )

        logger.info("Generating 7-day ensemble forecast...")
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
    parser.add_argument("--show", type=str, help="Show ensemble forecasts for a ticker")
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
