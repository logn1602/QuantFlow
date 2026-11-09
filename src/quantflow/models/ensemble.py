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
import warnings

import numpy as np
import pandas as pd
from sqlalchemy import text

from quantflow.config import TICKERS
from quantflow.db.connection import get_engine
from quantflow.db.forecasts import ForecastSet, load_base_forecasts, save_forecasts
from quantflow.db.metrics import save_model_metrics
from quantflow.db.prices import load_daily_close as load_prices
from quantflow.evaluation.metrics import mape as _mape
from quantflow.evaluation.out_of_fold import (
    DEFAULT_WARMUP_DAYS,
    expanding_window_predictions,
)
from quantflow.evaluation.splits import HOLDOUT_DAYS, train_holdout_split
from quantflow.features.engineering import load_features, load_news
from quantflow.models import FORECAST_DAYS
from quantflow.models.statistical import arima_holdout, prophet_holdout
from quantflow.utils.logger import get_logger
from quantflow.utils.trading_calendar import next_trading_days

warnings.filterwarnings("ignore")

logger = get_logger("ensemble")

META_WARMUP_DAYS = DEFAULT_WARMUP_DAYS

# Column order of the stacked holdout frame. The meta-learner's weight
# vector is positional, so this order is load-bearing.
BASE_MODELS = ["arima", "prophet", "xgboost", "lightgbm"]
# The columns each ensemble weighs (Phase 3, E6, decided per ensemble): the
# base models' forecasts and naive's, the latest close, so the weights can
# lean toward "no change". Positional, like BASE_MODELS. STACK_COLUMNS is the
# live 30-day stack's (ensemble_stack), ROLLING_COLUMNS the rolling-origin
# weights' (ensemble_rolling, evaluated only).
NAIVE = "naive"
STACK_COLUMNS: list[str] = [*BASE_MODELS, NAIVE]
ROLLING_COLUMNS: list[str] = [*BASE_MODELS, NAIVE]
MLFLOW_EXP = "stock_forecasting"


# ── Holdout prediction helpers ────────────────────────────────────────────────


def _xgb_lgb_holdout(raw_features: pd.DataFrame, ticker: str) -> dict:
    """Train XGBoost & LightGBM on train split, return holdout predictions.

    raw_features is load_features' merged frame for the ticker.
    """
    from quantflow.features.engineering import engineer_features
    from quantflow.models.boosting import train_lightgbm, train_xgboost

    if raw_features.empty:
        return {}
    df = engineer_features(raw_features)
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


def collect_holdout_stacks(
    ticker: str, news: pd.DataFrame | None = None
) -> pd.DataFrame:
    """The holdout stack for a ticker, from the stored prices and features.
    news is load_news' frame, read once by the caller for every ticker (read
    here if not given)."""
    return build_holdout_stack(load_prices(ticker), load_features(ticker, news), ticker)


def build_holdout_stack(
    price_df: pd.DataFrame, raw_features: pd.DataFrame, ticker: str
) -> pd.DataFrame:
    """
    Re-runs all 4 base models on their train/holdout split and aligns
    their predictions into a single DataFrame for meta-learner training.

    price_df is load_prices' ds/y series and raw_features load_features'
    merged frame. Taking them as arguments rather than loading them lets the
    same stack be built from any history, not only the stored one.

    All models predict the same 30 close prices (the last 30 in the series):
    - ARIMA: forecast(steps=30) from the end of the train split
    - Prophet: predict on the 30 holdout dates themselves, so its column
      lines up with actual across market holidays
    - XGBoost/LightGBM: predict(X_holdout) where target = close.shift(-1)

    - Naive: the last train close for every holdout day, the forecast naive
      would have made at the end of the train split, as ARIMA's and
      Prophet's columns are forecasts from there

    Returns DataFrame: [actual, arima, prophet, xgboost, lightgbm, naive]
    """
    if price_df.empty or len(price_df) < HOLDOUT_DAYS + 60:
        logger.warning(
            f"{ticker}: insufficient history for stacking "
            f"(have {len(price_df)}, need {HOLDOUT_DAYS + 60})"
        )
        return pd.DataFrame()

    train, holdout = train_holdout_split(price_df)
    actual = holdout["y"].to_numpy()

    logger.info("  [1/4] ARIMA holdout predictions...")
    arima_preds = arima_holdout(price_df)

    logger.info("  [2/4] Prophet holdout predictions...")
    prophet_preds = prophet_holdout(price_df)

    logger.info("  [3/4] XGBoost + LightGBM holdout predictions...")
    ml_preds = _xgb_lgb_holdout(raw_features, ticker)
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
            NAIVE: np.full(n, float(train["y"].iloc[-1])),
        }
    )


# ── Meta-learner ──────────────────────────────────────────────────────────────


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


def _fit_nnls(X, y, strict: bool = False) -> NNLSMeta:
    """
    Solve min ||Xw - y||^2 s.t. w >= 0, then normalise to a convex combination.

    Shared by the out-of-fold loop and the final production fit so both use
    one implementation of the meta-learner. An all-zero solution falls back
    to equal weights, unless strict, when it is an error (the rolling-origin
    weights, docs/evaluation.md section 5: a guess is not a fit).
    """
    from scipy.optimize import nnls

    if hasattr(X, "values"):
        X = X.values
    if hasattr(y, "values"):
        y = y.values

    n_features = X.shape[1]
    raw_weights, _ = nnls(X, y)

    weight_sum = raw_weights.sum()
    if strict and weight_sum <= 0:
        raise ValueError("the NNLS solution is all zero: no weights to use")
    weights = (
        raw_weights / weight_sum if weight_sum > 0 else np.ones(n_features) / n_features
    )
    return NNLSMeta(weights)


def fit_nnls_strict(X, y) -> NNLSMeta:
    """_fit_nnls with an all-zero solution an error: the rolling-origin
    weights' fitter."""
    return _fit_nnls(X, y, strict=True)


def out_of_fold_meta_predictions(
    stacked_df: pd.DataFrame, warmup: int = META_WARMUP_DAYS
) -> tuple:
    """Out-of-fold ensemble predictions over an expanding window.

    A thin binding of the generic loop in quantflow.evaluation.out_of_fold to
    this project's NNLS meta-learner. The loop itself, and the reasoning for
    why the ensemble must be scored this way, live there.

    Returns (oof_preds, actuals) — both length len(stacked_df) - warmup.
    """
    return expanding_window_predictions(
        stacked_df, STACK_COLUMNS, _fit_nnls, warmup=warmup
    )


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
    feature_cols = STACK_COLUMNS
    X = stacked_df[feature_cols].values
    y = stacked_df["actual"].values

    # ── Honest evaluation: out-of-fold predictions ────────────────────────────
    oof_preds, oof_actuals = out_of_fold_meta_predictions(stacked_df)
    eval_days = len(oof_actuals)
    warmup = len(stacked_df) - eval_days

    base_mapes: dict[str, float | None]
    best_base: float | None
    ens_mape: float | None
    improvement: float | None

    if eval_days > 0:
        # Base models MUST be scored on the same window as the OOF ensemble.
        # Computed into concrete floats first: the outer variables are optional
        # because the else-branch has nothing to report, but nothing in here is.
        scored = {
            m: round(_mape(oof_actuals, stacked_df[m].values[warmup:]), 2)
            for m in feature_cols
        }
        best = min(scored.values())
        ensemble_score = round(_mape(oof_actuals, oof_preds), 2)

        base_mapes = dict(scored)
        best_base = best
        ens_mape = ensemble_score
        improvement = (
            round((best - ensemble_score) / best * 100, 2) if best > 0 else 0.0
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
    if improvement is not None:
        verdict = "beats" if improvement > 0 else "does NOT beat"
        logger.info(
            f"  Out-of-fold over {eval_days} days: ensemble {ens_mape}% "
            f"{verdict} best base {best_base}% ({improvement:+.2f}%)"
        )

    metrics = {
        "ensemble_mape": ens_mape,
        **{f"{m}_mape": base_mapes[m] for m in feature_cols},
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


class ForecastAlignmentError(ValueError):
    """The stored base forecasts do not cover the same forecast dates."""


def check_forecast_dates(df: pd.DataFrame, models: list[str], horizon: int) -> list:
    """The forecast dates every base model shares, or ForecastAlignmentError.

    Each model must cover exactly the same `horizon` dates. The ensemble used
    to pivot by date and drop any date one model lacked, so a base model
    dated one day off silently cost the published forecast a day. That only
    happens when an upstream run is broken, so it must stop the run rather
    than be smoothed over.

    It also refuses more than one row per model and date. Two predictions for
    one date cannot be combined into one ensemble row, and the pivot used to
    fail on them with a pandas error that stopped the remaining tickers.
    """
    dates = {m: sorted(set(df.loc[df["model"] == m, "forecast_date"])) for m in models}
    missing = [m for m, d in dates.items() if not d]
    if missing:
        raise ForecastAlignmentError(f"no forecasts stored for {', '.join(missing)}")

    base = df[df["model"].isin(models)]
    repeated = base[base.duplicated(["model", "forecast_date"])]
    if not repeated.empty:
        twice = sorted(set(repeated["model"]))
        raise ForecastAlignmentError(
            f"more than one stored forecast per date for {', '.join(twice)}"
        )

    reference = dates[models[0]]
    if len(reference) != horizon or any(d != reference for d in dates.values()):
        detail = "; ".join(
            f"{m}: {len(d)} dates, {d[0]} to {d[-1]}" for m, d in dates.items()
        )
        raise ForecastAlignmentError(
            f"base forecasts do not cover the same {horizon} dates ({detail})"
        )
    return reference


def combine_forecasts(meta_model, predicted, lower, upper) -> tuple:
    """The ensemble's forecast and band from base forecasts, one column per
    model in the meta-learner's column order: its prediction, and each
    bound as the base bounds weighted by |coef| / sum(|coef|). Shared by the
    live forecast and the evaluation, so both combine the same way."""
    abs_coefs = np.abs(meta_model.coef_)
    weights = abs_coefs / abs_coefs.sum()
    return (
        meta_model.predict(np.asarray(predicted, dtype=float)),
        (np.asarray(lower, dtype=float) * weights).sum(axis=1),
        (np.asarray(upper, dtype=float) * weights).sum(axis=1),
    )


def generate_ensemble_forecast(
    ticker: str,
    meta_model,
    latest_close: float | None = None,
    latest_date=None,
    forecasts: ForecastSet | None = None,
) -> pd.DataFrame:
    """
    Load 7-day base model forecasts from DB and apply meta-learner to produce
    ensemble predictions. Confidence intervals are propagated as a weighted
    average of base model bounds, using the NNLS weights. Those weights are
    already non-negative and sum to 1, so normalising |w| is a no-op here;
    it is kept because it makes the CI propagation independent of how the
    meta-learner was fitted.

    Raises ForecastAlignmentError unless all four base models cover the same
    seven dates. With naive among STACK_COLUMNS, naive's forecast is
    latest_close for every date, its band that same price (no width), and
    the base forecasts must start the trading day after latest_date, the
    close's own date: a failed base step leaves the previous run's forecasts
    in the table, which would otherwise be published beside today's close.

    The base forecasts come from forecasts, the set the daily run holds them
    in until it is done (run_models), or else from the table.
    """
    if forecasts is not None:
        df = forecasts.base(ticker, BASE_MODELS)
    else:
        df = load_base_forecasts(ticker, BASE_MODELS)
    try:
        dates = check_forecast_dates(df, BASE_MODELS, FORECAST_DAYS)
    except ForecastAlignmentError as e:
        raise ForecastAlignmentError(f"{ticker}: {e}") from e

    # One row per model and date is guaranteed by the check, so each pivot is
    # complete: dates down, the base models across in the meta-learner's order.
    naive_close = None
    if NAIVE in STACK_COLUMNS:
        if latest_close is None or latest_date is None:
            raise ValueError(
                f"{ticker}: the ensemble weighs naive, the latest close and its date"
            )
        first = next_trading_days(pd.Timestamp(latest_date), 1)[0]
        if pd.Timestamp(dates[0]).normalize() != first:
            raise ForecastAlignmentError(
                f"{ticker}: the base forecasts start {pd.Timestamp(dates[0]).date()}, "
                f"not {first.date()}, the trading day after the latest close"
            )
        naive_close = float(latest_close)

    def wide(column: str) -> pd.DataFrame:
        table = df.pivot(index="forecast_date", columns="model", values=column)
        table = table.reindex(index=dates, columns=BASE_MODELS)
        if column != "predicted_close":
            table = table.fillna(0)
        if naive_close is not None:
            table[NAIVE] = naive_close
        return table[STACK_COLUMNS]

    pivot = wide("predicted_close")
    lower = wide("lower_bound")
    upper = wide("upper_bound")

    ensemble_preds, ens_lower, ens_upper = combine_forecasts(
        meta_model, pivot.values, lower.values, upper.values
    )

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
            mlflow.log_param("base_models", ",".join(STACK_COLUMNS))
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
                *(f"{m}_mape" for m in STACK_COLUMNS),
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


def run(
    tickers: list | None = None,
    news: pd.DataFrame | None = None,
    forecasts: ForecastSet | None = None,
) -> dict:
    """news is load_news' frame, read here once for every ticker unless the
    caller has read it. forecasts is the set run_models holds every forecast
    in until the run is done: the ensemble reads the base forecasts from it
    and adds its own; without one, it reads the table and stores its own."""
    tickers = tickers or TICKERS
    save = forecasts.save if forecasts is not None else save_forecasts
    results = {}
    misaligned: list[str] = []
    if news is None:
        news = load_news()

    for ticker in tickers:
        logger.info(f"{'=' * 50}")
        logger.info(f"Stacking Ensemble — {ticker}")
        logger.info(f"{'=' * 50}")

        logger.info("Collecting out-of-fold holdout predictions...")
        # Loaded here rather than through collect_holdout_stacks: the same
        # reads, and the latest close is naive's forecast (STACK_COLUMNS).
        prices = load_prices(ticker)
        stacked_df = build_holdout_stack(prices, load_features(ticker, news), ticker)
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
            "  Weights: " + " | ".join(f"{m}: {w:.3f}" for m, w in coefs.items())
        )
        if metrics["eval_days"] > 0:
            base_best = min(metrics[f"{m}_mape"] for m in STACK_COLUMNS)
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
        try:
            forecast_df = generate_ensemble_forecast(
                ticker,
                meta_model,
                float(prices["y"].iloc[-1]),
                prices["ds"].iloc[-1],
                forecasts=forecasts,
            )
        except ForecastAlignmentError as e:
            # Carried to the end of the loop: one ticker's broken base run
            # should not cost the others their forecast, but the run must
            # still fail rather than publish a short or shifted forecast.
            logger.error(f"{e}; no ensemble forecast written")
            misaligned.append(ticker)
            results[ticker] = 0
            continue

        n = save(forecast_df)
        logger.info(f"  {n} ensemble forecast rows")

        log_to_mlflow(ticker, meta_model, metrics, stacked_df, forecast_df)

        results[ticker] = n

    if misaligned:
        raise ForecastAlignmentError(
            f"no ensemble forecast for {', '.join(misaligned)}: "
            "base forecasts do not cover the same dates (see the log above)"
        )
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
