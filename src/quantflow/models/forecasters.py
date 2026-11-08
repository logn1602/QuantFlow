"""Each model as a forecaster for the rolling-origin evaluation.

An adapter takes what the harness allows at one origin (evaluation's
OriginData: the prices and articles up to the cut, and the dates to
forecast) and calls exactly what the live run calls on the full history:
the same forecast functions, the same feature builder, the live boosters
fitted on every labelled row and rolled forward recursively. So the number
the evaluation reports describes the forecast the dashboard publishes.

They live in models because evaluation never imports models; the harness
takes them as functions.
"""

import numpy as np
import pandas as pd

from quantflow.evaluation.rolling_origin import (
    BASELINES,
    Combiner,
    Forecaster,
    OriginData,
)
from quantflow.features.engineering import (
    build_features,
    engineer_features,
    get_feature_cols,
)
from quantflow.models.boosting import fit_live_model, generate_forecast
from quantflow.models.ensemble import (
    BASE_MODELS,
    _fit_nnls,
    build_holdout_stack,
    combine_forecasts,
)
from quantflow.models.statistical import (
    arima_forecast,
    prophet_forecast,
    prophet_settings,
)

FORECAST_COLUMNS = ["yhat", "yhat_lower", "yhat_upper"]

# The news features the ablation leaves out (docs/evaluation.md, section 4).
NEWS_FEATURES = [
    "sentiment_compound",
    "sentiment_3d",
    "sentiment_change",
    "pos_count",
    "neg_count",
    "article_count",
    "has_news",
]


def _ds_y(data: OriginData) -> pd.DataFrame:
    """The prices as the statistical models read them (load_daily_close)."""
    return pd.DataFrame({"ds": data.prices["date"], "y": data.prices["close"]})


def arima(data: OriginData) -> pd.DataFrame:
    return arima_forecast(data.prices["close"], len(data.dates))


def prophet(data: OriginData) -> pd.DataFrame:
    history = _ds_y(data)
    forecast = prophet_forecast(history, data.dates)[FORECAST_COLUMNS]
    return forecast.reset_index(drop=True).assign(
        yearly_seasonality=prophet_settings(history)["yearly_seasonality"]
    )


def booster(model_name: str, without: tuple[str, ...] = ()) -> Forecaster:
    """XGBoost or LightGBM as boosting.run forecasts live: the live model
    fitted on every labelled row, rolled forward from the newest one.
    without names features to leave out, for an ablation."""

    def forecast(data: OriginData) -> pd.DataFrame:
        raw = build_features(
            data.prices, data.articles, data.ticker, fetched_at=data.fetched_at
        )
        features = engineer_features(raw, drop_unlabeled=False)
        labelled = features.dropna(subset=["target_price"])
        columns = [
            c for c in get_feature_cols() if c in labelled.columns and c not in without
        ]
        model = fit_live_model(model_name, labelled, columns)
        if model is None:
            raise RuntimeError(f"{model_name} is not installed")
        rolled = generate_forecast(
            model,
            features.iloc[[-1]][columns],
            features["date"].iloc[-1],
            list(features["close"].to_numpy()),
            model_name,
            data.ticker,
        )
        return pd.DataFrame(
            {
                "yhat": rolled["predicted_close"].to_numpy(),
                "yhat_lower": rolled["lower_bound"].to_numpy(),
                "yhat_upper": rolled["upper_bound"].to_numpy(),
            }
        )

    return forecast


def ensemble_stack(
    data: OriginData, base: dict[str, pd.DataFrame], earlier: pd.DataFrame
) -> pd.DataFrame:
    """The live ensemble as it stands: NNLS weights fitted on the 30-day
    holdout stack of the history up to the origin, applied to the four base
    forecasts made at the origin."""
    missing = [m for m in BASE_MODELS if m not in base]
    if missing:
        raise RuntimeError(f"no {', '.join(missing)} forecast to combine")
    raw = build_features(
        data.prices, data.articles, data.ticker, fetched_at=data.fetched_at
    )
    stack = build_holdout_stack(_ds_y(data), raw, data.ticker)
    if stack.empty:
        raise RuntimeError("the holdout stack could not be built")
    meta = _fit_nnls(stack[BASE_MODELS].to_numpy(), stack["actual"].to_numpy())

    def column(name: str) -> np.ndarray:
        # The live forecast fills a missing bound with 0 before weighting.
        return np.column_stack([base[m][name].fillna(0) for m in BASE_MODELS])

    yhat, lower, upper = combine_forecasts(
        meta, column("yhat"), column("yhat_lower"), column("yhat_upper")
    )
    return pd.DataFrame({"yhat": yhat, "yhat_lower": lower, "yhat_upper": upper})


FORECASTERS: dict[str, Forecaster] = {
    **BASELINES,
    "arima": arima,
    "prophet": prophet,
    "xgboost": booster("xgboost"),
    "lightgbm": booster("lightgbm"),
}
COMBINERS: dict[str, Combiner] = {"ensemble_stack": ensemble_stack}

# Run only when named in --models: the news ablation's pair (R6).
VARIANTS: dict[str, Forecaster] = {
    "xgboost_no_news": booster("xgboost", without=tuple(NEWS_FEATURES)),
    "lightgbm_no_news": booster("lightgbm", without=tuple(NEWS_FEATURES)),
}
