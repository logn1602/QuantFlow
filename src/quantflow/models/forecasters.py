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

from quantflow.evaluation.out_of_fold import rolling_origin_fit, rolling_origin_stack
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
    NAIVE,
    ROLLING_COLUMNS,
    STACK_COLUMNS,
    _fit_nnls,
    build_holdout_stack,
    combine_forecasts,
    fit_nnls_strict,
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


class Booster:
    """XGBoost or LightGBM as boosting.run forecasts live: the live model
    fitted on every labelled row, rolled forward from the newest one.
    without names features to leave out, for an ablation; params replace
    some of boosting.MODEL_PARAMS, for tuning.

    A class rather than a closure, so the harness's worker processes can
    receive it."""

    def __init__(
        self,
        model_name: str,
        without: tuple[str, ...] = (),
        params: dict | None = None,
    ):
        self.model_name = model_name
        self.without = tuple(without)
        self.params = dict(params or {})

    def __call__(self, data: OriginData) -> pd.DataFrame:
        raw = build_features(
            data.prices, data.articles, data.ticker, fetched_at=data.fetched_at
        )
        features = engineer_features(raw, drop_unlabeled=False)
        labelled = features.dropna(subset=["target_price"])
        columns = [
            c
            for c in get_feature_cols()
            if c in labelled.columns and c not in self.without
        ]
        model = fit_live_model(self.model_name, labelled, columns, self.params)
        if model is None:
            raise RuntimeError(f"{self.model_name} is not installed")
        rolled = generate_forecast(
            model,
            features.iloc[[-1]][columns],
            features["date"].iloc[-1],
            list(features["close"].to_numpy()),
            self.model_name,
            data.ticker,
        )
        return pd.DataFrame(
            {
                "yhat": rolled["predicted_close"].to_numpy(),
                "yhat_lower": rolled["lower_bound"].to_numpy(),
                "yhat_upper": rolled["upper_bound"].to_numpy(),
            }
        )


def booster(model_name: str, without: tuple[str, ...] = ()) -> Forecaster:
    return Booster(model_name, without)


def ensemble_stack(
    data: OriginData, base: dict[str, pd.DataFrame], earlier: pd.DataFrame
) -> pd.DataFrame:
    """The live ensemble as it stands: NNLS weights fitted on the 30-day
    holdout stack of the history up to the origin, applied to the forecasts
    made at the origin, in STACK_COLUMNS."""
    missing = [m for m in STACK_COLUMNS if m not in base]
    if missing:
        raise RuntimeError(f"no {', '.join(missing)} forecast to combine")
    raw = build_features(
        data.prices, data.articles, data.ticker, fetched_at=data.fetched_at
    )
    stack = build_holdout_stack(_ds_y(data), raw, data.ticker)
    if stack.empty:
        raise RuntimeError("the holdout stack could not be built")
    meta = _fit_nnls(stack[STACK_COLUMNS].to_numpy(), stack["actual"].to_numpy())
    return _combined(meta, base, STACK_COLUMNS)


# Separate rolling-origin weights for each horizon (Phase 3, E7): day 1 and
# day 7 are different tasks, and one set of weights serves neither best. Each
# day's weights are fitted on that day's rows of the earlier origins only
# (11 or 12 rows, against 82 for one set).
ROLLING_PER_HORIZON = True


def ensemble_rolling(
    data: OriginData, base: dict[str, pd.DataFrame], earlier: pd.DataFrame
) -> pd.DataFrame:
    """The ensemble with weights learned on the task it is used for (2c):
    NNLS fitted on the 7-day forecasts in ROLLING_COLUMNS at the 12 grid
    origins before this one, only the rows whose target is known by it,
    applied to those forecasts made here. Fewer than 8 complete earlier
    origins (ThinWindowError) or an all-zero solution fails the cells."""
    missing = [m for m in ROLLING_COLUMNS if m not in base]
    if missing:
        raise RuntimeError(f"no {', '.join(missing)} forecast to combine")
    if not ROLLING_PER_HORIZON:
        meta = rolling_origin_fit(
            earlier, data.origin, ROLLING_COLUMNS, fit_nnls_strict
        )
        return _combined(meta, base, ROLLING_COLUMNS)
    stack = rolling_origin_stack(earlier, data.origin, ROLLING_COLUMNS)
    days = []
    for h in range(1, len(data.dates) + 1):
        rows = stack[stack["h"] == h]
        meta = fit_nnls_strict(
            rows[ROLLING_COLUMNS].to_numpy(dtype=float),
            rows["actual"].to_numpy(dtype=float),
        )
        day = {m: base[m].iloc[[h - 1]].reset_index(drop=True) for m in ROLLING_COLUMNS}
        days.append(_combined(meta, day, ROLLING_COLUMNS))
    return pd.concat(days, ignore_index=True)


# The harness adds the warmup origins before the window when a combiner needs
# earlier origins' cells.
ensemble_rolling.needs_history = True  # type: ignore[attr-defined]


def _combined(meta, base: dict[str, pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    def column(name: str) -> np.ndarray:
        # The live forecast fills a missing bound with 0 before weighting;
        # naive has no band, so its bounds are its forecast, as live.
        return np.column_stack(
            [base[m]["yhat" if m == NAIVE else name].fillna(0) for m in columns]
        )

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
COMBINERS: dict[str, Combiner] = {
    "ensemble_stack": ensemble_stack,
    "ensemble_rolling": ensemble_rolling,
}

# Run only when named in --models: the news ablation's pair (R6).
VARIANTS: dict[str, Forecaster] = {
    "xgboost_no_news": booster("xgboost", without=tuple(NEWS_FEATURES)),
    "lightgbm_no_news": booster("lightgbm", without=tuple(NEWS_FEATURES)),
}
