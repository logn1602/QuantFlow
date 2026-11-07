"""The rolling-origin loop (docs/evaluation.md, section 2).

At each origin the loop cuts the data, so a forecaster sees only what
existed after that day's close, asks every forecaster for the next HORIZON
closes, and records one cell per forecast day:

    for each ticker, for each origin o:
        prices    dated on or before o
        articles  stored by news_cutoff(o)
        targets   the next HORIZON dates in the data, which must be the
                  next HORIZON trading days (CalendarMismatchError if not)
            |
            +--> forecasters(data)                 -> HORIZON forecasts each
            +--> combiners(data, base, earlier)    -> the ensemble, from the
            |                                         base forecasts at o and
            |                                         earlier origins' cells
            v
        cells: ticker, window, role, model, origin, h, target_date,
               origin_close, actual, yhat, yhat_lower, yhat_upper, status

Forecasters are passed in as functions, as out_of_fold.py takes its fitter:
evaluation never imports models. A forecaster that raises, or returns a
non-finite forecast, gives failed cells, recorded and counted; one that
returns the wrong number of rows is a bug in its adapter, so the loop
raises.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantflow.evaluation.splits import HORIZON
from quantflow.utils.logger import get_logger
from quantflow.utils.trading_calendar import news_cutoff, next_trading_days

logger = get_logger(__name__)

FORECAST_COLUMNS = ["yhat", "yhat_lower", "yhat_upper"]
CELL_COLUMNS = [
    "ticker",
    "window",
    "role",
    "model",
    "origin",
    "h",
    "target_date",
    "origin_close",
    "actual",
    *FORECAST_COLUMNS,
    "status",
    "error",
    "seconds",
]
# A model whose share of failed cells exceeds this fails the run, rather
# than being compared on the cells it happened to manage.
MAX_FAILURE_SHARE = 0.10


class ForecasterContractError(RuntimeError):
    """A forecaster returned the wrong shape: a bug in its adapter."""


class CalendarMismatchError(RuntimeError):
    """The data's next dates are not the next trading days."""


class FailureRateError(RuntimeError):
    """A model failed on more than MAX_FAILURE_SHARE of its cells."""


@dataclass(frozen=True)
class OriginData:
    """What a forecaster may see at one origin."""

    ticker: str
    origin: pd.Timestamp
    prices: pd.DataFrame  # daily bars dated on or before origin
    articles: pd.DataFrame  # stored by news_cutoff(origin)
    dates: pd.DatetimeIndex  # the HORIZON dates to forecast

    @property
    def close(self) -> float:
        return float(self.prices["close"].iloc[-1])


Forecaster = Callable[[OriginData], pd.DataFrame]
Combiner = Callable[[OriginData, dict[str, pd.DataFrame], pd.DataFrame], pd.DataFrame]


# ── Baselines ─────────────────────────────────────────────────────────────────


def naive_forecast(data: OriginData) -> pd.DataFrame:
    """Every future close equals the origin's close."""
    return pd.DataFrame(
        {
            "yhat": np.full(len(data.dates), data.close),
            "yhat_lower": np.nan,
            "yhat_upper": np.nan,
        }
    )


def drift_forecast(data: OriginData) -> pd.DataFrame:
    """A random walk with drift: the origin's close grown by the mean daily
    log return of the history up to the origin, h times for day h. Beside
    naive, it shows whether a model does more than ride the trend."""
    closes = data.prices["close"].to_numpy(dtype=float)
    drift = float(np.mean(np.diff(np.log(closes))))
    steps = np.arange(1, len(data.dates) + 1)
    return pd.DataFrame(
        {
            "yhat": data.close * np.exp(drift * steps),
            "yhat_lower": np.nan,
            "yhat_upper": np.nan,
        }
    )


BASELINES: dict[str, Forecaster] = {"naive": naive_forecast, "drift": drift_forecast}


# ── The loop ──────────────────────────────────────────────────────────────────


def origin_data(
    ticker: str,
    prices: pd.DataFrame,
    articles: pd.DataFrame,
    position: int,
    horizon: int,
) -> tuple[OriginData, pd.DataFrame]:
    """The data cut at the origin in row `position`, and its target rows."""
    origin = pd.Timestamp(prices["date"].iloc[position])
    targets = prices.iloc[position + 1 : position + 1 + horizon]
    expected = next_trading_days(origin, horizon)
    found = pd.DatetimeIndex(targets["date"]).normalize()
    if len(found) != horizon or not (found == expected).all():
        raise CalendarMismatchError(
            f"{ticker}: after {origin.date()} the data has "
            f"{[d.date().isoformat() for d in found]}, the calendar "
            f"{[d.date().isoformat() for d in expected]}"
        )
    if articles.empty:
        cut_articles = articles
    else:
        stored = pd.to_datetime(articles["inserted_at"], utc=True)
        keep = articles["ticker"].eq(ticker) & (stored <= news_cutoff(origin))
        cut_articles = articles[keep].reset_index(drop=True)
    data = OriginData(
        ticker=ticker,
        origin=origin,
        prices=prices.iloc[: position + 1].reset_index(drop=True),
        articles=cut_articles,
        dates=expected,
    )
    return data, targets.reset_index(drop=True)


def _checked(name: str, forecast, horizon: int) -> tuple[pd.DataFrame, str]:
    """The forecast with every column present, and an error for a forecast
    that cannot be scored (empty when it can)."""
    if not isinstance(forecast, pd.DataFrame) or len(forecast) != horizon:
        got = len(forecast) if hasattr(forecast, "__len__") else type(forecast).__name__
        raise ForecasterContractError(
            f"{name} returned {got} rows; a forecaster returns exactly {horizon}"
        )
    if "yhat" not in forecast:
        raise ForecasterContractError(f"{name} returned no yhat column")
    out = forecast.reset_index(drop=True).copy()
    for column in FORECAST_COLUMNS:
        out[column] = pd.to_numeric(out.get(column, np.nan), errors="coerce")
    if not np.isfinite(out["yhat"].to_numpy(dtype=float)).all():
        return out, "non-finite forecast"
    return out, ""


def _cells(
    data: OriginData,
    targets: pd.DataFrame,
    model: str,
    forecast: pd.DataFrame | None,
    error: str,
    seconds: float,
) -> pd.DataFrame:
    h = np.arange(1, len(data.dates) + 1)
    cells = pd.DataFrame(
        {
            "ticker": data.ticker,
            "model": model,
            "origin": data.origin,
            "h": h,
            "target_date": data.dates,
            "origin_close": data.close,
            "actual": targets["close"].to_numpy(dtype=float),
            "status": "failed" if error else "ok",
            "error": error,
            "seconds": seconds,
        }
    )
    if forecast is None:
        for column in FORECAST_COLUMNS:
            cells[column] = np.nan
    else:
        # Extra columns a forecaster reports (Prophet's yearly flag, say)
        # travel with its cells.
        for column in forecast.columns:
            cells[column] = forecast[column].to_numpy()
        if error:
            cells["yhat"] = np.nan
    return cells


def _call(name: str, fn: Callable, args: tuple, data, targets, horizon) -> tuple:
    started = time.perf_counter()
    try:
        raw = fn(*args)
    except ForecasterContractError:
        raise
    except Exception as e:  # a model failure: recorded, not fatal
        error = f"{type(e).__name__}: {e}"[:300]
        logger.warning(f"{data.ticker} {data.origin.date()} {name}: {error}")
        cells = _cells(data, targets, name, None, error, time.perf_counter() - started)
        return None, cells
    forecast, error = _checked(name, raw, horizon)
    if error:
        logger.warning(f"{data.ticker} {data.origin.date()} {name}: {error}")
    cells = _cells(data, targets, name, forecast, error, time.perf_counter() - started)
    return (None if error else forecast), cells


def run_origins(
    prices: dict[str, pd.DataFrame],
    articles: pd.DataFrame,
    origins: Sequence[tuple[int, str]],
    forecasters: dict[str, Forecaster],
    combiners: dict[str, Combiner] | None = None,
    window: str = "validation",
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Every cell of every model at every origin.

    prices maps each ticker to its daily bars (date, open, high, low, close,
    volume) in date order; origins are (row position, role) pairs, role
    "scored" or "warmup"; articles are the stored news rows of every ticker.
    """
    combiners = combiners or {}
    frames: list[pd.DataFrame] = []
    for ticker, bars in prices.items():
        bars = bars.sort_values("date").reset_index(drop=True)
        ticker_cells: list[pd.DataFrame] = []
        started = time.perf_counter()
        for position, role in sorted(origins):
            data, targets = origin_data(ticker, bars, articles, position, horizon)
            base: dict[str, pd.DataFrame] = {}
            at_origin: list[pd.DataFrame] = []
            for name, forecaster in forecasters.items():
                forecast, cells = _call(
                    name, forecaster, (data,), data, targets, horizon
                )
                at_origin.append(cells)
                if forecast is not None:
                    base[name] = forecast
            earlier = (
                pd.concat(ticker_cells, ignore_index=True)
                if ticker_cells
                else pd.DataFrame(columns=CELL_COLUMNS)
            )
            for name, combiner in combiners.items():
                _, cells = _call(
                    name, combiner, (data, base, earlier), data, targets, horizon
                )
                at_origin.append(cells)
            for cells in at_origin:
                cells["role"] = role
            ticker_cells.extend(at_origin)
        logger.info(
            f"{ticker}: {len(origins)} origins in {time.perf_counter() - started:.0f}s"
        )
        frames.extend(ticker_cells)
    if not frames:
        return pd.DataFrame(columns=CELL_COLUMNS)
    cells = pd.concat(frames, ignore_index=True).assign(window=window)
    extra = [c for c in cells.columns if c not in CELL_COLUMNS]
    return cells[CELL_COLUMNS + extra]


def check_failure_rate(cells: pd.DataFrame, limit: float = MAX_FAILURE_SHARE) -> None:
    """Raise FailureRateError if any model failed on more than `limit` of
    its cells."""
    if cells.empty:
        return
    share = cells["status"].eq("failed").groupby(cells["model"]).mean()
    over = share[share > limit]
    if not over.empty:
        detail = ", ".join(f"{m} {s:.0%}" for m, s in over.items())
        raise FailureRateError(
            f"failed on more than {limit:.0%} of their cells: {detail}"
        )
