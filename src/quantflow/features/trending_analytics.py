"""One trending symbol's analytics (docs/trending.md, 7.4 and section 10).

The core's own indicator and anomaly functions on the symbol's bars, one row
per date: the run summarises them onto its tickers row, and the tab draws
them. Shared, so the two can never compute them differently. Pure: no
database, no network.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantflow.db.prices import PRICE_COLUMNS, daily_bars, indexed_by_date
from quantflow.features.anomalies import detect_anomalies
from quantflow.features.indicators import compute_indicators


@dataclass(frozen=True)
class Analysis:
    daily: pd.DataFrame  # one row per date (db.prices.daily_bars), a date column
    indicators: pd.DataFrame  # compute_indicators, indexed by date
    flags: pd.DataFrame  # detect_anomalies, indexed by date


def analyse(bars: pd.DataFrame, symbol: str) -> Analysis:
    """bars (trending_prices rows: ts, open, high, low, close, volume) as one
    row per date, with its indicators and anomaly flags."""
    daily = daily_bars(bars, symbol)
    return Analysis(
        daily=daily,
        indicators=compute_indicators(indexed_by_date(daily, PRICE_COLUMNS)),
        flags=detect_anomalies(indexed_by_date(daily, ["close", "volume"]), symbol),
    )
