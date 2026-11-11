"""One trending symbol's analytics (docs/trending.md, 7.4 and section 10).

The core's own indicator and anomaly functions on the symbol's latest bars,
one row per date: the run summarises them onto its tickers row, and the tab
draws them. Shared, window included, so the two can never compute them
differently. Pure: no database, no network.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantflow.db.prices import PRICE_COLUMNS, daily_bars, indexed_by_date
from quantflow.features.anomalies import detect_anomalies
from quantflow.features.indicators import compute_indicators

# The dates a symbol's analytics read: its latest WINDOW_BARS, about a year of
# sessions. The run's summary and the tab's chart both analyse this window and
# no more: the IQR anomaly detector sets its fences from quantiles over its
# whole input, so a longer input can flag a date a shorter one does not (the
# Red Team pass, 2026-11-10). A history grows a bar a day while tracked.
WINDOW_BARS = 260


@dataclass(frozen=True)
class Analysis:
    daily: pd.DataFrame  # one row per date (db.prices.daily_bars), a date column
    indicators: pd.DataFrame  # compute_indicators, indexed by date
    flags: pd.DataFrame  # detect_anomalies, indexed by date


def analyse(bars: pd.DataFrame, symbol: str) -> Analysis:
    """bars (trending_prices rows: ts, open, high, low, close, volume) as one
    row per date, its latest WINDOW_BARS, with their indicators and anomaly
    flags."""
    daily = daily_bars(bars, symbol).tail(WINDOW_BARS).reset_index(drop=True)
    return Analysis(
        daily=daily,
        indicators=compute_indicators(indexed_by_date(daily, PRICE_COLUMNS)),
        flags=detect_anomalies(indexed_by_date(daily, ["close", "volume"]), symbol),
    )
