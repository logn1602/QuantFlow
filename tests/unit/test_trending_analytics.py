"""tests/unit/test_trending_analytics.py

One trending symbol's analytics (features/trending_analytics.py): the run's
summary and the tab's chart analyse the same window of bars, so the table's
flag count and the chart's flags always agree (docs/trending.md, 7.4 and
section 10).
"""

import numpy as np
import pandas as pd
import pytest

from quantflow.dashboard import trending_view as view
from quantflow.features import trending_analytics as analytics
from quantflow.ingestion import trending_rules as rules
from quantflow.pipelines import trending

pytestmark = pytest.mark.trending

DAYS = pd.bdate_range(end="2026-11-10", periods=400)


def _rows(seed=0):
    """400 sessions, quiet for 250 then volatile: the IQR detector's fences,
    quantiles over its whole input, move with the window it is given."""
    rng = np.random.default_rng(seed)
    returns = np.concatenate([rng.normal(0, 0.002, 250), rng.normal(0, 0.04, 150)])
    closes = 50 * np.exp(np.cumsum(returns))
    frame = pd.DataFrame(
        {
            "Open": closes,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Adj Close": closes,
            "Volume": 1_000_000,
        },
        index=pd.DatetimeIndex(DAYS),
    )
    return trending.price_rows(trending.as_daily_bars(frame), "ZZZ")


def test_the_table_and_the_chart_count_the_same_recent_flags():
    """The run summarised every bar it downloaded, the chart its latest 260:
    past 260 stored bars the table said 8 flags in the last 10 dates and the
    chart drew 3 (the Red Team pass, 2026-11-10)."""
    rows = _rows()
    summary, _ = trending.summarize(rows, "ZZZ")
    stored = pd.DataFrame(rows[-view.CHART_BARS :])  # what the tab reads
    _, _, flags = view.chart_frames(stored.drop(columns="ticker"), "ZZZ")
    start = pd.Timestamp(rules.recent_start(summary["summary_session"]))
    assert int((flags["date"] >= start).sum()) == summary["recent_flags"]


def test_analytics_read_the_latest_window_of_bars():
    analysis = analytics.analyse(pd.DataFrame(_rows()), "ZZZ")
    assert len(analysis.daily) == analytics.WINDOW_BARS
    assert analysis.daily["date"].iloc[-1] == DAYS[-1]
    assert analysis.daily["date"].iloc[0] == DAYS[-analytics.WINDOW_BARS]


def test_the_chart_reads_the_window_the_analytics_use():
    assert view.CHART_BARS == analytics.WINDOW_BARS == 260
