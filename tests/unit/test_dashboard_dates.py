"""tests/unit/test_dashboard_dates.py

Decision K (Checkpoint 1): the dashboard dates indicator, anomaly and
sentiment rows by their UTC calendar date, the date the price candles carry
(db.prices.daily_bars). A bare ts::date follows the session's TimeZone: on a
session west of UTC, every row stored at 00:00 UTC sat one day left of its
candle. The casts were fixed then; this is the regression test promised for
when a loader could be tested.

app.py runs Streamlit as it is imported (Known problem 5), so its SQL is
checked as text. The Trending tab's queries, importable in
dashboard/trending_view.py, are checked in test_trending_view.py.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = (ROOT / "src" / "quantflow" / "dashboard" / "app.py").read_text(encoding="utf-8")


def _loader(name):
    match = re.search(rf"\ndef {name}\(.*?(?=\n@cached|\ndef |\n# ──)", APP, re.DOTALL)
    assert match, f"no loader {name} in app.py"
    return match.group(0)


def test_the_candle_aligned_loaders_date_their_rows_in_utc():
    for name, column in (
        ("load_indicators", "ts"),
        ("load_anomalies", "ts"),
        ("load_sentiment", "published_at"),
    ):
        sql = _loader(name)
        assert f"({column} AT TIME ZONE 'UTC')::date AS date" in sql, name


def test_no_stored_bar_or_article_time_is_cast_to_a_date_bare():
    code = "\n".join(
        line for line in APP.splitlines() if not line.lstrip().startswith("#")
    )
    assert not re.search(r"(?<![\w(])(ts|published_at)::date", code)
