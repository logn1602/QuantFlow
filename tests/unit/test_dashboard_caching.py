"""tests/unit/test_dashboard_caching.py

The dashboard's loaders keep what they read for an hour (dashboard/caching.py),
except an empty result. A loader that ran while the forecasts were being
replaced, before a ticker's first run, or during an outage kept "nothing" for
the hour, and every visitor saw it although the data was back.
"""

import pandas as pd
import pytest
import streamlit as st

from quantflow.dashboard import caching


@pytest.fixture(autouse=True)
def fresh_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _loader(answers):
    """A loader returning each answer in turn, counting its reads."""
    reads = []

    def load(ticker):
        reads.append(ticker)
        return answers[min(len(reads), len(answers)) - 1]

    return load, reads


def test_an_empty_result_is_read_again_next_time():
    empty = pd.DataFrame(columns=["ticker", "forecast_date"])
    full = pd.DataFrame({"ticker": ["AAPL"], "forecast_date": ["2026-11-10"]})
    load, reads = _loader([empty, full])
    cached = caching.cached(load)

    assert cached("AAPL").empty
    assert len(cached("AAPL")) == 1
    assert len(reads) == 2


def test_a_result_with_rows_is_kept():
    full = pd.DataFrame({"ticker": ["AAPL"], "forecast_date": ["2026-11-10"]})
    load, reads = _loader([full])
    cached = caching.cached(load)

    cached("AAPL")
    cached("AAPL")
    assert len(reads) == 1


@pytest.mark.parametrize("empty", [{}, [], None], ids=["dict", "list", "None"])
def test_other_empty_results_are_not_kept_either(empty):
    load, reads = _loader([empty, {"AAPL": 1.0}])
    cached = caching.cached(load)

    assert cached("AAPL") == empty
    assert cached("AAPL") == {"AAPL": 1.0}
    assert len(reads) == 2


def test_two_loaders_keep_their_own_results():
    """Streamlit keys a cache by the function's name and source; each loader
    passes its own to it."""

    def load_one(ticker):
        return pd.DataFrame({"x": [1]})

    def load_two(ticker):
        return pd.DataFrame({"x": [2]})

    one, two = caching.cached(load_one), caching.cached(load_two)
    assert one("AAPL")["x"].iloc[0] == 1
    assert two("AAPL")["x"].iloc[0] == 2


def test_the_cache_lifetime_picks_up_the_daily_refresh_within_an_hour():
    assert 0 < caching.CACHE_SECONDS <= 3600
