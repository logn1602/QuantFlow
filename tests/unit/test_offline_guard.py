"""tests/unit/test_offline_guard.py

Covers the _no_network guard in tests/conftest.py. yfinance fetches through
its own curl_cffi session, not requests, so refusing yf.download and
requests left every other yfinance call (yf.Ticker(...).splits, which the
split guard reads, among them) free to reach Yahoo from a unit test.
yfinance catches most request errors itself and returns empty data, so what
matters is whether a request got as far as the HTTP client.
"""

import curl_cffi.requests
import pytest
import yfinance as yf
from yfinance.ticker import Ticker


@pytest.fixture
def network_calls(monkeypatch):
    """Record any request that gets as far as curl_cffi, and stop it there."""
    calls = []

    def record(self, *args, **kwargs):
        calls.append(kwargs.get("url") or args[1:2])
        raise RuntimeError("reached curl_cffi")

    monkeypatch.setattr(curl_cffi.requests.Session, "request", record)
    return calls


def _error_from(call):
    """The exception call raised, or None."""
    try:
        call()
    except Exception as e:
        return e
    return None


def test_yf_ticker_is_refused(network_calls):
    error = _error_from(lambda: yf.Ticker("AAPL").splits)
    assert network_calls == []
    assert "tests are offline" in str(error)


class _KnownZone:
    """yfinance's time zone cache, answering from memory. The real one is a
    SQLite file in the user's cache folder: a machine that has run the
    pipeline knows AAPL's zone and goes on to the price request, a fresh CI
    runner stops at the zone request, and both write outside the test."""

    def lookup(self, ticker):
        return "America/New_York"

    def store(self, ticker, zone):
        pass


def test_a_yfinance_request_from_any_other_entry_point_is_refused(
    network_calls, monkeypatch
):
    """Ticker's class, reached without the module attribute, stands in for
    every other way into yfinance (Tickers, Search, the screeners). The
    refusal may not surface: yfinance can catch it and return empty data
    instead, so only the request is checked."""
    import yfinance.cache

    monkeypatch.setattr(yfinance.cache, "get_tz_cache", lambda: _KnownZone())
    _error_from(lambda: Ticker("AAPL").splits)
    assert network_calls == []


def test_an_engine_for_another_role_is_refused_too():
    """db.readonly reads the live database through a second engine, logged in
    as the read-only role; the unit suite must refuse that one as well."""
    from quantflow.db import connection, readonly

    for module in (connection, readonly):
        with pytest.raises(RuntimeError, match="offline"):
            module.engine_as("quantflow_readonly", "pw")
