"""tests/unit/test_trending_sources.py

The trending tier's two Yahoo sources (docs/trending.md, section 3), on the
answers captured in tests/fixtures/trending/ and on broken ones made from
them: each list comes back as a ListFetch, its items or the reason it failed,
and a source never raises, whatever Yahoo sends.

Offline: yf.screen and requests.get are handed in as fakes; conftest refuses
the real ones.
"""

import copy
import datetime as dt
import json
from pathlib import Path

import pytest
import requests

from quantflow.ingestion import trending_sources as ts

pytestmark = pytest.mark.trending

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "trending"
NOW = dt.datetime(2026, 11, 10, 20, 7, tzinfo=dt.UTC)


def _fixture(name):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _screen(answers):
    """A fake yf.screen serving answers by list name; an exception in
    answers is raised, as yf.screen raises on an HTTP error status."""
    calls = []

    def screen(name, count=None):
        calls.append((name, count))
        answer = answers[name]
        if isinstance(answer, BaseException):
            raise answer
        return copy.deepcopy(answer)

    screen.calls = calls
    return screen


def _screener(answers, lists=ts.SCREENER_LISTS):
    return ts.YahooScreenerSource(lists, screen=_screen(answers), now=lambda: NOW)


def _captured_screens():
    return {name: _fixture(name) for name in ts.SCREENER_LISTS}


def _response(body, status=200):
    """A real requests.Response holding body (bytes, or JSON-encoded)."""
    response = requests.Response()
    response.status_code = status
    response._content = body if isinstance(body, bytes) else json.dumps(body).encode()
    response.url = ts.TRENDING_URL + "?count=25&useQuotes=true"
    return response


def _trending(answer):
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        if isinstance(answer, BaseException):
            raise answer
        return answer

    source = ts.YahooTrendingSource(get=get, now=lambda: NOW)
    return source, calls


# ── The screeners ─────────────────────────────────────────────────────────────


def test_the_captured_screener_lists_become_items_in_yahoos_order():
    fetched = _screener(_captured_screens()).fetch()

    assert [f.list_name for f in fetched] == list(ts.SCREENER_LISTS)
    assert all(f.succeeded and f.source == "yahoo_screener" for f in fetched)
    assert [len(f.items) for f in fetched] == [25, 25, 25]
    actives = fetched[0].items
    assert [item.rank for item in actives] == list(range(1, 26))
    nvda = actives[0]
    assert (nvda.symbol, nvda.quote_type, nvda.short_name) == (
        "NVDA",
        "EQUITY",
        "NVIDIA Corporation",
    )
    assert nvda.price == pytest.approx(167.0805)
    assert nvda.pct_change == pytest.approx(-6.486535)
    assert (nvda.volume, nvda.market_cap) == (345697762, 4034492891136)
    assert nvda.first_trade_date == dt.date(1999, 1, 22)
    assert nvda.fetched_at == NOW


def test_spcx_is_on_most_actives_with_its_listing_as_first_trade_date():
    """The captured day's SPCX: the current company's listing, not the old
    fund's, so Yahoo's metadata already names the new holder."""
    actives = _screener(_captured_screens()).fetch()[0].items
    spcx = next(item for item in actives if item.symbol == "SPCX")
    assert spcx.first_trade_date == dt.date(2026, 6, 12)
    assert spcx.quote_type == "EQUITY"
    # Yahoo sends "Space Exploration Technologies " with a trailing space.
    assert spcx.short_name == "Space Exploration Technologies"


def test_each_list_is_asked_for_25_items_once():
    answers = _captured_screens()
    source = ts.YahooScreenerSource(screen=_screen(answers), now=lambda: NOW)
    source.fetch()
    assert source._screen.calls == [(name, 25) for name in ts.SCREENER_LISTS]


def test_an_empty_list_succeeds_with_no_items():
    answers = _captured_screens()
    answers["day_gainers"]["quotes"] = []
    gainers = _screener(answers).fetch()[1]
    assert gainers.succeeded and gainers.items == ()


@pytest.mark.parametrize(
    ("break_it", "reason"),
    [
        (lambda body: body.pop("quotes"), "has no quotes"),
        (lambda body: body.update(quotes={"0": "NVDA"}), "not a list"),
        (lambda body: body["quotes"].insert(3, "NVDA"), "item 4 is not an object"),
        (
            lambda body: body["quotes"][2].update(regularMarketPrice="141.30"),
            "item 3: regularMarketPrice is str, not a number",
        ),
        (
            lambda body: body["quotes"][0].update(regularMarketVolume=True),
            "regularMarketVolume is bool",
        ),
        (
            lambda body: body["quotes"][1].update(shortName=12),
            "shortName is int, not text",
        ),
    ],
)
def test_a_malformed_list_fails_and_the_others_still_answer(break_it, reason):
    answers = _captured_screens()
    break_it(answers["most_actives"])

    actives, gainers, losers = _screener(answers).fetch()

    assert not actives.succeeded and actives.items == ()
    assert reason in actives.error
    assert gainers.succeeded and len(gainers.items) == 25
    assert losers.succeeded and len(losers.items) == 25


def test_an_answer_that_is_not_an_object_fails_its_list():
    answers = _captured_screens()
    answers["day_losers"] = ["NVDA"]
    losers = _screener(answers).fetch()[2]
    assert losers.error == "MalformedListError: answer is list"


def test_an_http_error_fails_its_list_without_raising():
    answers = _captured_screens()
    answers["day_gainers"] = requests.HTTPError("429 Client Error: Too Many Requests")
    actives, gainers, losers = _screener(answers).fetch()
    assert gainers.error == "HTTPError: 429 Client Error: Too Many Requests"
    assert actives.succeeded and losers.succeeded


def test_any_exception_from_the_call_fails_its_list_without_raising():
    answers = _captured_screens()
    answers["most_actives"] = KeyError("finance")
    assert _screener(answers).fetch()[0].error == "KeyError: 'finance'"


def test_items_with_no_valid_symbol_fail_the_list_rather_than_empty_it():
    """What a renamed symbol field looks like: every item there, none
    usable. Read as an empty list, it would deactivate nothing and say
    nothing; as a failure, the run ends red."""
    answers = _captured_screens()
    for quote in answers["most_actives"]["quotes"]:
        quote["ticker"] = quote.pop("symbol")
    actives = _screener(answers).fetch()[0]
    assert actives.error == "MalformedListError: no valid symbol among 25 items"


def test_an_invalid_symbol_is_dropped_and_the_rest_kept(caplog):
    answers = _captured_screens()
    quotes = answers["most_actives"]["quotes"]
    quotes[1]["symbol"] = "intc"  # lower case
    quotes[4]["symbol"] = "IREN\nFAKE LOG LINE"
    quotes[6]["symbol"] = "TOOLONGSYMBOL"

    actives = _screener(answers).fetch()[0]

    assert actives.succeeded
    assert len(actives.items) == 22
    assert {item.symbol for item in actives.items}.isdisjoint(
        {"intc", "INTC", "TOOLONGSYMBOL"}
    )
    # Ranks are positions in Yahoo's list, gaps and all.
    assert [item.rank for item in actives.items][:3] == [1, 3, 4]
    # Logged escaped: the newline never reaches the log as a line break.
    logged = [r.getMessage() for r in caplog.records if "not symbols" in r.message]
    assert logged and "\n" not in logged[0]
    assert "'IREN\\nFAKE" in logged[0]


def test_a_symbol_repeated_in_one_list_keeps_its_first_rank():
    answers = _captured_screens()
    quotes = answers["day_losers"]["quotes"]
    quotes.append(copy.deepcopy(quotes[0]))
    items = _screener(answers).fetch()[2].items
    assert len(items) == 25
    assert [i.rank for i in items if i.symbol == quotes[0]["symbol"]] == [1]


def test_missing_fields_are_none_not_a_failure():
    answers = _captured_screens()
    quote = answers["day_gainers"]["quotes"][0]
    for field in ts.QUOTE_FIELDS[1:]:
        quote.pop(field)
    item = _screener(answers).fetch()[1].items[0]
    assert (item.symbol, item.rank) == (quote["symbol"], 1)
    assert item.price is None and item.quote_type is None
    assert item.short_name is None and item.first_trade_date is None
    assert item.volume is None and item.market_cap is None


def test_values_the_tables_cannot_hold_are_treated_as_missing():
    answers = _captured_screens()
    quote = answers["day_gainers"]["quotes"][0]
    quote.update(
        regularMarketChangePercent=2e6,
        regularMarketVolume=-5,
        marketCap=2**64,
        regularMarketPrice=float("nan"),
    )
    item = _screener(answers).fetch()[1].items[0]
    assert item.pct_change is None and item.volume is None
    assert item.market_cap is None and item.price is None


def test_a_first_trade_date_before_1970_is_read():
    """Yahoo gives one as negative milliseconds (^TNX: 1962). On Windows,
    datetime.fromtimestamp refuses those."""
    quote = {"symbol": "IBM", "firstTradeDateMilliseconds": -252326400000}
    (item,) = ts.items_from_quotes("yahoo_screener", "most_actives", [quote], NOW)
    assert item.first_trade_date == dt.date(1962, 1, 2)


def test_short_names_are_cleaned_and_cut_to_100_characters():
    quote = {
        "symbol": "XYZ",
        "shortName": "Acme\x00 Corp\u200b \x1b[31mred\x1b[0m\n" + "x" * 200,
        "quoteType": "EQUITY" * 10,
    }
    (item,) = ts.items_from_quotes("yahoo_screener", "most_actives", [quote], NOW)
    assert item.short_name.startswith("Acme Corp [31mred[0m x")
    assert len(item.short_name) == 100
    assert all(ch.isprintable() for ch in item.short_name)
    assert len(item.quote_type) == 20


def test_an_unknown_screener_list_is_refused_when_built():
    with pytest.raises(ValueError, match="not a screener list: trending_tickers"):
        ts.YahooScreenerSource(["trending_tickers"])


def test_without_a_fake_the_screener_goes_through_yfinance_and_fails_soft():
    """conftest refuses every yfinance request, which the source must catch
    like any other failure."""
    fetched = ts.YahooScreenerSource(["most_actives"]).fetch()
    assert len(fetched) == 1 and not fetched[0].succeeded
    assert "offline" in fetched[0].error


# ── The trending list ────────────────────────────────────────────────────────


def test_the_captured_trending_list_becomes_items():
    source, _ = _trending(_response(_fixture("trending_tickers")))
    (fetched,) = source.fetch()

    assert (fetched.source, fetched.list_name) == ("yahoo_trending", "trending_tickers")
    assert fetched.succeeded
    symbols = [item.symbol for item in fetched.items]
    # ^TNX is an index, not a symbol the tier can store: dropped. A private
    # company passes validation; the EQUITY rule removes it later.
    assert "^TNX" not in symbols and len(symbols) == 24
    private = next(item for item in fetched.items if item.symbol == "ANTH.PVT")
    assert private.quote_type == "PRIVATE_COMPANY" and private.rank == 7
    assert private.volume is None and private.market_cap is None
    assert fetched.items[0].symbol == "NVDA" and fetched.items[0].rank == 1
    # Yahoo cuts long names short, trailing comma and space included.
    ttwo = next(item for item in fetched.items if item.symbol == "TTWO")
    assert ttwo.short_name == "Take-Two Interactive Software,"


def test_the_trending_request_carries_its_timeout_user_agent_and_fields():
    source, calls = _trending(_response(_fixture("trending_tickers")))
    source.fetch()
    ((url, kwargs),) = calls
    assert url == "https://query1.finance.yahoo.com/v1/finance/trending/US"
    assert kwargs["timeout"] == 10
    assert kwargs["headers"] == {
        "User-Agent": "QuantFlow/1.0 (+https://github.com/logn1602/QuantFlow)"
    }
    assert kwargs["params"]["count"] == 25
    assert kwargs["params"]["useQuotes"] == "true"
    assert set(kwargs["params"]["fields"].split(",")) == set(ts.QUOTE_FIELDS)


def test_an_empty_trending_list_succeeds_with_no_items():
    body = _fixture("trending_tickers")
    body["finance"]["result"][0]["quotes"] = []
    source, _ = _trending(_response(body))
    (fetched,) = source.fetch()
    assert fetched.succeeded and fetched.items == ()


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (b"<html>Will be right back</html>", "JSONDecodeError"),
        ({"finance": {"result": None, "error": {"code": "x"}}}, "no result"),
        ({"finance": {"result": []}}, "no result"),
        ({"quotes": []}, "no result"),
        ([1, 2], "no result"),
        ({"finance": {"result": [{"count": 0}]}}, "the result has no quotes"),
        ({"finance": {"result": [{"quotes": "NVDA"}]}}, "not a list"),
    ],
)
def test_a_malformed_trending_answer_fails_without_raising(body, reason):
    source, _ = _trending(_response(body))
    (fetched,) = source.fetch()
    assert not fetched.succeeded and fetched.items == ()
    assert reason in fetched.error


def test_an_http_error_status_fails_the_trending_list():
    source, _ = _trending(_response(b"busy", status=503))
    (fetched,) = source.fetch()
    assert fetched.error.startswith("HTTPError: 503 Server Error")
    # The URL's query string is cut from the stored reason.
    assert "useQuotes" not in fetched.error and "trending/US?..." in fetched.error


@pytest.mark.parametrize(
    "error",
    [
        requests.ConnectionError("connection refused"),
        requests.Timeout("read timed out"),
        RuntimeError("anything at all"),
    ],
)
def test_a_failed_request_fails_the_trending_list_without_raising(error):
    source, _ = _trending(error)
    (fetched,) = source.fetch()
    assert fetched.error == f"{type(error).__name__}: {error}"


def test_an_answer_far_larger_than_the_list_is_refused():
    big = _response(b"{" + b" " * (ts.MAX_BODY_BYTES + 1) + b"}")
    source, _ = _trending(big)
    (fetched,) = source.fetch()
    assert "MalformedListError: answer of" in fetched.error


def test_without_a_fake_the_trending_request_fails_soft():
    (fetched,) = ts.YahooTrendingSource().fetch()
    assert not fetched.succeeded and "offline" in fetched.error


# ── Reasons and wiring ────────────────────────────────────────────────────────


def test_a_stored_reason_is_one_short_line_without_query_strings():
    error = RuntimeError(
        "401 for url https://query1.finance.yahoo.com/v1/finance/screener/"
        "predefined/saved?scrIds=most_actives&crumb=abc123\nsecond line " + "x" * 300
    )
    reason = ts._reason(error)
    assert "crumb" not in reason and "abc123" not in reason
    assert "\n" not in reason and len(reason) <= 200
    assert reason.startswith("RuntimeError: 401 for url https://query1")


def test_sources_are_built_for_the_lists_named():
    sources = ts.sources_for(["day_losers", "trending_tickers"])
    assert [type(s).__name__ for s in sources] == [
        "YahooTrendingSource",
        "YahooScreenerSource",
    ]
    assert sources[1].lists == ("day_losers",)
    assert ts.sources_for(["most_actives"])[0].lists == ("most_actives",)
    assert ts.sources_for([]) == []
