"""tests/unit/test_split_jump_check.py

Covers the split guard's jump check (quantflow.ingestion.split_guard) and its
acknowledged list (config.split_jump_acknowledged).

The basis check compares stored closes with Yahoo's, so it cannot run when
Yahoo's reference download fails; the ticker then counts as unchecked, and
the scheduler reused the previous trading day's pass and modelled a split
that took effect today. The jump check needs nothing from Yahoo: it fails on
any stored close at or below 0.733 times, or at or above 1.35 times, the
stored close before it. That is the size of a 3-for-2 split or its reverse
with 10% for the day's own move, so a split of any larger ratio is caught.

A real move of that size is listed, with its reason, in
config/split_jump_acknowledged.txt. The listing silences the move only when
Yahoo's comparison ran in the same run and Yahoo's raw closes agree with the
stored ones on the move date and the date before; otherwise the move fails.
"""

import logging
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from quantflow import config
from quantflow.ingestion import split_guard, yfinance_fetcher

# 2026-05-04 to 2026-05-15; DAYS[5] is Monday 2026-05-11.
DAYS = pd.bdate_range("2026-05-04", periods=10)
EXCHANGE_TZ = "America/New_York"


def _closes(values):
    return pd.DataFrame({"date": DAYS[: len(values)], "close": values})


@pytest.fixture(autouse=True)
def acknowledged_file(tmp_path, monkeypatch):
    """An acknowledged list of this test's own, empty until it writes one."""
    path = tmp_path / "split_jump_acknowledged.txt"
    path.write_text("", encoding="utf-8")
    monkeypatch.setattr(config, "SPLIT_JUMP_ACKNOWLEDGED_FILE", path)
    return path


def _stored(read_tables, price_bars, closes):
    """AAPL's stored daily bars, one per day of DAYS."""
    read_tables["raw_prices"] = price_bars.frame(
        [
            price_bars.daily(day.date().isoformat(), close)
            for day, close in zip(DAYS, closes, strict=True)
        ]
    )


def _yahoo_daily(monkeypatch, closes):
    """yf.download serves these raw daily closes; an empty list serves nothing."""
    if not closes:
        frame = pd.DataFrame()
    else:
        frame = pd.DataFrame(
            {
                "Open": closes,
                "High": closes,
                "Low": closes,
                "Close": closes,
                "Adj Close": closes,
                "Volume": 1e6,
            },
            index=DAYS[: len(closes)],
        )
    monkeypatch.setattr(
        yfinance_fetcher.yf, "download", lambda ticker, **kwargs: frame.copy()
    )


def _yahoo_splits(monkeypatch, events):
    """yf.Ticker(t) serves Yahoo's split events: a Series of ratios (new
    shares per old) by date, None for an empty answer, or an exception to
    raise.

    Both ways yfinance offers them are faked as yfinance behaves: history()
    returns a "Stock Splits" column over the days it is asked for, and the
    splits property downloads every day there is and keeps the events (an
    empty answer reads as no events). Returns what was asked for: the tickers
    built and the keyword arguments of each history() request.
    """
    calls = SimpleNamespace(tickers=[], history=[])

    class FakeTicker:
        def __init__(self, ticker):
            calls.tickers.append(ticker)

        def _frame(self):
            if isinstance(events, Exception):
                raise events
            if events is None:
                return pd.DataFrame()
            days = DAYS.tz_localize(EXCHANGE_TZ).union(events.index)
            return pd.DataFrame(
                {"Close": 100.0, "Stock Splits": events.reindex(days, fill_value=0.0)},
                index=days,
            )

        def history(self, **kwargs):
            calls.history.append(kwargs)
            return self._frame()

        @property
        def splits(self):
            frame = self._frame()
            if frame.empty:
                return NO_SPLITS
            return frame.loc[frame["Stock Splits"] != 0, "Stock Splits"]

    monkeypatch.setattr(yfinance_fetcher.yf, "Ticker", FakeTicker)
    return calls


def _yahoo_split_on(day, ratio):
    return pd.Series(
        [ratio],
        index=pd.DatetimeIndex([pd.Timestamp(day, tz=EXCHANGE_TZ)]),
        name="Stock Splits",
    )


NO_SPLITS = pd.Series(
    [], index=pd.DatetimeIndex([], tz=EXCHANGE_TZ), dtype=float, name="Stock Splits"
)


def _cli_err(monkeypatch, capsys):
    """Run the guard's command for AAPL; return its stderr after checking it
    exited 1."""
    monkeypatch.setattr(split_guard, "TICKERS", ["AAPL"])
    assert split_guard.main([]) == 1
    return capsys.readouterr().err


# ── split_sized_jumps ─────────────────────────────────────────────────────────

# Price after / price before for a split of each ratio, and its name. The
# old ratio table missed the ones marked: they are flagged by size now.
SPLITS = [
    (1 / 2, "2-for-1 split"),
    (1 / 3, "3-for-1 split"),
    (2 / 3, "3-for-2 split"),
    (1 / 4, "4-for-1 split"),
    (1 / 5, "5-for-1 split"),
    (1 / 7, "7-for-1 split"),  # AAPL, 2014
    (1 / 8, "8-for-1 split"),
    (1 / 10, "10-for-1 split"),
    (1 / 20, "20-for-1 split"),
    (1 / 50, "50-for-1 split"),  # CMG, 2024
    (2, "1-for-2 reverse split"),
    (3, "1-for-3 reverse split"),
    (3 / 2, "2-for-3 reverse split"),
    (4, "1-for-4 reverse split"),
    (5, "1-for-5 reverse split"),
    (8, "1-for-8 reverse split"),  # GE, 2021
    (10, "1-for-10 reverse split"),
    (20, "1-for-20 reverse split"),
]


@pytest.mark.parametrize("ratio, name", SPLITS)
def test_a_split_of_any_ratio_is_flagged_and_named(ratio, name):
    stored = _closes([100.0] * 5 + [100.0 * ratio] * 5)  # effective on DAYS[5]

    jumps = split_guard.split_sized_jumps(stored)

    assert list(jumps["date"]) == [DAYS[5]]
    assert list(jumps["previous_date"]) == [DAYS[4]]
    assert list(jumps["nearest"]) == [name]
    assert jumps["ratio"].iloc[0] == pytest.approx(ratio)


@pytest.mark.parametrize("ratio", [1 / 7, 1 / 50, 8.0, 0.58, 0.4, 2.5])
def test_moves_the_old_ratio_table_let_through_are_flagged(ratio):
    """7-for-1, 50-for-1 and 1-for-8 were not in the table, and moves between
    its bands (0.58, a 2-for-1 on a day the stock also fell 20% at 0.4, 2.5)
    were never flagged. A split reaches the models whatever its ratio."""
    jumps = split_guard.split_sized_jumps(_closes([100.0, 100.0 * ratio]))
    assert list(jumps["date"]) == [DAYS[1]]


@pytest.mark.parametrize("day_move", [-0.2, -0.105, 0.105, 0.2])
def test_a_split_on_a_day_the_stock_also_moved_is_still_flagged(day_move):
    """The split day's own move compounds with the split: a 2-for-1 split on
    a day the stock fell 20% leaves 0.4 of the previous close, not 0.5."""
    jumps = split_guard.split_sized_jumps(_closes([100.0, 50.0 * (1 + day_move)]))
    assert list(jumps["date"]) == [DAYS[1]]


@pytest.mark.parametrize("move", [0.92, 1.12, 0.75, 1.30, 0.734, 1.349])
def test_moves_short_of_the_limits_are_not_flagged(move):
    """Ordinary days, and falls under 26.7% or rises under 35%."""
    assert split_guard.split_sized_jumps(_closes([100.0, 100.0 * move])).empty


@pytest.mark.parametrize("move", [0.733, 1.351])
def test_a_fall_to_0_733_or_a_rise_to_1_35_of_the_close_is_flagged(move):
    """The outer edges of 3-for-2 and its reverse with 10% for the day's own
    move: 2/3 x 1.1 and 3/2 x 0.9."""
    jumps = split_guard.split_sized_jumps(_closes([100.0, 100.0 * move]))
    assert list(jumps["date"]) == [DAYS[1]]


def test_a_missing_or_non_positive_close_is_skipped():
    """The move is measured from the last usable close, so a bad print the
    day before a split cannot hide it."""
    jumps = split_guard.split_sized_jumps(_closes([100.0, float("nan"), 0.0, 50.0]))
    assert list(jumps["date"]) == [DAYS[3]]
    assert list(jumps["previous_date"]) == [DAYS[0]]


def test_a_one_day_spike_is_flagged_on_both_days():
    """A spike that reverses the next day is often a bad print, but one row
    stored on the new basis among rows on the old one (a gap filled after a
    split) looks exactly the same. So unlike the basis check, which excuses
    isolated mismatches, this check excuses nothing by itself."""
    jumps = split_guard.split_sized_jumps(_closes([400.0, 400.0, 200.0, 400.0]))
    assert list(jumps["date"]) == [DAYS[2], DAYS[3]]
    assert list(jumps["nearest"]) == ["2-for-1 split", "1-for-2 reverse split"]


def test_the_name_is_only_the_nearest_common_split():
    """TSLA's 5-for-1 split on 2020-08-31 fell on a day the stock rose 12.6%:
    2213.40 then 498.32, 0.2251 of the close before, which is nearer 4-for-1.
    Hence the failure points to Yahoo's split events for the actual ratio."""
    jumps = split_guard.split_sized_jumps(_closes([2213.40, 498.32]))
    assert list(jumps["nearest"]) == ["4-for-1 split"]


@pytest.mark.parametrize(
    "shares, name",
    [
        (10.0, "10-for-1 split"),
        (0.1, "1-for-10 reverse split"),
        (1.5, "3-for-2 split"),
        (2 / 3, "2-for-3 reverse split"),
        (7.0, "7-for-1 split"),
        (0.125, "1-for-8 reverse split"),
    ],
)
def test_yahoos_split_ratios_are_named_like_the_flagged_moves(shares, name):
    """yfinance gives new shares per old; the failure message puts Yahoo's
    events next to the flagged moves, so both must use the same names."""
    assert split_guard._split_name(shares) == name


# ── run / main ────────────────────────────────────────────────────────────────


def test_a_split_sized_move_fails_the_run_when_yahoo_returns_nothing(
    read_tables, price_bars, monkeypatch
):
    """Without a reference from Yahoo the basis check cannot run. Before, the
    ticker only went unchecked, and the scheduler reused the previous trading
    day's pass and modelled the split."""
    _stored(read_tables, price_bars, [400.0] * 5 + [200.0] * 5)
    _yahoo_daily(monkeypatch, [])

    with pytest.raises(split_guard.PriceBasisError, match="AAPL"):
        split_guard.run(["AAPL"], threshold=0.15)


def test_the_cli_fails_on_a_split_sized_move_that_yahoo_shows_too(
    read_tables, price_bars, monkeypatch, capsys
):
    """Yahoo's raw closes show the same one-day fall of 34%, so the basis
    check passes. The run still stops until someone has looked: either the
    move was real (then it goes in the acknowledged list) or Yahoo has not
    adjusted its history yet. Yahoo's split events help decide which."""
    closes = [100.0] * 5 + [66.0] * 5
    _stored(read_tables, price_bars, closes)
    _yahoo_daily(monkeypatch, closes)
    _yahoo_splits(monkeypatch, _yahoo_split_on("2026-05-11", 1.5))

    err = _cli_err(monkeypatch, capsys)

    assert "SPLIT GUARD FAILED" in err
    assert "AAPL (2026-05-11: 0.6600x, nearest 3-for-2 split)" in err
    assert (
        "Yahoo's split events for AAPL since 2026-05-04: 3-for-2 split on 2026-05-11"
        in err
    )


def test_the_cli_says_when_yahoo_lists_no_split_without_calling_it_proof(
    read_tables, price_bars, monkeypatch, capsys
):
    """Yahoo can be late to record a split, so the message says what Yahoo
    lists, not that there was none."""
    closes = [100.0] * 5 + [66.0] * 5
    _stored(read_tables, price_bars, closes)
    _yahoo_daily(monkeypatch, closes)
    _yahoo_splits(monkeypatch, _yahoo_split_on("2020-08-31", 4.0))  # before history

    err = _cli_err(monkeypatch, capsys)
    assert "Yahoo's split events for AAPL since 2026-05-04: none listed" in err


def test_split_events_that_cannot_be_fetched_do_not_hide_the_failure(
    read_tables, price_bars, monkeypatch, capsys
):
    closes = [100.0] * 5 + [66.0] * 5
    _stored(read_tables, price_bars, closes)
    _yahoo_daily(monkeypatch, closes)
    _yahoo_splits(monkeypatch, RuntimeError("Yahoo is down"))

    err = _cli_err(monkeypatch, capsys)
    assert "Yahoo's split events for AAPL could not be fetched" in err


def test_an_empty_answer_for_split_events_reads_as_not_fetched_not_as_none(
    read_tables, price_bars, monkeypatch, capsys
):
    """yfinance catches most request errors and returns empty data, which
    the splits property cannot tell from no splits at all. A daily history
    from the first stored date is never empty when the request worked."""
    closes = [100.0] * 5 + [66.0] * 5
    _stored(read_tables, price_bars, closes)
    _yahoo_daily(monkeypatch, closes)
    _yahoo_splits(monkeypatch, None)

    err = _cli_err(monkeypatch, capsys)
    assert "Yahoo's split events for AAPL could not be fetched" in err


def test_split_events_are_requested_only_from_the_first_stored_date(
    read_tables, price_bars, monkeypatch, capsys
):
    """Ticker.splits downloads a ticker's whole history, decades of bars, to
    list a few events; the failure needs only the stored window."""
    closes = [100.0] * 5 + [66.0] * 5
    _stored(read_tables, price_bars, closes)
    _yahoo_daily(monkeypatch, closes)
    calls = _yahoo_splits(monkeypatch, NO_SPLITS)

    _cli_err(monkeypatch, capsys)
    assert [request["start"] for request in calls.history] == ["2026-05-04"]


def test_split_events_that_cannot_be_read_do_not_hide_the_failure(
    read_tables, price_bars, monkeypatch, capsys
):
    closes = [100.0] * 5 + [66.0] * 5
    _stored(read_tables, price_bars, closes)
    _yahoo_daily(monkeypatch, closes)
    _yahoo_splits(monkeypatch, _yahoo_split_on("2026-05-11", float("inf")))

    err = _cli_err(monkeypatch, capsys)
    assert "SPLIT GUARD FAILED" in err
    assert "Yahoo's split events for AAPL could not be read" in err


def test_the_guard_itself_asks_yahoo_for_no_split_events(
    read_tables, price_bars, monkeypatch
):
    """The scheduler pauses on the error run() raises, and price jobs that
    start meanwhile still pass the old gate, so run() raises at once. Only
    the command, which a person reads, fetches Yahoo's split events."""
    closes = [100.0] * 5 + [66.0] * 5
    _stored(read_tables, price_bars, closes)
    _yahoo_daily(monkeypatch, closes)
    calls = _yahoo_splits(monkeypatch, NO_SPLITS)

    with pytest.raises(split_guard.PriceBasisError):
        split_guard.run(["AAPL"], threshold=0.15)
    assert calls.tickers == []


def test_a_split_shows_as_a_break_and_a_move_with_its_events_listed_once(
    read_tables, price_bars, monkeypatch, capsys
):
    """What a real split leaves once Yahoo has adjusted its history: the rows
    before it disagree with Yahoo, and the stored close halves in one day."""
    _stored(read_tables, price_bars, [400.0] * 5 + [200.0] * 5)
    _yahoo_daily(monkeypatch, [200.0] * 10)
    _yahoo_splits(monkeypatch, _yahoo_split_on("2026-05-11", 2.0))

    err = _cli_err(monkeypatch, capsys)

    assert "disagree with Yahoo for AAPL" in err
    assert "AAPL (2026-05-11: 0.5000x, nearest 2-for-1 split)" in err
    assert err.count("Yahoo's split events for AAPL") == 1


def test_the_failure_lists_five_moves_then_counts_the_rest(
    read_tables, price_bars, monkeypatch
):
    closes = [100.0, 200.0] * 5  # nine split-sized moves
    _stored(read_tables, price_bars, closes)
    _yahoo_daily(monkeypatch, closes)

    with pytest.raises(split_guard.PriceBasisError) as failure:
        split_guard.run(["AAPL"], threshold=0.15)
    message = str(failure.value)
    assert message.count("nearest") == 5
    assert "; and 4 more)" in message


def test_a_move_found_before_the_yahoo_comparison_raised_is_still_reported(
    read_tables, price_bars, monkeypatch
):
    """The jump check runs before the reference download and keeps its
    result whatever happens to the comparison after it."""
    _stored(read_tables, price_bars, [400.0] * 5 + [200.0] * 5)

    def broken(ticker, start):
        raise KeyError("Close")

    monkeypatch.setattr(split_guard, "yahoo_daily_closes", broken)

    with pytest.raises(
        split_guard.PriceBasisError, match=r"AAPL.*could not check AAPL"
    ):
        split_guard.run(["AAPL"], threshold=0.15)


# ── acknowledged moves ────────────────────────────────────────────────────────


def test_an_acknowledged_move_that_yahoo_confirms_passes(
    read_tables, price_bars, monkeypatch, acknowledged_file, caplog
):
    closes = [100.0] * 5 + [66.0] * 5
    _stored(read_tables, price_bars, closes)
    _yahoo_daily(monkeypatch, closes)
    acknowledged_file.write_text(
        "AAPL:2026-05-11  # fell 34% on earnings, no split announced\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.INFO):
        split_guard.run(["AAPL"], threshold=0.15)  # must not raise

    assert "acknowledged and confirmed by Yahoo: fell 34% on earnings" in caplog.text


def test_an_acknowledged_move_fails_when_yahoo_returns_nothing(
    read_tables, price_bars, monkeypatch, acknowledged_file
):
    """Nothing confirms the move without Yahoo's reference, so it counts,
    acknowledged or not: the run fails and the scheduler pauses rather than
    reuse a recent pass."""
    _stored(read_tables, price_bars, [100.0] * 5 + [66.0] * 5)
    _yahoo_daily(monkeypatch, [])
    acknowledged_file.write_text("AAPL:2026-05-11  # real\n", encoding="utf-8")

    with pytest.raises(split_guard.PriceBasisError) as failure:
        split_guard.run(["AAPL"], threshold=0.15)
    message = str(failure.value)
    assert "acknowledged but not confirmed by Yahoo" in message
    assert "not compared with Yahoo (no reference prices): AAPL" in message


def test_an_acknowledged_move_fails_when_the_comparison_raises(
    read_tables, price_bars, monkeypatch, acknowledged_file
):
    _stored(read_tables, price_bars, [100.0] * 5 + [66.0] * 5)
    acknowledged_file.write_text("AAPL:2026-05-11  # real\n", encoding="utf-8")

    def broken(ticker, start):
        raise KeyError("Close")

    monkeypatch.setattr(split_guard, "yahoo_daily_closes", broken)

    with pytest.raises(
        split_guard.PriceBasisError,
        match=r"acknowledged but not confirmed by Yahoo.*could not check AAPL",
    ):
        split_guard.run(["AAPL"], threshold=0.15)


@pytest.mark.parametrize(
    "closes, acknowledged",
    [
        pytest.param(
            [100.0] * 5 + [50.0] + [100.0] * 4,
            ["2026-05-11", "2026-05-12"],
            id="a-row-on-another-basis",
        ),
        pytest.param(
            [400.0] * 3 + [4000.0] + [400.0] * 6,
            ["2026-05-07", "2026-05-08"],
            id="a-tenfold-bad-print",
        ),
    ],
)
def test_a_move_yahoo_disagrees_with_cannot_be_acknowledged_away(
    read_tables, price_bars, monkeypatch, acknowledged_file, closes, acknowledged
):
    """Yahoo's closes stay flat, so one stored row made the move. The basis
    check excuses a single mismatch as a bad print; an acknowledgement must
    not finish the job. The row has to be fixed."""
    _stored(read_tables, price_bars, closes)
    _yahoo_daily(monkeypatch, [closes[0]] * 10)
    acknowledged_file.write_text(
        "".join(f"AAPL:{day}  # looked real\n" for day in acknowledged),
        encoding="utf-8",
    )

    with pytest.raises(split_guard.PriceBasisError) as failure:
        split_guard.run(["AAPL"], threshold=0.15)
    assert str(failure.value).count("acknowledged but not confirmed by Yahoo") == 2


def test_an_acknowledgement_cannot_hide_a_real_split(
    read_tables, price_bars, monkeypatch, acknowledged_file
):
    """Acknowledging the day a split took effect: the rows stored before it
    still disagree with Yahoo's adjusted history, and Yahoo does not confirm
    the move either."""
    _stored(read_tables, price_bars, [400.0] * 5 + [200.0] * 5)
    _yahoo_daily(monkeypatch, [200.0] * 10)  # Yahoo serves it all split-adjusted
    acknowledged_file.write_text(
        "AAPL:2026-05-11  # taken for a real move by mistake\n", encoding="utf-8"
    )

    with pytest.raises(split_guard.PriceBasisError) as failure:
        split_guard.run(["AAPL"], threshold=0.15)
    message = str(failure.value)
    assert "disagree with Yahoo for AAPL" in message
    assert "acknowledged but not confirmed by Yahoo" in message


def test_an_acknowledgement_covers_only_its_own_ticker_and_date(
    read_tables, price_bars, monkeypatch, acknowledged_file
):
    closes = [100.0] * 5 + [66.0] * 5
    _stored(read_tables, price_bars, closes)
    _yahoo_daily(monkeypatch, closes)
    acknowledged_file.write_text(
        "MSFT:2026-05-11  # another ticker\nAAPL:2026-05-08  # the day before\n",
        encoding="utf-8",
    )

    with pytest.raises(split_guard.PriceBasisError, match=r"AAPL \(2026-05-11: "):
        split_guard.run(["AAPL"], threshold=0.15)


def test_an_acknowledgement_that_matches_no_move_is_reported(
    read_tables, price_bars, monkeypatch, acknowledged_file, caplog
):
    """A typo in the date silences nothing; say so, rather than leave the
    real move failing with no clue why its line did not apply."""
    closes = [100.0] * 5 + [66.0] * 5
    _stored(read_tables, price_bars, closes)
    _yahoo_daily(monkeypatch, closes)
    acknowledged_file.write_text(
        "AAPL:2026-05-11  # real\nAAPL:2026-05-08  # a typo\n", encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING):
        split_guard.run(["AAPL"], threshold=0.15)  # 2026-05-11 still passes

    assert (
        "AAPL: the acknowledged move on 2026-05-08 matches no split-sized move"
        in caplog.text
    )


def test_the_cli_reports_a_bad_acknowledged_list_before_reading_prices(
    acknowledged_file, capsys
):
    """No read_tables here: the list is checked before any database read."""
    acknowledged_file.write_text("# header\nAAPL:2026-05-11\n", encoding="utf-8")

    assert split_guard.main([]) == 1

    err = capsys.readouterr().err
    assert "SPLIT GUARD MISCONFIGURED" in err
    assert "split_jump_acknowledged.txt, line 2" in err


# ── the acknowledged list (config.split_jump_acknowledged) ───────────────────


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_the_list_maps_each_ticker_and_date_to_its_reason(tmp_path):
    path = _write(
        tmp_path / "ack.txt",
        "# a header comment\n"
        "\n"
        "AAPL:2026-05-11  # fell 34% on earnings\n"
        "msft:2025-01-02 #  rose 50% on a bid, see #12  \n"
        "JPM:2025-03-03\t# a tab before the reason\r\n",
    )
    assert config.split_jump_acknowledged(path) == {
        ("AAPL", date(2026, 5, 11)): "fell 34% on earnings",
        ("MSFT", date(2025, 1, 2)): "rose 50% on a bid, see #12",
        ("JPM", date(2025, 3, 3)): "a tab before the reason",
    }


@pytest.mark.parametrize("ticker", ["BRK.B", "BRK-B", "ABCDEFGHIJ"])
def test_symbols_the_pattern_allows_are_accepted(tmp_path, ticker):
    """Class shares as listed (BRK.B) and as Yahoo writes them (BRK-B), up to
    the 10 characters raw_prices.ticker holds."""
    path = _write(tmp_path / "ack.txt", f"{ticker}:2026-05-11  # real\n")
    assert config.split_jump_acknowledged(path) == {(ticker, date(2026, 5, 11)): "real"}


def test_a_byte_order_mark_from_a_windows_editor_is_ignored(tmp_path):
    path = tmp_path / "ack.txt"
    path.write_text("# header\nAAPL:2026-05-11  # real\n", encoding="utf-8-sig")
    assert config.split_jump_acknowledged(path) == {("AAPL", date(2026, 5, 11)): "real"}


@pytest.mark.parametrize(
    "line",
    [
        "AAPL:2026-05-11",  # no reason
        "AAPL:2026-05-11  #   ",  # an empty reason
        "AAPL 2026-05-11  # no colon",
        "AAPL:2026-5-11  # not YYYY-MM-DD",
        "AAPL:2026-02-30  # no such date",
        ":2026-05-11  # no ticker",
        "ABCDEFGHIJK:2026-05-11  # longer than a stored symbol",
        "AAPL:2026-05-11:x  # trailing text",
        "AAPL:2099-01-02  # acknowledged before it happened",
    ],
)
def test_a_bad_line_is_rejected_naming_the_file_and_line(tmp_path, line):
    path = _write(tmp_path / "ack.txt", "# header\n" + line + "\n")
    with pytest.raises(ValueError, match=r"ack\.txt, line 2"):
        config.split_jump_acknowledged(path)


@pytest.mark.parametrize(
    "separator", ["\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"]
)
def test_a_line_break_git_does_not_show_cannot_hide_an_entry(tmp_path, separator):
    """Python's splitlines() also ends a line at these, git and a review diff
    do not: '#<sep>XYZ:...' reads as a comment in review and loaded as an
    entry, silencing a move nobody approved."""
    path = _write(
        tmp_path / "ack.txt",
        f"# header\n#{separator}  XYZ:2026-05-01  # fell 34% on earnings\n",
    )
    with pytest.raises(ValueError, match=r"ack\.txt, line 2"):
        config.split_jump_acknowledged(path)


def test_a_repeated_entry_is_rejected_naming_both_lines(tmp_path):
    path = _write(
        tmp_path / "ack.txt",
        "AAPL:2026-05-11  # one\n# comment\nAAPL:2026-05-11  # two\n",
    )
    with pytest.raises(ValueError, match=r"line 3: repeats line 1"):
        config.split_jump_acknowledged(path)


def test_a_missing_list_is_an_error_not_an_empty_list(tmp_path):
    """The file is committed, so a run without it is a broken checkout."""
    with pytest.raises(ValueError, match=r"missing\.txt not found"):
        config.split_jump_acknowledged(tmp_path / "missing.txt")


def test_a_list_that_cannot_be_read_is_rejected_naming_the_file(tmp_path):
    """A directory in its place, or a file the user cannot read."""
    with pytest.raises(ValueError, match="cannot be read"):
        config.split_jump_acknowledged(tmp_path)


def test_a_list_not_saved_as_utf8_is_rejected_naming_the_file(tmp_path):
    """A curly apostrophe saved in a Windows code page is byte 0x92."""
    path = tmp_path / "ack.txt"
    path.write_bytes(b"AAPL:2026-05-11  # Yahoo\x92s history was wrong\n")
    with pytest.raises(ValueError, match=r"ack\.txt is not UTF-8"):
        config.split_jump_acknowledged(path)
