"""tests/unit/test_trending_config.py

The trending tier's settings (docs/trending.md, section 11), symbol validation
(4.1) and the listing-date file (section 5).

Every setting is read when the tier runs, never at import: a blank value
takes the default, and a bad one stops the trending run alone, never the
config import the dashboard shares. These tests set config's attributes
directly, since settings() reads them when called, and pass the core
watchlist in: no trending test reads config.TICKERS (CI and the daily run set
different ones).
"""

import importlib
import re
from pathlib import Path

import pytest

import quantflow.config as config_module
from quantflow.ingestion import trending_rules as rules

pytestmark = pytest.mark.trending

ROOT = Path(__file__).resolve().parents[2]
CORE = ("AAPL", "MSFT")
SETTINGS = (
    "TRENDING_PINNED",
    "TRENDING_LISTS",
    "TRENDING_MAX_ACTIVE",
    "TRENDING_MAX_NEW_PER_RUN",
    "TRENDING_INACTIVE_AFTER_DAYS",
    "TRENDING_MIN_PRICE",
    "TRENDING_MIN_MARKET_CAP",
    "TRENDING_PROMOTION_MIN_DAYS",
    "PROMOTION_MIN_HISTORY_DAYS",
)


@pytest.fixture
def blank(monkeypatch):
    """Every trending setting blank, as .env.example leaves most of them."""
    for name in SETTINGS:
        monkeypatch.setattr(config_module, name, "")
    return monkeypatch


def test_blank_settings_take_the_documented_defaults(blank):
    s = rules.settings(CORE)
    assert s.pinned == ("SPCX",)
    assert s.lists == (
        "trending_tickers",
        "most_actives",
        "day_gainers",
        "day_losers",
    )
    assert (s.max_active, s.max_new_per_run, s.inactive_after) == (15, 5, 5)
    assert s.min_price == 5.0 and s.min_market_cap is None
    assert (s.promotion_min_days, s.promotion_min_history_days) == (20, 504)


def test_whitespace_counts_as_blank(blank):
    blank.setattr(config_module, "TRENDING_MAX_ACTIVE", "  ")
    blank.setattr(config_module, "TRENDING_PINNED", " ")
    s = rules.settings(CORE)
    assert s.max_active == 15 and s.pinned == ("SPCX",)


def test_values_are_read(blank):
    for name, value in {
        "TRENDING_LISTS": " day_losers , MOST_ACTIVES,day_losers",
        "TRENDING_MAX_ACTIVE": "3",
        "TRENDING_MAX_NEW_PER_RUN": "0",
        "TRENDING_INACTIVE_AFTER_DAYS": "2",
        "TRENDING_MIN_PRICE": "0",
        "TRENDING_MIN_MARKET_CAP": "2e9",
        "TRENDING_PROMOTION_MIN_DAYS": "10",
        "PROMOTION_MIN_HISTORY_DAYS": "252",
    }.items():
        blank.setattr(config_module, name, value)
    s = rules.settings(CORE)
    assert s.lists == ("day_losers", "most_actives")
    assert (s.max_active, s.max_new_per_run, s.inactive_after) == (3, 0, 2)
    assert s.min_price == 0.0  # 0 turns the floor off
    assert s.min_market_cap == 2e9
    assert (s.promotion_min_days, s.promotion_min_history_days) == (10, 252)


def test_a_zero_market_cap_floor_is_off(blank):
    blank.setattr(config_module, "TRENDING_MIN_MARKET_CAP", "0")
    assert rules.settings(CORE).min_market_cap is None


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("TRENDING_MAX_ACTIVE", "fifteen", "must be a whole number"),
        ("TRENDING_MAX_ACTIVE", "1.5", "must be a whole number"),
        ("TRENDING_MAX_ACTIVE", "-1", "must be at least 0"),
        ("TRENDING_INACTIVE_AFTER_DAYS", "0", "must be at least 1"),
        ("TRENDING_PROMOTION_MIN_DAYS", "0", "must be at least 1"),
        ("PROMOTION_MIN_HISTORY_DAYS", "two years", "must be a whole number"),
        ("TRENDING_MIN_PRICE", "$5", "must be a number of dollars"),
        ("TRENDING_MIN_PRICE", "-1", "must be 0 or more"),
        ("TRENDING_MIN_PRICE", "nan", "must be 0 or more"),
        ("TRENDING_MIN_MARKET_CAP", "inf", "must be 0 or more"),
        ("TRENDING_LISTS", "most_actives,top_gainers", "unknown list 'top_gainers'"),
        ("TRENDING_LISTS", " , ", "names no list"),
    ],
)
def test_a_bad_setting_stops_the_run_with_its_name(blank, name, value, message):
    blank.setattr(config_module, name, value)
    with pytest.raises(rules.TrendingConfigError, match=re.escape(message)) as error:
        rules.settings(CORE)
    assert name in str(error.value)


def test_a_bad_setting_does_not_break_the_config_import(monkeypatch):
    """The dashboard imports config: a typo in a trending setting must stop
    only the trending run."""
    monkeypatch.setenv("TRENDING_MAX_ACTIVE", "fifteen")
    monkeypatch.setenv("TRENDING_LISTS", "nonsense")
    try:
        reloaded = importlib.reload(config_module)
        assert reloaded.TRENDING_MAX_ACTIVE == "fifteen"
        with pytest.raises(rules.TrendingConfigError):
            rules.settings(CORE)
    finally:
        monkeypatch.undo()
        importlib.reload(config_module)


# ── Pinned symbols (section 5) ────────────────────────────────────────────────


def test_pinned_symbols_are_upcased_and_repeats_dropped():
    assert rules.pinned_symbols(" spcx, XYZ ,spcx,,", CORE) == ("SPCX", "XYZ")


def test_an_invalid_pinned_symbol_is_left_out_never_fatal(caplog):
    pinned = rules.pinned_symbols("SPCX,BAD SYMBOL,^GSPC,TOOLONGSYMBOL,X\nY", CORE)
    assert pinned == ("SPCX",)
    warnings = [r.getMessage() for r in caplog.records]
    assert sum("is not a symbol" in w for w in warnings) == 4
    assert all("\n" not in w for w in warnings)


def test_a_pinned_core_ticker_is_left_out(caplog):
    assert rules.pinned_symbols("AAPL,SPCX", CORE) == ("SPCX",)
    assert any("AAPL is a core ticker" in r.getMessage() for r in caplog.records)


def test_at_most_ten_pinned_symbols_are_read(caplog):
    raw = ",".join(f"P{i}" for i in range(12))
    pinned = rules.pinned_symbols(raw, CORE)
    assert pinned == tuple(f"P{i}" for i in range(10))
    assert any("more than 10" in r.getMessage() for r in caplog.records)


def test_the_pinned_default_is_the_same_in_config_and_env_example():
    """Kept in config.py, .env.example and trending.yml (section 5); the
    workflow's copy is checked with the workflow."""
    example = re.search(
        r"^TRENDING_PINNED=(.*)$",
        (ROOT / ".env.example").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert example, "no TRENDING_PINNED line in .env.example"
    assert example.group(1).strip() == config_module.DEFAULT_TRENDING_PINNED == "SPCX"


def test_every_trending_setting_is_in_env_example():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for name in SETTINGS:
        assert re.search(rf"^{name}=", text, re.MULTILINE), name


# ── Symbol validation (4.1) ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "symbol", ["A", "SPCX", "BRK.B", "BRK-B", "BF.A", "ANTH.PVT", "ABCDEFGHIJ", "X1"]
)
def test_valid_symbols(symbol):
    assert rules.valid_symbol(symbol)


@pytest.mark.parametrize(
    "symbol",
    [
        "",
        "ABCDEFGHIJK",  # 11 characters: raw_prices.ticker is VARCHAR(10)
        "spcx",  # lower case is refused, not upcased
        "Spcx",
        "1ABC",  # starts with a digit
        ".ABC",
        "^GSPC",
        "AB C",
        "AB_C",
        "AB/C",
        "AB'C",
        "SPCX\n",
        "\u00c5APL",  # not ASCII
        "\uff21\uff21\uff30\uff2c",  # full-width letters
        None,
        42,
        ["SPCX"],
    ],
)
def test_invalid_symbols(symbol):
    assert not rules.valid_symbol(symbol)


def test_shown_escapes_and_cuts_an_untrusted_value():
    assert rules.shown("X\nY") == "'X\\nY'"
    assert rules.shown("A" * 50) == "'" + "A" * 16 + "..."
    assert len(rules.shown("A" * 50)) == 20


def test_clean_text_removes_control_and_format_characters():
    assert rules.clean_text("a\x00b\x1bc\u200bd\u202ee\tf\n g", 50) == "abcde f g"
    assert rules.clean_text("  \x00 ", 50) is None
    assert rules.clean_text(None, 50) is None
    assert rules.clean_text(5, 50) is None
    assert rules.clean_text("x" * 20, 5) == "xxxxx"


# ── The listing-date file (section 5) ─────────────────────────────────────────


def test_the_committed_listing_dates_parse_with_the_shared_parser():
    """The file parses and keeps SPCX's entry. Not its whole contents: it is
    the procedure for every symbol Yahoo reuses (docs/trending.md, section
    5), and a test pinned to one line would turn CI and every trending run
    red the day the owner adds a second."""
    import datetime as dt

    listed = config_module.split_jump_acknowledged(
        config_module.TRENDING_LISTING_DATES_FILE
    )
    assert listed[("SPCX", dt.date(2026, 6, 12))] == (
        "the symbol was a Tuttle Capital ETF's before this listing"
    )
