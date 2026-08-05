"""
tests/test_config.py
--------------------
Covers ticker parsing and config.validate().

config is a module of import-time globals, so each test reloads it under a
patched environment. The autouse fixture reloads it once more afterwards so a
patched copy never leaks into another test module.
"""

import importlib

import pytest

import config as config_module


@pytest.fixture(autouse=True)
def restore_config():
    yield
    importlib.reload(config_module)


def _reload_with(monkeypatch, **env):
    """Reload config with the given env vars applied.

    load_dotenv() does not override variables already present in the
    environment, so monkeypatch.setenv wins over .env — including when the
    value is an empty string.
    """
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(config_module)


# ── TICKERS parsing ───────────────────────────────────────────────────────────

def test_tickers_are_upcased(monkeypatch):
    cfg = _reload_with(monkeypatch, TICKERS="aapl,msft")
    assert cfg.TICKERS == ["AAPL", "MSFT"]


def test_tickers_whitespace_is_stripped(monkeypatch):
    cfg = _reload_with(monkeypatch, TICKERS="  AAPL ,  MSFT  , GOOGL ")
    assert cfg.TICKERS == ["AAPL", "MSFT", "GOOGL"]


def test_tickers_empty_entries_are_dropped(monkeypatch):
    cfg = _reload_with(monkeypatch, TICKERS="AAPL,,MSFT, ,")
    assert cfg.TICKERS == ["AAPL", "MSFT"]


def test_tickers_single_value(monkeypatch):
    cfg = _reload_with(monkeypatch, TICKERS="tsla")
    assert cfg.TICKERS == ["TSLA"]


def test_tickers_all_blank_yields_empty_list(monkeypatch):
    cfg = _reload_with(monkeypatch, TICKERS=" , , ")
    assert cfg.TICKERS == []


def test_tickers_mixed_case_and_padding_together(monkeypatch):
    cfg = _reload_with(monkeypatch, TICKERS=" nVdA , ,jpm ")
    assert cfg.TICKERS == ["NVDA", "JPM"]


# ── validate() ────────────────────────────────────────────────────────────────

def _all_present():
    return {
        "DB_PASSWORD":           "sekret",
        "ALPHA_VANTAGE_API_KEY": "av-key",
        "NEWS_API_KEY":          "news-key",
        "TICKERS":               "AAPL",
    }


def test_validate_true_when_everything_is_set(monkeypatch):
    cfg = _reload_with(monkeypatch, **_all_present())
    assert cfg.validate() is True


@pytest.mark.parametrize("missing", [
    "DB_PASSWORD",
    "ALPHA_VANTAGE_API_KEY",
    "NEWS_API_KEY",
])
def test_validate_false_when_a_required_var_is_blank(monkeypatch, missing):
    env = _all_present()
    env[missing] = ""
    cfg = _reload_with(monkeypatch, **env)
    assert cfg.validate() is False


def test_validate_false_when_tickers_list_is_empty(monkeypatch):
    env = _all_present()
    env["TICKERS"] = " , "
    cfg = _reload_with(monkeypatch, **env)
    assert cfg.TICKERS == []
    assert cfg.validate() is False


def test_validate_reports_every_missing_var_not_just_the_first(monkeypatch, capsys):
    env = _all_present()
    env["DB_PASSWORD"]  = ""
    env["NEWS_API_KEY"] = ""
    cfg = _reload_with(monkeypatch, **env)

    assert cfg.validate() is False
    warnings = capsys.readouterr().out
    assert "DB_PASSWORD" in warnings
    assert "NEWS_API_KEY" in warnings


# ── Other settings ────────────────────────────────────────────────────────────

def test_database_url_is_assembled_from_parts(monkeypatch):
    cfg = _reload_with(
        monkeypatch,
        DB_USER="quant", DB_PASSWORD="pw", DB_HOST="db.example.com",
        DB_PORT="6543", DB_NAME="flow",
    )
    assert cfg.DATABASE_URL == "postgresql://quant:pw@db.example.com:6543/flow"


def test_numeric_settings_are_coerced_to_int(monkeypatch):
    cfg = _reload_with(monkeypatch, DB_PORT="6543", FETCH_INTERVAL_MINUTES="30")
    assert cfg.DB_PORT == 6543
    assert cfg.FETCH_INTERVAL_MINUTES == 30
