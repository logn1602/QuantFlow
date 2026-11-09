"""
tests/test_config.py
--------------------
Covers ticker parsing and config.validate().

config is a module of import-time globals, so each test reloads it under a
patched environment. The autouse fixture reloads it once more afterwards so a
patched copy never leaks into another test module.
"""

import importlib
import re
from pathlib import Path

import pytest

import quantflow.config as config_module


@pytest.fixture(autouse=True)
def restore_config(monkeypatch):
    yield
    # Undo the test's environment first. Fixture teardown runs in reverse
    # setup order, and conftest's autouse fixtures set monkeypatch up before
    # this one, so without the undo the reload below still saw the patched
    # environment and left that copy of config behind for every later module.
    monkeypatch.undo()
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
        "DB_PASSWORD": "sekret",
        "ALPHA_VANTAGE_API_KEY": "av-key",
        "NEWS_API_KEY": "news-key",
        "TICKERS": "AAPL",
    }


def test_validate_true_when_everything_is_set(monkeypatch):
    cfg = _reload_with(monkeypatch, **_all_present())
    assert cfg.validate() is True


@pytest.mark.parametrize(
    "missing",
    [
        "DB_PASSWORD",
        "ALPHA_VANTAGE_API_KEY",
        "NEWS_API_KEY",
    ],
)
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
    env["DB_PASSWORD"] = ""
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
        DB_USER="quant",
        DB_PASSWORD="pw",
        DB_HOST="db.example.com",
        DB_PORT="6543",
        DB_NAME="flow",
    )
    assert cfg.DATABASE_URL == "postgresql://quant:pw@db.example.com:6543/flow"


def test_the_read_only_role_is_read_from_the_environment(monkeypatch):
    cfg = _reload_with(
        monkeypatch, DB_READONLY_USER="quantflow_readonly.ref", DB_READONLY_PASSWORD="r"
    )
    assert cfg.DB_READONLY_USER == "quantflow_readonly.ref"
    assert cfg.DB_READONLY_PASSWORD == "r"


@pytest.mark.parametrize(
    "name, default", [("DB_PORT", 5432), ("FETCH_INTERVAL_MINUTES", 15)]
)
def test_a_blank_numeric_setting_falls_back_to_its_default(monkeypatch, name, default):
    """`KEY=` with no value is how .env.example writes optional settings;
    int('') at import took down every entry point, dashboard included."""
    value = getattr(_reload_with(monkeypatch, **{name: ""}), name)
    assert value == default


def test_numeric_settings_are_coerced_to_int(monkeypatch):
    cfg = _reload_with(monkeypatch, DB_PORT="6543", FETCH_INTERVAL_MINUTES="30")
    assert cfg.DB_PORT == 6543
    assert cfg.FETCH_INTERVAL_MINUTES == 30


@pytest.mark.parametrize("raw", ["", "15%", "fifteen"])
def test_a_bad_split_guard_threshold_does_not_break_the_config_import(monkeypatch, raw):
    """`KEY=` with no value is how .env.example writes optional settings, and
    "15%" is an easy slip. Parsed at import, either took down every entry
    point, the dashboard included. split_guard parses it instead."""
    kept = _reload_with(monkeypatch, SPLIT_GUARD_THRESHOLD=raw).SPLIT_GUARD_THRESHOLD
    assert kept == raw


def _reload_with_defaults(monkeypatch, *unset):
    """Reload config with the given variables absent from the environment
    and from .env, so the code defaults are what gets read."""
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    for key in unset:
        monkeypatch.delenv(key, raising=False)
    return importlib.reload(config_module)


def test_split_guard_threshold_defaults_to_fifteen_percent(monkeypatch):
    from quantflow.ingestion import split_guard

    _reload_with_defaults(monkeypatch, "SPLIT_GUARD_THRESHOLD")
    assert split_guard.configured_threshold() == 0.15


COMMITTED_ACKNOWLEDGED_LIST = (
    Path(__file__).resolve().parents[2] / "config" / "split_jump_acknowledged.txt"
)


def test_the_acknowledged_moves_list_is_the_committed_file(monkeypatch):
    """Resolved from the package rather than the working directory, so a run
    from anywhere reads the list committed with the code. Reloaded, as at
    import: tests/conftest.py points every test at an empty list."""
    cfg = _reload_with(monkeypatch)
    assert COMMITTED_ACKNOWLEDGED_LIST == cfg.SPLIT_JUMP_ACKNOWLEDGED_FILE


def test_the_committed_acknowledged_moves_list_parses():
    """A bad edit fails here, in CI, before the daily refresh trips on it."""
    acknowledged = config_module.split_jump_acknowledged(COMMITTED_ACKNOWLEDGED_LIST)
    assert all(reason.strip() for reason in acknowledged.values())


# ── The default ticker list ───────────────────────────────────────────────────
#
# The tests above set TICKERS explicitly, as CI does (AAPL,MSFT), so they say
# nothing about the fallback. The fallback used to be four tickers while
# .env.example, the daily workflow and the README all track eight.

ROOT = Path(__file__).resolve().parents[2]


def _env_example_tickers():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    match = re.search(r"^TICKERS=(.+)$", text, re.MULTILINE)
    assert match, "no TICKERS line in .env.example"
    return match.group(1).strip().split(",")


def test_the_default_ticker_list_is_the_tracked_eight(monkeypatch):
    cfg = _reload_with_defaults(monkeypatch, "TICKERS")
    assert cfg.TICKERS == [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "TSLA",
        "META",
        "JPM",
    ]


def test_the_default_ticker_list_matches_env_example(monkeypatch):
    default = _reload_with_defaults(monkeypatch, "TICKERS").TICKERS
    assert default == _env_example_tickers()


def _readme_default_tickers():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"^\| `TICKERS` \|.*defaults to `([^`]+)`", text, re.MULTILINE)
    assert match, "no TICKERS row with a default in the README's settings table"
    return match.group(1).split(",")


def test_the_readme_documents_the_default_ticker_list(monkeypatch):
    default = _reload_with_defaults(monkeypatch, "TICKERS").TICKERS
    assert default == _readme_default_tickers()


# ── require_or_exit() ─────────────────────────────────────────────────────────
#
# The hard gate. validate() only warns; this is what actually stops a pipeline
# from starting against a broken .env. It is deliberately narrower than
# validate(): the API keys are optional at runtime, so treating them as fatal
# would kill a run that could have completed on the RSS fallback.


def test_require_or_exit_passes_when_the_fatal_vars_are_set(monkeypatch):
    cfg = _reload_with(monkeypatch, DB_PASSWORD="pw", TICKERS="AAPL")
    cfg.require_or_exit()  # must not raise


def test_require_or_exit_rejects_a_missing_db_password(monkeypatch):
    cfg = _reload_with(monkeypatch, DB_PASSWORD="", TICKERS="AAPL")
    with pytest.raises(SystemExit) as exc:
        cfg.require_or_exit()
    assert exc.value.code == 1


def test_require_or_exit_rejects_an_empty_ticker_list(monkeypatch):
    """An empty list means every loop body is skipped and the pipeline
    'succeeds' having done nothing."""
    cfg = _reload_with(monkeypatch, DB_PASSWORD="pw", TICKERS=" , ")
    with pytest.raises(SystemExit) as exc:
        cfg.require_or_exit()
    assert exc.value.code == 1


def test_require_or_exit_writes_the_reason_to_stderr(monkeypatch, capsys):
    cfg = _reload_with(monkeypatch, DB_PASSWORD="", TICKERS="AAPL")
    with pytest.raises(SystemExit):
        cfg.require_or_exit()
    err = capsys.readouterr().err
    assert "DB_PASSWORD" in err
    assert ".env" in err


def test_require_or_exit_reports_every_fatal_var_not_just_the_first(
    monkeypatch, capsys
):
    cfg = _reload_with(monkeypatch, DB_PASSWORD="", TICKERS=" , ")
    with pytest.raises(SystemExit):
        cfg.require_or_exit()
    err = capsys.readouterr().err
    assert "DB_PASSWORD" in err
    assert "TICKERS" in err


@pytest.mark.parametrize("optional", ["ALPHA_VANTAGE_API_KEY", "NEWS_API_KEY"])
def test_require_or_exit_tolerates_missing_api_keys(monkeypatch, optional):
    """Neither key is fatal. Alpha Vantage is unused by the default ingestion
    path, and sentiment degrades to the RSS feed when NEWS_API_KEY is absent.
    Exiting for either would be worse than the missing data."""
    env = _all_present()
    env[optional] = ""
    cfg = _reload_with(monkeypatch, **env)
    cfg.require_or_exit()  # must not raise


def test_require_or_exit_message_is_ascii_only(monkeypatch, capsys):
    """These lines reach Windows consoles and CI logs where a non-UTF-8 code
    page turns an em dash into mojibake."""
    cfg = _reload_with(monkeypatch, DB_PASSWORD="", TICKERS="AAPL")
    with pytest.raises(SystemExit):
        cfg.require_or_exit()
    err = capsys.readouterr().err
    assert err.isascii(), f"non-ASCII in stderr output: {err!r}"
