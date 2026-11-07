"""tests/unit/test_config_isolation.py

Runs after test_config.py, which reloads config under a patched environment in
every test. Its autouse fixture promises that a patched copy never leaks into
another module; this checks that promise from the outside.
"""

import os

import quantflow.config as config


def test_config_is_not_left_patched_by_test_config():
    raw = os.getenv("TICKERS", config.DEFAULT_TICKERS)
    loaded = config.TICKERS
    assert loaded == [t.strip().upper() for t in raw.split(",") if t.strip()]
