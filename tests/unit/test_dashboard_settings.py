"""tests/unit/test_dashboard_settings.py

The dashboard's page settings, read as text: app.py connects to the database
when imported, so it cannot be imported here (docs/architecture.md, Known
problem 5). The theme it is styled for, every loader cached for one shared
lifetime that still picks up the daily refresh (dashboard/caching.py, tested
in test_dashboard_caching.py), no public control that clears the cache, and
no model crowned above the others.
"""

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "src" / "quantflow" / "dashboard" / "app.py").read_text(
    encoding="utf-8"
)
TREE = ast.parse(SOURCE)


def test_the_dashboard_sets_the_dark_theme_it_is_styled_for():
    """Its charts use Plotly's dark template and its cards a dark background;
    following a viewer's light theme hid the white "Actual" line."""
    config = tomllib.loads(
        (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    )
    assert config["theme"]["base"] == "dark"


def test_every_loader_is_cached_for_the_same_lifetime_except_when_empty():
    """The price loader was a bare import, so every rerun read the ticker's
    prices from the database again; the latest closes were read uncached
    too. Then every loader kept an empty result for the hour as well: one
    that ran while the daily run replaced the forecasts showed none to every
    visitor until it expired."""
    loaders = {
        node.name: node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("load_")
    }
    assert {"load_prices", "load_latest_closes", "load_forecasts"} <= set(loaders)
    for name, node in loaders.items():
        decorators = [ast.unparse(d) for d in node.decorator_list]
        assert decorators == ["cached"], name
    assert "st.cache_data" not in SOURCE


def test_no_public_control_clears_the_cache():
    """A visitor's click cleared the cache for every visitor, forcing fresh
    reads of the live database."""
    assert "cache_data.clear" not in SOURCE


def test_the_sentiment_shown_is_the_news_the_models_count():
    """Both sentiment reads keep to the sources the features count
    (features.news.NEWS_SOURCES): RSS only, from 2026-11-09."""
    assert SOURCE.count("source = ANY(:sources)") == 2
    assert "NEWS_SOURCES" in SOURCE


def test_no_model_is_crowned():
    """The ensemble had a star and a thicker, solid line; on the measured 7-day
    error it does not earn either."""
    assert "⭐" not in SOURCE
    assert "is_ensemble" not in SOURCE
