"""tests/unit/test_dashboard_settings.py

The dashboard's page settings, read as text: app.py connects to the database
when imported, so it cannot be imported here (docs/architecture.md, Known
problem 5). The theme it is styled for, every loader cached for one shared
lifetime that still picks up the daily refresh, no public control that
clears the cache, and no model crowned above the others.
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


def test_every_loader_is_cached_for_the_same_lifetime():
    """The price loader was a bare import, so every rerun read the ticker's
    prices from the database again; the latest closes were read uncached
    too."""
    loaders = {
        node.name: node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("load_")
    }
    assert {"load_prices", "load_latest_closes"} <= set(loaders)
    for name, node in loaders.items():
        decorators = [ast.unparse(d) for d in node.decorator_list]
        assert "st.cache_data(ttl=CACHE_SECONDS)" in decorators, name


def test_the_cache_lifetime_picks_up_the_daily_refresh_within_an_hour():
    lifetime = next(
        node.value
        for node in TREE.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "CACHE_SECONDS"
    )
    assert isinstance(lifetime, ast.Constant)
    assert 0 < lifetime.value <= 3600


def test_no_public_control_clears_the_cache():
    """A visitor's click cleared the cache for every visitor, forcing fresh
    reads of the live database."""
    assert "cache_data.clear" not in SOURCE


def test_no_model_is_crowned():
    """The ensemble had a star and a thicker, solid line; on the measured 7-day
    error it does not earn either."""
    assert "⭐" not in SOURCE
    assert "is_ensemble" not in SOURCE
