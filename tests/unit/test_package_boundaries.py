"""tests/unit/test_package_boundaries.py

evaluation/ never imports quantflow.models. Scoring code that can reach into
training internals is how an evaluation quietly starts measuring something
other than generalisation, so the rule is checked on the source, lazy imports
inside functions included, rather than trusted to review.
"""

import ast
from pathlib import Path

import quantflow.evaluation as evaluation_package

EVALUATION = Path(evaluation_package.__file__).parent
PACKAGE = "quantflow.evaluation"


def _imports(path: Path):
    """Every module a file imports, wherever the import statement sits."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative: resolve against this package
                base = PACKAGE.rsplit(".", node.level - 1)[0]
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            yield module
            yield from (f"{module}.{alias.name}" for alias in node.names)
        elif (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", getattr(node.func, "id", None))
            in {"import_module", "__import__"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            yield str(node.args[0].value)


def _is_models(module: str) -> bool:
    return module == "quantflow.models" or module.startswith("quantflow.models.")


def test_evaluation_never_imports_models():
    files = sorted(EVALUATION.glob("*.py"))
    assert files, "found no evaluation modules to check"
    offenders = [
        f"{path.name} imports {module}"
        for path in files
        for module in _imports(path)
        if _is_models(module)
    ]
    assert offenders == []


def test_the_check_sees_a_lazy_import(tmp_path):
    """The check must catch an import inside a function, the form the
    backtest used, or the test above proves nothing."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        "def f():\n    from quantflow.models.ensemble import BASE_MODELS\n",
        encoding="utf-8",
    )
    assert any(_is_models(module) for module in _imports(probe))


# ── The trending tier (docs/trending.md, section 13) ──────────────────────────
# Kept apart from the core by construction: no core module imports a trending
# one, so nothing the core or the evaluation runs can reach the tier's code,
# and the tier never imports the models or the evaluation. The dashboard's app
# is the one core file allowed in: it shows the Trending tab.

SRC = EVALUATION.parent
TRENDING_MODULES = {
    "quantflow.ingestion.trending_sources",
    "quantflow.ingestion.trending_rules",
    "quantflow.db.trending",
    "quantflow.pipelines.trending",
    "quantflow.features.trending_analytics",
    "quantflow.dashboard.trending_view",
}
ALLOWED_IN = {"quantflow.dashboard.app"}


def _module_name(path: Path) -> str:
    return "quantflow." + ".".join(path.relative_to(SRC).with_suffix("").parts)


def _is_trending(module: str) -> bool:
    return any(module == m or module.startswith(m + ".") for m in TRENDING_MODULES)


def test_no_core_module_imports_the_trending_tier():
    offenders = [
        f"{_module_name(path)} imports {module}"
        for path in sorted(SRC.rglob("*.py"))
        if _module_name(path) not in TRENDING_MODULES | ALLOWED_IN
        for module in _imports(path)
        if _is_trending(module)
    ]
    assert offenders == []


def test_the_trending_tier_never_imports_the_models_or_the_evaluation():
    files = [
        path for path in SRC.rglob("*.py") if _module_name(path) in TRENDING_MODULES
    ]
    assert files, "found no trending modules to check"
    offenders = [
        f"{_module_name(path)} imports {module}"
        for path in files
        for module in _imports(path)
        if _is_models(module) or module.startswith("quantflow.evaluation")
    ]
    assert offenders == []
