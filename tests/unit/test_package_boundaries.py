"""tests/unit/test_package_boundaries.py

evaluation/ never imports quantflow.models. Scoring code that can reach into
training internals is how an evaluation quietly starts measuring something
other than generalisation, so the rule is checked on the source, lazy imports
inside functions included, rather than trusted to review.
"""

import ast
from pathlib import Path

import pytest

import quantflow.evaluation as evaluation_package

EVALUATION = Path(evaluation_package.__file__).parent
SRC = EVALUATION.parent


def _module_name(path: Path) -> str:
    return "quantflow." + ".".join(path.relative_to(SRC).with_suffix("").parts)


def _package_of(path: Path) -> str:
    """The package a file's relative imports start from: its own, for a
    package's __init__.py; the one it sits in, for any other module."""
    name = _module_name(path)
    return (
        name.removesuffix(".__init__")
        if path.name == "__init__.py"
        else (name.rsplit(".", 1)[0])
    )


def _imports(path: Path, package: str | None = None):
    """Every module a file imports, wherever the import statement sits. A
    relative import is resolved against the file's own package (package,
    when the file is not under src, as a probe is not)."""
    package = package or _package_of(path)
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative: one level is the package itself
                base = package.rsplit(".", node.level - 1)[0]
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
    assert any(_is_models(module) for module in _imports(probe, "quantflow.evaluation"))


@pytest.mark.parametrize(
    ("package", "source", "expected"),
    [
        ("quantflow.evaluation", "from ..models import ensemble\n", "quantflow.models"),
        ("quantflow.evaluation", "from . import metrics\n", "quantflow.evaluation"),
        (
            "quantflow.ingestion",
            "from .trending_rules import valid_symbol\n",
            "quantflow.ingestion.trending_rules",
        ),
        (
            "quantflow.dashboard",
            "from ..pipelines import trending\n",
            "quantflow.pipelines.trending",
        ),
    ],
)
def test_a_relative_import_resolves_against_the_files_own_package(
    tmp_path, package, source, expected
):
    """Resolved against quantflow.evaluation whatever the file, a trending
    module's relative import of another would have named a module that
    does not exist, and passed every check."""
    probe = tmp_path / "probe.py"
    probe.write_text(source, encoding="utf-8")
    assert expected in set(_imports(probe, package))


def test_each_file_resolves_against_its_own_package():
    sources = SRC / "ingestion" / "trending_sources.py"
    assert _package_of(sources) == "quantflow.ingestion"
    assert _package_of(SRC / "evaluation" / "__init__.py") == "quantflow.evaluation"


# ── The trending tier (docs/trending.md, section 13) ──────────────────────────
# Kept apart from the core by construction: no core module imports a trending
# one, so nothing the core or the evaluation runs can reach the tier's code,
# and the tier never imports the models or the evaluation. The dashboard's app
# is the one core file allowed in: it shows the Trending tab.

TRENDING_MODULES = {
    "quantflow.ingestion.trending_sources",
    "quantflow.ingestion.trending_rules",
    "quantflow.db.trending",
    "quantflow.pipelines.trending",
    "quantflow.features.trending_analytics",
    "quantflow.dashboard.trending_view",
}
ALLOWED_IN = {"quantflow.dashboard.app"}


def _is_trending(module: str) -> bool:
    return any(module == m or module.startswith(m + ".") for m in TRENDING_MODULES)


@pytest.mark.trending
def test_the_trending_modules_are_every_trending_file():
    """A trending module left off the list would be checked as core, and
    one the list names that is gone would check nothing."""
    files = {_module_name(path) for path in SRC.rglob("*trending*.py")}
    assert files == TRENDING_MODULES


@pytest.mark.trending
def test_no_core_module_imports_the_trending_tier():
    offenders = [
        f"{_module_name(path)} imports {module}"
        for path in sorted(SRC.rglob("*.py"))
        if _module_name(path) not in TRENDING_MODULES | ALLOWED_IN
        for module in _imports(path)
        if _is_trending(module)
    ]
    assert offenders == []


@pytest.mark.trending
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
