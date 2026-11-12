"""tests/unit/test_commands_named.py

Every command the code and the docs tell someone to run names something that
exists: a `python -m quantflow....` module, or a `python path/to/script.py`
script. The modules moved into the src/ package long ago, and 36 usage lines
and printed hints still named the scripts they used to be (`python
forecasting.py`, `python xgboost_model.py`), among them the seed's last line,
"You can now run: python scheduler/job_runner.py", which `make all` prints
on a first install (found in the release read-through, 2026-11-12). A module
docstring's title that names a file names its own.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "quantflow"
SCANNED = sorted(
    {
        *PACKAGE.rglob("*.py"),
        *(ROOT / "scripts").rglob("*.py"),
        *(ROOT / "scripts").rglob("*.sql"),
        *(ROOT / "sql").rglob("*.sql"),
        *(ROOT / "docs").glob("*.md"),
        *(ROOT / ".github" / "workflows").glob("*.yml"),
        ROOT / "README.md",
        ROOT / "Makefile",
        ROOT / "Dockerfile",
        ROOT / ".env.example",
    }
)
MODULE = re.compile(r"python3? -m (quantflow(?:\.\w+)+)")
SCRIPT = re.compile(r"python3? ((?:[\w.-]+/)*[\w.-]+\.py)\b")


def _mentions(pattern):
    found = []
    for path in SCANNED:
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            for match in pattern.finditer(line):
                found.append((f"{path.relative_to(ROOT)}:{number}", match.group(1)))
    return found


def _module_exists(dotted):
    path = ROOT / "src" / Path(*dotted.split("."))
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def test_every_module_named_to_run_exists():
    mentions = _mentions(MODULE)
    assert len(mentions) > 50  # the scan finds the commands at all
    missing = [(where, name) for where, name in mentions if not _module_exists(name)]
    assert missing == []


def test_every_script_named_to_run_exists():
    mentions = _mentions(SCRIPT)
    missing = [(where, name) for where, name in mentions if not (ROOT / name).is_file()]
    assert missing == []


def test_every_sql_file_named_to_run_exists():
    """psql's -f, as make setup and the migrations' headers give it."""
    mentions = _mentions(re.compile(r"-f \"?((?:[\w.-]+/)*[\w.-]+\.sql)\b"))
    assert mentions  # the scan finds the commands at all
    missing = [(where, name) for where, name in mentions if not (ROOT / name).is_file()]
    assert missing == []


def test_a_module_docstring_title_that_names_a_file_names_its_own():
    wrong = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
        title = (docstring or "").splitlines()[0].strip() if docstring else ""
        named = re.match(r"^([\w./-]+\.py)\b", title)
        if not named:
            continue
        own = path.relative_to(PACKAGE).as_posix()
        if not (own == named.group(1) or own.endswith("/" + named.group(1))):
            wrong.append((own, named.group(1)))
    assert wrong == []
