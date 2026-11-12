"""tests/unit/test_dockerfile.py

The Dockerfile, read as text. requirements.txt ends with ".", the quantflow
package itself (Streamlit Cloud installs nothing else), and "." can only be
installed once the source is in the image. The Dockerfile installed
requirements.txt with only that file copied, so `docker build` stopped at
that step and the README's Docker Quickstart could not work (found in the
release read-through, 2026-11-12).
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
REQUIREMENTS = (ROOT / "requirements.txt").read_text(encoding="utf-8")


def _instructions():
    """(instruction, argument text) for each line of the Dockerfile, with
    continuation lines joined and comments dropped."""
    joined = re.sub(r"\\\n", " ", DOCKERFILE)
    found = []
    for line in joined.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            word, _, rest = line.partition(" ")
            found.append((word.upper(), " ".join(rest.split())))
    return found


def test_requirements_txt_still_ends_with_the_package_itself():
    """What the Dockerfile has to allow for; if this changes, so can it."""
    lines = [
        line.strip()
        for line in REQUIREMENTS.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert lines[-1] == "."


def test_nothing_installs_the_package_before_its_source_is_copied():
    """Every install of requirements.txt, or of ".", comes after the COPY
    that brings in pyproject.toml and src/, unless it leaves the "." line
    out."""
    source_copied = False
    for instruction, argument in _instructions():
        if instruction == "COPY" and argument.split()[0] == ".":
            source_copied = True
        if instruction != "RUN" or source_copied:
            continue
        for command in re.split(r"&&|;", argument):
            words = command.split()
            if "install" not in words:
                continue
            if "-r" in words:
                target = words[words.index("-r") + 1]
                assert target != "requirements.txt", (
                    f"installs requirements.txt, whose last line is '.', "
                    f"before the source is copied: {command.strip()}"
                )
            assert "." not in words, f"installs '.' before the source: {command}"


def test_the_package_is_installed_once_the_source_is_in_the_image():
    instructions = _instructions()
    copy = next(
        i
        for i, (instruction, argument) in enumerate(instructions)
        if instruction == "COPY" and argument.split()[0] == "."
    )
    installs = [
        argument
        for instruction, argument in instructions[copy:]
        if instruction == "RUN" and re.search(r"pip install .*(\s\.|-e \.)", argument)
    ]
    assert installs, "the package is never installed after COPY . ."


def test_the_header_names_what_the_image_serves_and_no_old_deployment():
    header = DOCKERFILE.split("\nFROM ", 1)[0]
    assert "Cloud Run" not in header and "GCP" not in header
    assert "Streamlit" in header


def test_the_third_party_list_leaves_the_package_line_out():
    """The step's own grep pattern, applied to requirements.txt: what it
    installs before the source is copied must not hold the "." line."""
    run = next(
        argument
        for instruction, argument in _instructions()
        if instruction == "RUN" and "-r /tmp/third-party.txt" in argument
    )
    match = re.search(
        r"grep -vx '([^']+)' requirements\.txt > /tmp/third-party\.txt", run
    )
    assert match, run
    kept = [
        line
        for line in REQUIREMENTS.splitlines()
        if not re.fullmatch(match.group(1), line)
    ]
    assert "." not in {line.strip() for line in kept}
    assert len(kept) == len(REQUIREMENTS.splitlines()) - 1


DOCKERIGNORE = (ROOT / ".dockerignore").read_text(encoding="utf-8")


def _docker_pattern(pattern):
    """A .dockerignore pattern as a regular expression over a path relative
    to the context, as Docker reads it: Go's filepath.Match on the cleaned
    pattern, ** for any number of directories, and no match below the root
    unless the pattern says so (unlike .gitignore)."""
    pattern = pattern.strip().rstrip("/")
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "[":
            end = pattern.index("]", i)
            out.append(pattern[i : end + 1])
            i = end + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out))


def _excluded(path):
    """Whether .dockerignore keeps path out of the build context: the last
    pattern that matches it or a directory above it decides, ! re-including."""
    parts = path.split("/")
    candidates = ["/".join(parts[: n + 1]) for n in range(len(parts))]
    decision = False
    for line in DOCKERIGNORE.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        regex = _docker_pattern(line[1:] if negate else line)
        if any(regex.fullmatch(candidate) for candidate in candidates):
            decision = not negate
    return decision


def test_the_build_context_leaves_out_what_git_leaves_out():
    """COPY . . copies the whole working tree. Once docker build got past
    the install step (above), an image built from the owner's checkout
    would have carried the evaluation's export of the live database, the
    local review reports and notes, and the tools' settings, none of which
    is in git (found in the full-diff review, 2026-11-12)."""
    for path in (
        ".env",
        ".claude/settings.local.json",
        ".gstack/browse-audit.jsonl",
        "WORK_BRIEF.md",
        "WORK_PROGRESS.md",
        "eval_snapshots/2026-11-06/prices.parquet",
        "eval_runs/P4b/cells.csv",
        "mlflow.db",
        "mlruns/0/meta.yaml",
        "mlruns_artifacts/1/a.csv",
        "logs/pipeline.log",
        ".venv/pyvenv.cfg",
        "venv-311/pyvenv.cfg",
        "build/lib/quantflow/config.py",
        "src/quantflow.egg-info/PKG-INFO",
        "src/quantflow/__pycache__/config.cpython-311.pyc",
        ".git/HEAD",
        ".coverage",
        ".mypy_cache/3.11/cache.db",
        ".pytest_cache/v/cache/lastfailed",
        ".ruff_cache/CACHEDIR.TAG",
        ".streamlit/secrets.toml",
    ):
        assert _excluded(path), path


def test_the_build_context_keeps_what_the_image_needs():
    """pyproject.toml names README.md and LICENSE, so the package build
    needs them; the dashboard reads its theme, the committed lists and the
    published accuracy table."""
    for path in (
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "requirements.txt",
        "src/quantflow/config.py",
        "src/quantflow/dashboard/app.py",
        "src/quantflow/dashboard/published_accuracy.csv",
        ".streamlit/config.toml",
        "config/split_jump_acknowledged.txt",
        "config/trending_listing_dates.txt",
    ):
        assert not _excluded(path), path
