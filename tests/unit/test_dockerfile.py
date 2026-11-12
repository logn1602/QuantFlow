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
