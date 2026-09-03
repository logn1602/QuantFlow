"""
scripts/clean.py
----------------
Cross-platform cleanup for `make clean`.

Replaces the old shell target, which used `find` and `rm -rf` and therefore
failed on Windows PowerShell — the maintainer's primary platform.

Removes build/test debris and rotated logs. Deliberately does NOT touch
mlflow.db: it is the only record of past experiment runs and is not
regenerable. Pass --mlflow to include it.

Usage:
    python scripts/clean.py            # logs + caches
    python scripts/clean.py --dry-run  # show what would go
    python scripts/clean.py --mlflow   # also delete mlflow.db
"""

import argparse
import contextlib
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories removed wholesale, anywhere in the tree.
CACHE_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}

# Top-level directories removed wholesale.
ARTIFACT_DIRS = ["mlruns", "mlruns_artifacts"]

# Never walk into these while searching for caches.
SKIP = {".git", "venv", "venv-311", ".venv", "env", "node_modules"}


def human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def dir_size(p: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(p):
        for f in filenames:
            with contextlib.suppress(OSError):
                total += (Path(dirpath) / f).stat().st_size
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description="Clean logs and build artifacts")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be removed, delete nothing",
    )
    ap.add_argument(
        "--mlflow",
        action="store_true",
        help="Also delete mlflow.db (past experiment history)",
    )
    args = ap.parse_args()

    targets: list[tuple[Path, int]] = []

    # Rotated + current logs
    logs = ROOT / "logs"
    if logs.is_dir():
        for f in sorted(logs.glob("*.log*")):
            targets.append((f, f.stat().st_size))

    # Cache directories, skipping virtualenvs and .git
    for dirpath, dirnames, _ in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP]
        for d in list(dirnames):
            if d in CACHE_DIRS:
                p = Path(dirpath) / d
                targets.append((p, dir_size(p)))
                dirnames.remove(d)

    # MLflow artifact directories
    for name in ARTIFACT_DIRS:
        p = ROOT / name
        if p.is_dir():
            targets.append((p, dir_size(p)))

    if args.mlflow:
        db = ROOT / "mlflow.db"
        if db.is_file():
            targets.append((db, db.stat().st_size))

    if not targets:
        print("Nothing to clean.")
        return 0

    total = sum(s for _, s in targets)
    verb = "Would remove" if args.dry_run else "Removing"
    for p, size in targets:
        print(f"  {verb}: {p.relative_to(ROOT)}  ({human(size)})")

    if args.dry_run:
        print(f"\n{len(targets)} item(s), {human(total)} — dry run, nothing deleted.")
        return 0

    removed = 0
    for p, _ in targets:
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
            removed += 1
        except OSError as e:
            print(f"  skipped {p.relative_to(ROOT)}: {e}", file=sys.stderr)

    print(f"\nCleaned {removed} item(s), freed {human(total)}.")
    if not args.mlflow and (ROOT / "mlflow.db").is_file():
        print("Kept mlflow.db (experiment history). Use --mlflow to remove it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
