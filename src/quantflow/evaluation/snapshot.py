"""Snapshot files: the data every evaluation run reads (docs/evaluation.md,
section 6).

One read-only export of the live database is written once, to a git-ignored
directory, with a manifest holding each file's SHA-256. Every run reads only
the snapshot and records the manifest's digest, so two runs on the same
digest saw the same data, and repeated runs cost the database nothing.

A snapshot is never overwritten, and a file that no longer matches its hash
is refused.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

FORMAT = 1
MANIFEST = "manifest.json"
FILES = {
    "prices": "prices.csv",
    "articles": "articles.csv",
    "stored_indicators": "stored_indicators.csv",
    "stored_anomalies": "stored_anomalies.csv",
}
_TIMESTAMPS = {"articles": ["published_at", "inserted_at"]}
_DATES = {
    "prices": ["date"],
    "stored_indicators": ["date"],
    "stored_anomalies": ["date"],
}


class SnapshotError(RuntimeError):
    """A snapshot is missing, altered, or unusable for the run asked of it."""


@dataclass(frozen=True)
class Snapshot:
    directory: Path
    digest: str  # SHA-256 of the manifest, which holds every file's hash
    as_of: pd.Timestamp
    tickers: list[str]
    prices: dict[str, pd.DataFrame]  # daily bars per ticker, date order
    articles: pd.DataFrame
    stored_indicators: pd.DataFrame
    stored_anomalies: pd.DataFrame
    built_dates: dict[str, list[pd.Timestamp]]  # dates built from 15-min bars

    @property
    def trading_dates(self) -> pd.DatetimeIndex:
        """The tickers' trading dates, which must be the same for all of
        them so the origins line up across tickers."""
        dates = {t: tuple(df["date"]) for t, df in self.prices.items()}
        if len(set(dates.values())) != 1:
            raise SnapshotError("the tickers in the snapshot have different dates")
        return pd.DatetimeIndex(next(iter(dates.values())))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_snapshot(
    directory: Path,
    *,
    as_of,
    exported_at,
    prices: dict[str, pd.DataFrame],
    articles: pd.DataFrame,
    stored_indicators: pd.DataFrame,
    stored_anomalies: pd.DataFrame,
    built_dates: dict[str, list],
) -> Path:
    """Write a new snapshot directory; refuses one that already exists."""
    directory = Path(directory)
    if directory.exists():
        raise SnapshotError(f"{directory} exists; a snapshot is never overwritten")
    directory.mkdir(parents=True)
    tables = {
        "prices": pd.concat(
            [df.assign(ticker=t) for t, df in prices.items()], ignore_index=True
        )[["ticker", "date", "open", "high", "low", "close", "volume"]],
        "articles": articles,
        "stored_indicators": stored_indicators,
        "stored_anomalies": stored_anomalies,
    }
    files = {}
    for name, table in tables.items():
        path = directory / FILES[name]
        table.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f%z")
        files[name] = {"file": FILES[name], "sha256": _sha256(path), "rows": len(table)}
    manifest = {
        "format": FORMAT,
        "as_of": pd.Timestamp(as_of).date().isoformat(),
        "exported_at": pd.Timestamp(exported_at).isoformat(),
        "tickers": sorted(prices),
        "files": files,
        "built_from_15min": {
            t: [pd.Timestamp(d).date().isoformat() for d in dates]
            for t, dates in sorted(built_dates.items())
        },
    }
    (directory / MANIFEST).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return directory


def read_snapshot(directory: Path) -> Snapshot:
    """Read a snapshot, refusing any file whose hash no longer matches."""
    directory = Path(directory)
    manifest_path = directory / MANIFEST
    if not manifest_path.exists():
        raise SnapshotError(f"no {MANIFEST} in {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise SnapshotError(f"unknown snapshot format {manifest.get('format')!r}")
    tables = {}
    for name, entry in manifest["files"].items():
        path = directory / entry["file"]
        if not path.exists() or _sha256(path) != entry["sha256"]:
            raise SnapshotError(f"{path.name} is missing or no longer matches its hash")
        table = pd.read_csv(path)
        for column in _DATES.get(name, []):
            table[column] = pd.to_datetime(table[column].str[:10])
        for column in _TIMESTAMPS.get(name, []):
            table[column] = pd.to_datetime(table[column], utc=True, format="ISO8601")
        tables[name] = table
    prices = {
        ticker: group.drop(columns="ticker").sort_values("date").reset_index(drop=True)
        for ticker, group in tables["prices"].groupby("ticker", sort=True)
    }
    return Snapshot(
        directory=directory,
        digest=_sha256(manifest_path),
        as_of=pd.Timestamp(manifest["as_of"]),
        tickers=manifest["tickers"],
        prices=prices,
        articles=tables["articles"],
        stored_indicators=tables["stored_indicators"],
        stored_anomalies=tables["stored_anomalies"],
        built_dates={
            t: [pd.Timestamp(d) for d in dates]
            for t, dates in manifest["built_from_15min"].items()
        },
    )


def check_built_dates(snapshot: Snapshot, used_dates) -> None:
    """Refuse a run whose origins or targets include a date built from
    15-min bars rather than an official daily close."""
    used = set(pd.DatetimeIndex(used_dates))
    clashes = {
        t: sorted(d.date().isoformat() for d in dates if d in used)
        for t, dates in snapshot.built_dates.items()
    }
    clashes = {t: d for t, d in clashes.items() if d}
    if clashes:
        raise SnapshotError(
            f"dates built from 15-min bars fall at an origin or target: {clashes}"
        )
