"""
scripts/diagnostics/run_1a_diagnostics.py
-----------------------------------------
Run the four daily-close diagnostics (1a_diagnostics.sql) in one read-only
transaction and write the results as Markdown tables.

Read-only three ways:
  - every statement in 1a_diagnostics.sql is a SELECT;
  - the transaction is opened READ ONLY, so Postgres rejects any write, and
    the script checks transaction_read_only is on before running anything;
  - the transaction is rolled back at the end, whatever happens.

The connection is the project's own: quantflow.db.connection.get_engine(),
configured from .env. Nothing here prints the URL, host, user or password, and
an error message is printed with those values masked.

A non-local database is refused unless --live is given, so a trial run against
a throwaway local cluster cannot reach the live one by mistake.

Usage, from the repo root:
    uv run python scripts/diagnostics/run_1a_diagnostics.py --out results.md
    uv run python scripts/diagnostics/run_1a_diagnostics.py --live --out results.md

The first refuses anything but a local database; --live allows the one in .env.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from quantflow import config  # noqa: F401  (tests patch its settings through here)
from quantflow.db.readonly import is_local_database, masked, read_only_connection

SQL_FILE = Path(__file__).with_name("1a_diagnostics.sql")


def queries() -> dict[str, str]:
    """The SQL file split on its "-- @query <name>" markers."""
    text = SQL_FILE.read_text(encoding="utf-8")
    parts = re.split(r"^-- @query (.+)$", text, flags=re.MULTILINE)
    return {
        name.strip(): sql.strip()
        for name, sql in zip(parts[1::2], parts[2::2], strict=True)
    }


def as_markdown(df: pd.DataFrame) -> str:
    def cell(value) -> str:
        return (
            ""
            if value is None or (isinstance(value, float) and pd.isna(value))
            else str(value)
        )

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join(lines)


def run(out: Path) -> None:
    results = {}
    # One read-only snapshot transaction for all four queries (db.readonly).
    with read_only_connection() as conn:
        # Each query goes through the same connection's own cursor with no
        # parameters, so it reaches Postgres exactly as written (a "%" in the
        # SQL is an operator, not a placeholder).
        dbapi = conn.connection.dbapi_connection
        if dbapi is None:
            raise RuntimeError("no DBAPI connection to run the queries on")
        cursor = dbapi.cursor()
        try:
            for name, sql in queries().items():
                cursor.execute(sql)
                columns = [column[0] for column in cursor.description]
                results[name] = pd.DataFrame(
                    cursor.fetchall(), columns=columns, dtype=object
                )
                print(f"  {name}: {len(results[name])} rows")
        finally:
            cursor.close()

    sections = [f"### {name}\n\n{as_markdown(df)}\n" for name, df in results.items()]
    out.write_text("\n".join(sections), encoding="utf-8")
    print(f"Results written to {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the 1a diagnostics read-only")
    parser.add_argument(
        "--live", action="store_true", help="allow the non-local database in .env"
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="results file (Markdown)"
    )
    args = parser.parse_args(argv)

    local = is_local_database()
    if not local and not args.live:
        print(
            "Refusing: .env points at a non-local database; pass --live.",
            file=sys.stderr,
        )
        return 2
    print(
        "Target: "
        + ("a local database" if local else "the live database configured in .env")
        + "; one read-only transaction, rolled back at the end"
    )
    try:
        run(args.out)
    except Exception as e:
        print(f"Failed: {type(e).__name__}: {masked(str(e))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
