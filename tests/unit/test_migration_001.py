"""tests/unit/test_migration_001.py

sql/migrations/001_dedupe_anomalies.sql is obsolete: checked read-only on
2026-11-12, the live anomalies table held 771 rows and 771 distinct
(ticker, ts), so there was nothing to delete. The file stays as the record;
it must refuse to run before any statement that changes the table.
tests/integration/test_writes_on_postgres.py runs it on Postgres.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIGRATION_001 = (ROOT / "sql" / "migrations" / "001_dedupe_anomalies.sql").read_text(
    encoding="utf-8"
)


def _statements(sql):
    """The script's statements in order, comments dropped; a DO block, with
    the semicolons inside it, counts as one."""
    body = "\n".join(line.split("--", 1)[0] for line in sql.splitlines())
    found = []
    for match in re.finditer(r"DO \$\$.*?\$\$;|[^;]+;", body, re.DOTALL):
        statement = " ".join(match.group(0).split())
        if statement != ";":
            found.append(statement)
    return found


def test_it_says_at_the_top_that_it_is_obsolete_and_why():
    banner = MIGRATION_001.split("\n-- 001_dedupe_anomalies.sql\n", 1)[0]
    assert "OBSOLETE (2026-11-12): not to be run." in banner
    assert "771 rows" in banner and "771 distinct (ticker, ts)" in banner


def test_it_raises_inside_the_transaction_before_any_change():
    """Inside the transaction, so a psql run that carries on past an error
    finds the transaction aborted and changes nothing either."""
    statements = _statements(MIGRATION_001)
    assert statements[0] == "BEGIN;"
    assert statements[1].startswith("DO $$ BEGIN RAISE EXCEPTION")
    assert "is obsolete" in statements[1]
    writes = [
        n
        for n, s in enumerate(statements)
        if re.match(r"(DELETE|ALTER|INSERT|UPDATE)", s)
    ]
    assert writes and min(writes) > 1, statements[:4]
