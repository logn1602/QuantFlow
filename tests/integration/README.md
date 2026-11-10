# Integration tests

Tests that need a real PostgreSQL: the write-path behaviour the unit suite can
only fake. Mark them `@pytest.mark.integration` so the default suite and CI
skip them.

They run only against a disposable local database passed with `--pg-url`. The
fixture refuses any URL that would connect anywhere but this machine, and
without the option every test is skipped:

```
uv run pytest -m integration --pg-url postgresql://user@127.0.0.1:5432/scratch
```

Each test creates its own schema from `sql/schema.sql`,
`sql/schema_sentiment.sql` and `sql/schema_trending.sql` (the `pg` fixture in
`tests/conftest.py`) and drops it afterwards, so a database that also holds other
data is left as it was. The trending tier's migration and price replacement are
checked in `test_trending_on_postgres.py`; its tests that create a role clean
it up before the schema goes.
