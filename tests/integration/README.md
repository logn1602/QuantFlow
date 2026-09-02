# Integration tests

Tests that need a live PostgreSQL connection. Mark them `@pytest.mark.integration`
so the default suite and CI skip them:

```
uv run pytest -m integration
```

Empty for now — see the repository's open items.
