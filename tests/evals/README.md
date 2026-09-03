# Evaluation suite

End-to-end model evaluation: fits the models on real seeded price history and
reproduces the figures reported in the README. Mark them `@pytest.mark.eval`,
which the default suite and CI exclude because they need a seeded database and
roughly two years of history per ticker.

```
uv run pytest -m eval
```

Empty for now — see the repository's open items.
