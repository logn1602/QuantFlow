"""tests/unit/test_sentiment.py

Covers what the sentiment stage writes to news_sentiment.

save_sentiment counted every attempted row as inserted, including the many
headlines ON CONFLICT DO NOTHING skips (RSS feeds return the same headlines
on every run), and caught errors per row inside one transaction, which on
Postgres rolls the whole batch back while reporting success.
"""

import datetime as dt

import pytest
from sqlalchemy.exc import DataError

from quantflow.features import sentiment

PUBLISHED = dt.datetime(2026, 11, 6, 14, 0, tzinfo=dt.UTC)


def _article(headline, source="rss"):
    return {
        "headline": headline,
        "url": f"https://example.com/{abs(hash(headline))}",
        "published_at": PUBLISHED,
        "source": source,
        "sentiment": "positive",
        "score_pos": 0.7,
        "score_neg": 0.2,
        "score_neu": 0.1,
        "compound": 0.5,
    }


# ── The write path ────────────────────────────────────────────────────────────


def test_save_sentiment_counts_only_rows_the_database_wrote(fake_db):
    fake_db.seed(
        "news_sentiment",
        [{"ticker": "AAPL", "headline": "seen before", "published_at": PUBLISHED}],
    )
    articles = [_article("seen before"), _article("new one"), _article("new two")]

    assert sentiment.save_sentiment(articles, "AAPL") == 2


def test_save_sentiment_writes_one_batched_insert(fake_db):
    sentiment.save_sentiment([_article(f"headline {i}") for i in range(20)], "AAPL")
    assert len(fake_db.inserts("news_sentiment")) == 1


def test_save_sentiment_write_error_raises_instead_of_being_logged(fake_db):
    fake_db.fail_next_insert = DataError("INSERT", {}, Exception("value too long"))
    with pytest.raises(DataError):
        sentiment.save_sentiment([_article("a"), _article("b")], "AAPL")


def test_the_same_headline_twice_in_one_run_is_stored_once(fake_db):
    """RSS and NewsAPI can both return a headline with the same timestamp."""
    articles = [_article("dup", "rss"), _article("dup", "newsapi")]
    assert sentiment.save_sentiment(articles, "AAPL") == 1
