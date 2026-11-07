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


def test_untrusted_feed_text_cannot_break_the_batched_insert(fake_db):
    """psycopg2 rejects a NUL byte or a lone surrogate before the statement
    reaches the server. In one batched INSERT, a single such headline from a
    feed would fail the whole write, and with it the daily refresh."""
    bad = _article("Apple beats\x00 estimates")
    bad["url"] = "https://example.com/\ud83d"

    assert sentiment.save_sentiment([bad, _article("clean headline")], "AAPL") == 2
    for row in fake_db.tables["news_sentiment"]:
        for field in ("headline", "url"):
            assert "\x00" not in row[field]
            row[field].encode("utf-8")  # a lone surrogate raises here, as in psycopg2


def test_a_headline_that_is_empty_after_cleaning_is_not_written(fake_db):
    """An RSS item with no title, or one made only of NUL bytes, cleans to an
    empty string: not news, and not a row."""
    articles = [_article(""), _article("\x00\x00"), _article("real one")]
    assert sentiment.save_sentiment(articles, "AAPL") == 1
    assert [r["headline"] for r in fake_db.tables["news_sentiment"]] == ["real one"]


def test_the_same_headline_twice_in_one_run_is_stored_once(fake_db):
    """RSS and NewsAPI can both return a headline with the same timestamp."""
    articles = [_article("dup", "rss"), _article("dup", "newsapi")]
    assert sentiment.save_sentiment(articles, "AAPL") == 1


# ── Missing is not neutral ────────────────────────────────────────────────────
#
# When FinBERT could not run, every headline used to be stored as "neutral"
# with compound 0.0, which the feature matrix cannot tell apart from news
# that genuinely was neutral.


def _label_scores():
    return [
        {"label": "positive", "score": 0.7},
        {"label": "negative", "score": 0.2},
        {"label": "neutral", "score": 0.1},
    ]


class FakeFinBERT:
    """Scores every headline alike; raises on any containing UNSCORABLE."""

    def __call__(self, inputs):
        if isinstance(inputs, list):
            if any("UNSCORABLE" in h for h in inputs):
                raise RuntimeError("batch failed")
            return [_label_scores() for _ in inputs]
        if "UNSCORABLE" in inputs:
            raise RuntimeError("cannot score this headline")
        return [_label_scores()]


@pytest.fixture
def news(monkeypatch):
    """Fetched headlines with no network and no politeness sleep."""

    def fetched(*headlines):
        return [
            {"headline": h, "url": "", "published_at": PUBLISHED, "source": "rss"}
            for h in headlines
        ]

    monkeypatch.setattr(
        sentiment,
        "fetch_rss",
        lambda ticker: fetched("Apple beats estimates", "UNSCORABLE headline"),
    )
    monkeypatch.setattr(sentiment, "fetch_newsapi", lambda ticker: [])
    monkeypatch.setattr(sentiment.time, "sleep", lambda seconds: None)


def test_nothing_is_written_when_finbert_fails_to_load(fake_db, news, monkeypatch):
    monkeypatch.setattr(sentiment, "get_finbert", lambda: None)

    with pytest.raises(sentiment.FinBERTUnavailableError):
        sentiment.run(["AAPL"])
    assert fake_db.tables["news_sentiment"] == []


def test_the_cli_exits_non_zero_when_finbert_cannot_load(
    fake_db, news, monkeypatch, capsys
):
    """It logged an error and exited 0, so the daily refresh showed green with
    stale sentiment."""
    monkeypatch.setattr(sentiment, "get_finbert", lambda: None)
    assert sentiment.main(["--ticker", "AAPL"]) == 1
    assert "FinBERT" in capsys.readouterr().err


def test_a_headline_finbert_cannot_score_is_not_written(fake_db, news, monkeypatch):
    monkeypatch.setattr(sentiment, "get_finbert", FakeFinBERT)

    sentiment.run(["AAPL"])

    stored = fake_db.tables["news_sentiment"]
    assert [row["headline"] for row in stored] == ["Apple beats estimates"]
    assert stored[0]["compound"] == pytest.approx(0.5)


def test_every_rss_feed_is_specific_to_its_ticker():
    """A general-market feed listed under one ticker credits every market
    headline to that ticker's sentiment features."""
    for ticker, feeds in sentiment.RSS_FEEDS.items():
        for url in feeds:
            assert f"s={ticker}&" in url, f"{ticker}: not a per-symbol feed: {url}"
