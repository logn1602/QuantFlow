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
        "read_feeds",
        lambda ticker: sentiment.FeedRead(
            fetched("Apple beats estimates", "UNSCORABLE headline"), feeds=1
        ),
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


# ── The core's path, unchanged by the trending tier's two parameters ──────────
# docs/trending.md, section 7.5: fetch_rss gained feeds and save_sentiment
# table, both optional. Without them the core reads RSS_FEEDS and writes
# news_sentiment exactly as before. Not marked trending: these guard the core,
# so the daily refresh's test step runs them.


def _feed(*titles):
    """What feedparser.parse returns for a feed holding these headlines,
    published an hour ago."""
    import time

    import feedparser

    published = time.gmtime(time.time() - 3600)
    return feedparser.FeedParserDict(
        entries=[
            feedparser.FeedParserDict(
                title=title, link=f"https://example.com/{n}", published_parsed=published
            )
            for n, title in enumerate(titles)
        ]
    )


@pytest.fixture
def parsed(monkeypatch):
    """Records each URL fetch_rss hands feedparser, and serves one headline."""
    urls = []

    def parse(url):
        urls.append(url)
        return _feed(f"headline from {url}")

    monkeypatch.setattr(sentiment.feedparser, "parse", parse)
    return urls


def test_fetch_rss_without_feeds_reads_the_tickers_rss_feeds_entry(parsed):
    articles = sentiment.fetch_rss("AAPL")
    assert parsed == sentiment.RSS_FEEDS["AAPL"]
    assert [a["source"] for a in articles] == ["rss"]
    assert articles[0]["headline"] == f"headline from {sentiment.RSS_FEEDS['AAPL'][0]}"


def test_fetch_rss_without_feeds_reads_nothing_for_a_ticker_with_none(parsed):
    assert sentiment.fetch_rss("NOTCORE") == []
    assert parsed == []


def test_fetch_rss_reads_the_same_articles_with_its_own_feeds_passed(parsed):
    default = sentiment.fetch_rss("MSFT")
    explicit = sentiment.fetch_rss("MSFT", feeds=sentiment.RSS_FEEDS["MSFT"])
    assert [a["headline"] for a in default] == [a["headline"] for a in explicit]


# ── Failed feeds and the socket timeout (full-diff review, 2026-11-12) ────────
# feedparser raises nothing for a refused connection or an error status, and
# has no timeout of its own. The core counted neither: a total RSS outage gave
# a green run with no news, and one stalled host could hold the refresh to its
# 45-minute timeout. Not marked trending: these guard the core.


def _failing(*urls_or_all):
    """A feedparser.parse that answers 503 for these URLs (all with none
    given) and one headline for the rest."""
    import feedparser

    def parse(url):
        if not urls_or_all or url in urls_or_all:
            return feedparser.FeedParserDict(status=503, entries=[], bozo=0)
        return _feed(f"headline from {url}")

    return parse


def test_every_core_feed_is_read_with_a_socket_timeout_put_back_after(monkeypatch):
    import socket

    seen = []

    def parse(url):
        seen.append(socket.getdefaulttimeout())
        return _feed("a headline")

    monkeypatch.setattr(sentiment.feedparser, "parse", parse)
    before = socket.getdefaulttimeout()
    sentiment.fetch_rss("AAPL")
    assert seen == [sentiment.RSS_TIMEOUT_SECONDS] * len(sentiment.RSS_FEEDS["AAPL"])
    assert socket.getdefaulttimeout() == before


@pytest.mark.parametrize(
    "answer",
    [
        {"status": 503, "entries": [], "bozo": 0},
        {"status": 429, "entries": [], "bozo": 1},
        {"entries": [], "bozo": 1, "bozo_exception": OSError("connection refused")},
    ],
)
def test_an_error_status_or_an_unreadable_feed_is_a_failed_feed(monkeypatch, answer):
    import feedparser

    monkeypatch.setattr(
        sentiment.feedparser, "parse", lambda url: feedparser.FeedParserDict(answer)
    )
    read = sentiment.read_feeds("AAPL")
    assert read.articles == [] and read.failed == sentiment.RSS_FEEDS["AAPL"]


def test_a_feed_that_raises_is_a_failed_feed(monkeypatch):
    def parse(url):
        raise TimeoutError("timed out")

    monkeypatch.setattr(sentiment.feedparser, "parse", parse)
    assert sentiment.read_feeds("AAPL").failed == sentiment.RSS_FEEDS["AAPL"]


def test_a_quiet_feed_is_not_a_failed_one(monkeypatch):
    """No entries and no error: a day without news, not an outage."""
    import feedparser

    monkeypatch.setattr(
        sentiment.feedparser,
        "parse",
        lambda url: feedparser.FeedParserDict(status=200, entries=[], bozo=0),
    )
    read = sentiment.read_feeds("AAPL")
    assert read.articles == [] and read.failed == []
    assert read.feeds == len(sentiment.RSS_FEEDS["AAPL"])


@pytest.fixture
def quiet_run(monkeypatch):
    """The sentiment run with FinBERT faked, no NewsAPI and no pauses."""
    monkeypatch.setattr(sentiment, "get_finbert", FakeFinBERT)
    monkeypatch.setattr(sentiment, "fetch_newsapi", lambda ticker: [])
    monkeypatch.setattr(sentiment.time, "sleep", lambda seconds: None)


def test_every_feed_failing_turns_the_sentiment_step_red(
    fake_db, quiet_run, monkeypatch, capsys
):
    """It exited 0 with no news read, so the daily refresh stayed green."""
    monkeypatch.setattr(sentiment.feedparser, "parse", _failing())
    assert sentiment.main([]) == 1
    assert "every RSS feed failed" in capsys.readouterr().err
    assert fake_db.tables["news_sentiment"] == []


def test_some_feeds_failing_is_a_warning_and_the_rest_are_stored(
    fake_db, quiet_run, monkeypatch, caplog
):
    monkeypatch.setattr(
        sentiment.feedparser, "parse", _failing(*sentiment.RSS_FEEDS["AAPL"])
    )
    results = sentiment.run(["AAPL", "MSFT"])
    assert results == {"AAPL": 0, "MSFT": len(sentiment.RSS_FEEDS["MSFT"])}
    assert any(
        "feeds failed; the rest were read" in r.getMessage() for r in caplog.records
    )


def test_save_sentiment_writes_news_sentiment_by_default(fake_db):
    assert sentiment.save_sentiment([_article("a"), _article("b")], "AAPL") == 2
    assert len(fake_db.tables["news_sentiment"]) == 2
    assert fake_db.inserts("trending_news") == []


def test_the_sentiment_writer_refuses_any_other_table(fake_db):
    from quantflow.db.sentiment import insert_sentiment

    row = {"ticker": "AAPL", "headline": "x", "published_at": PUBLISHED}
    for table in ("raw_prices", "news_sentiment; DROP TABLE x", ""):
        with pytest.raises(ValueError, match="not a sentiment table"):
            insert_sentiment([row], table=table)
    assert fake_db.statements == []
