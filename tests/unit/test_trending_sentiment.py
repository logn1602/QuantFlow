"""tests/unit/test_trending_sentiment.py

The trending tier's news (docs/trending.md, section 7.5): RSS only, from a
feed built for each symbol by the template the core's feeds follow, scored
by the same FinBERT code and written to trending_news, never news_sentiment.
The core's own path, with neither parameter, is guarded in test_sentiment.py.
"""

import datetime as dt
import time

import feedparser
import pytest

from quantflow.features import sentiment
from quantflow.ingestion import trending_rules as rules

pytestmark = pytest.mark.trending

PUBLISHED = dt.datetime(2026, 11, 10, 14, 0, tzinfo=dt.UTC)


def test_the_feed_template_reproduces_every_core_feed_exactly():
    """So a trending symbol's feed is the same kind of feed as the core's."""
    assert sentiment.RSS_FEEDS, "no core feeds to compare with"
    for ticker, feeds in sentiment.RSS_FEEDS.items():
        assert [rules.rss_feed(ticker)] == feeds


def test_a_feed_is_built_from_a_valid_symbol_only():
    assert rules.rss_feed("BRK.B") == (
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BRK.B&region=US&lang=en-US"
    )
    for bad in ("spcx", "X&s=EVIL", "A B", "", "TOOLONGSYMBOL"):
        with pytest.raises(ValueError, match="not a symbol"):
            rules.rss_feed(bad)


def test_fetch_rss_reads_only_the_feeds_given(monkeypatch):
    urls = []

    def parse(url):
        urls.append(url)
        published = time.gmtime(time.time() - 3600)
        return feedparser.FeedParserDict(
            entries=[
                feedparser.FeedParserDict(
                    title="Hims jumps", link="https://e.x/1", published_parsed=published
                )
            ]
        )

    monkeypatch.setattr(sentiment.feedparser, "parse", parse)
    feed = rules.rss_feed("HIMS")
    articles = sentiment.fetch_rss("HIMS", feeds=[feed])
    assert urls == [feed]
    assert [(a["headline"], a["source"]) for a in articles] == [("Hims jumps", "rss")]


def _scored(headline):
    return {
        "headline": headline,
        "url": "https://e.x/1",
        "published_at": PUBLISHED,
        "source": "rss",
        "sentiment": "positive",
        "score_pos": 0.7,
        "score_neg": 0.1,
        "score_neu": 0.2,
        "compound": 0.6,
    }


def test_trending_articles_go_to_trending_news_and_never_news_sentiment(fake_db):
    written = sentiment.save_sentiment(
        [_scored("Hims jumps"), _scored("Hims falls")], "HIMS", table="trending_news"
    )
    assert written == 2
    assert [r["ticker"] for r in fake_db.tables["trending_news"]] == ["HIMS", "HIMS"]
    assert fake_db.inserts("news_sentiment") == []


def test_a_trending_headline_already_stored_is_skipped_and_not_counted(fake_db):
    fake_db.seed(
        "trending_news",
        [{"ticker": "HIMS", "headline": "Hims jumps", "published_at": PUBLISHED}],
    )
    written = sentiment.save_sentiment(
        [_scored("Hims jumps"), _scored("new")], "HIMS", table="trending_news"
    )
    assert written == 1
