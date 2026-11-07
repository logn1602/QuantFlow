"""tests/unit/test_news.py

Covers quantflow.features.news: which stored articles count, from when, and
on which trading date (docs/evaluation.md, section 4). Also the news cutoff
in the trading calendar.
"""

import pandas as pd
import pytest

from quantflow.features import news
from quantflow.utils.trading_calendar import news_cutoff

# Tue 2026-03-03 to Mon 2026-03-09: Wednesday to Friday, then the weekend.
DATES = pd.bdate_range("2026-03-03", "2026-03-09")


def _article(published, inserted, *, source="rss", ticker="MSFT", **overrides):
    row = {
        "ticker": ticker,
        "source": source,
        "url_domain": "finance.yahoo.com",
        "published_at": pd.Timestamp(published, tz="UTC"),
        "inserted_at": pd.Timestamp(inserted, tz="UTC"),
        "sentiment": "positive",
        "score_pos": 0.8,
        "score_neg": 0.1,
        "score_neu": 0.1,
        "compound": 0.7,
    }
    return row | overrides


def _date_of(*article):
    frame = pd.DataFrame(
        [_article(*article[:2], **(article[2] if len(article) > 2 else {}))]
    )
    return news.window_dates(news.available_at(frame), DATES).iloc[0]


def test_the_cutoff_is_midnight_utc_after_the_trading_date():
    assert news_cutoff("2026-03-03") == pd.Timestamp("2026-03-04 00:00", tz="UTC")
    late_new_york = pd.Timestamp("2026-03-03 23:30", tz="America/New_York")
    assert news_cutoff(late_new_york) == pd.Timestamp("2026-03-05 00:00", tz="UTC")


def test_an_article_stored_after_the_workflow_counts_that_day():
    assert _date_of("2026-03-03 21:30", "2026-03-03 22:35") == pd.Timestamp(
        "2026-03-03"
    )


def test_an_article_stored_after_midnight_counts_the_next_day():
    assert _date_of("2026-03-03 21:30", "2026-03-04 00:05") == pd.Timestamp(
        "2026-03-04"
    )


def test_a_saturday_article_lands_on_monday():
    assert _date_of("2026-03-07 10:00", "2026-03-07 10:05") == pd.Timestamp(
        "2026-03-09"
    )


def test_a_newsapi_article_counts_a_day_after_it_was_published():
    stored_early = _date_of(
        "2026-03-03 10:00", "2026-03-03 10:05", {"source": "newsapi"}
    )
    assert stored_early == pd.Timestamp("2026-03-04")


def test_an_article_fetched_late_counts_from_when_it_arrived():
    """After an outage the next fetch stores days-old articles; they count
    from their arrival, not on their publish date."""
    assert _date_of("2026-03-03 12:00", "2026-03-05 22:40") == pd.Timestamp(
        "2026-03-05"
    )


def test_news_after_the_last_cutoff_has_no_date_yet():
    assert pd.isna(_date_of("2026-03-09 23:00", "2026-03-10 00:30"))


def test_placeholder_rows_are_bad_and_real_neutral_rows_are_not():
    frame = pd.DataFrame(
        [
            _article("2026-03-03 10:00", "2026-03-03 10:05", sentiment="neutral",
                     score_pos=0.33, score_neg=0.33, score_neu=0.34, compound=0.0),
            _article("2026-03-03 10:00", "2026-03-03 10:05", sentiment="neutral",
                     score_pos=0.05, score_neg=0.05, score_neu=0.90, compound=0.0),
        ]
    )  # fmt: skip
    assert news.bad_rows(frame).tolist() == [True, False]


@pytest.mark.parametrize(
    ("ticker", "domain", "bad"),
    [
        ("AAPL", "feeds.marketwatch.com", True),
        ("AAPL", "marketwatch.com", True),
        ("MSFT", "feeds.marketwatch.com", False),  # only AAPL was credited
        ("AAPL", "finance.yahoo.com", False),
        ("AAPL", "notmarketwatch.com", False),
    ],
)
def test_the_general_feed_is_bad_only_under_aapl(ticker, domain, bad):
    frame = pd.DataFrame(
        [
            _article(
                "2026-03-03 10:00", "2026-03-03 10:05", ticker=ticker, url_domain=domain
            )
        ]
    )
    assert news.bad_rows(frame).tolist() == [bad]


def test_dates_with_news_skip_bad_rows_and_count_each_pair_once():
    frame = pd.DataFrame(
        [
            _article("2026-03-03 10:00", "2026-03-03 10:05"),
            _article("2026-03-03 11:00", "2026-03-03 11:05"),
            _article("2026-03-04 10:00", "2026-03-04 10:05", ticker="AAPL",
                     url_domain="feeds.marketwatch.com"),
        ]
    )  # fmt: skip
    pairs = news.dates_with_news(frame, DATES)
    assert pairs.to_dict("records") == [
        {"ticker": "MSFT", "date": pd.Timestamp("2026-03-03")}
    ]


def test_no_articles_gives_no_dates():
    assert news.dates_with_news(pd.DataFrame(), DATES).empty


def test_the_daily_workflow_runs_after_the_close_and_before_the_cutoff():
    """The cutoff only works if the workflow's run, and the sentiment step
    in it, falls between the session's close and that cutoff."""
    import re
    from pathlib import Path

    workflow = Path(__file__).parents[2] / ".github" / "workflows" / "daily_refresh.yml"
    match = re.search(r"cron: '(\d+) (\d+) \* \* 1-5'", workflow.read_text("utf-8"))
    assert match, "the daily schedule is not where this test looks for it"
    minute, hour = int(match[1]), int(match[2])
    for day in ("2026-01-15", "2026-07-15"):  # winter and summer time
        run = pd.Timestamp(f"{day} {hour:02d}:{minute:02d}", tz="UTC")
        close = pd.Timestamp(f"{day} 16:00", tz="America/New_York").tz_convert("UTC")
        assert close < run < news_cutoff(day)
