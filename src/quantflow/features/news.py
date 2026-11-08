"""Which stored news articles count, and from when.

The rules of docs/evaluation.md, section 4 (decisions D6 and D7). They live
here so one definition serves both the evaluation's news coverage report
and the features (features.engineering.build_features).

Articles are news_sentiment rows: ticker, source, url_domain, published_at,
inserted_at, sentiment, score_pos, score_neg, score_neu, compound.

The features never need the rows themselves, only their sums per news day
(news_days): every trading date's window is a run of whole news days, so a
window's counts and compound sum are the sums of its days'. The live run
reads those sums from the database (db.features.load_news_days) rather than
every article; the evaluation sums the snapshot's rows here. Both then go
through the same code.
"""

from decimal import Decimal

import numpy as np
import pandas as pd

from quantflow.utils.trading_calendar import NEWS_CUTOFF_AFTER_DATE, news_cutoff

# NewsAPI's free tier serves articles about a day after publication, so the
# live run on day t has none of day t's NewsAPI articles. RSS arrives live.
SOURCE_DELAY = {"newsapi": pd.Timedelta(hours=24)}

# The neutral row stored for every headline while FinBERT could not load,
# removed in 29d1957. Real FinBERT output never lands on these exact scores.
PLACEHOLDER_SENTIMENT = "neutral"
PLACEHOLDER_SCORES = {
    "compound": 0.0,
    "score_pos": 0.33,
    "score_neg": 0.33,
    "score_neu": 0.34,
}

# np.isclose's defaults, which bad_rows uses; the database query applies the
# same test (db.features.load_news_days).
CLOSE_RTOL = 1e-05
CLOSE_ATOL = 1e-08

# MarketWatch's general-market feed was credited to AAPL until 29d1957; its
# headlines are not about Apple.
GENERAL_FEED = ("AAPL", "marketwatch.com")

# A news-days summary: the stored articles counted and summed by ticker, the
# news day each became available on, the news day it was stored on, and
# whether it counts (not a bad row). Built from rows by news_days, read from
# the database by db.features.load_news_days, the same frame either way.
NEWS_DAY_KEYS = ["ticker", "available_day", "stored_day", "counts"]
NEWS_DAY_SUMS = ["articles", "positive", "negative", "compounds", "compound_sum"]
NEWS_DAY_COLUMNS = [*NEWS_DAY_KEYS, *NEWS_DAY_SUMS]


def bad_rows(articles: pd.DataFrame) -> pd.Series:
    """True for the rows no feature should use: FinBERT placeholders, and
    MarketWatch's general headlines credited to AAPL."""
    placeholder = articles["sentiment"].eq(PLACEHOLDER_SENTIMENT)
    for column, value in PLACEHOLDER_SCORES.items():
        placeholder &= np.isclose(
            pd.to_numeric(articles[column]), value, rtol=CLOSE_RTOL, atol=CLOSE_ATOL
        )
    ticker, domain = GENERAL_FEED
    domains = articles["url_domain"].fillna("").str.lower()
    general = articles["ticker"].eq(ticker) & (
        domains.eq(domain) | domains.str.endswith("." + domain)
    )
    return pd.Series(placeholder | general, index=articles.index)


def available_at(articles: pd.DataFrame) -> pd.Series:
    """When each article became available to a run: the later of its publish
    time plus its source's delay and the moment it reached the database.

    inserted_at is what makes an article fetched late, after an outage,
    count from when it arrived rather than from its publish date, the same in
    training as live.
    """
    published = pd.to_datetime(articles["published_at"], utc=True)
    inserted = pd.to_datetime(articles["inserted_at"], utc=True)
    delay = pd.to_timedelta(
        articles["source"].map(SOURCE_DELAY).fillna(pd.Timedelta(0))
    )
    return pd.concat([published + delay, inserted], axis=1).max(axis=1)


def news_day(times) -> pd.Series:
    """The news day each time falls in: the calendar date d with
    news_cutoff(d - 1 day) < time <= news_cutoff(d), as a midnight.

    A trading date's window runs from the previous trading date's cutoff to
    its own, so it holds exactly the news days after the previous trading
    date, up to and including its own date.
    """
    stamps = pd.Series(pd.to_datetime(pd.Series(times, dtype=object), utc=True))
    naive = stamps.dt.tz_convert("UTC").dt.tz_localize(None)
    return (naive - NEWS_CUTOFF_AFTER_DATE).dt.ceil("D")


def day_window_dates(days, trading_dates) -> pd.Series:
    """The trading date whose news window holds each news day: the first
    trading date on or after it. A day after the last trading date has no
    date yet (NaT); one before the first lands on the first."""
    dates = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(trading_dates))))
    # news_cutoff reads only a date's calendar day, in UTC.
    calendar = pd.DatetimeIndex([news_cutoff(d) for d in dates])
    calendar = (calendar - NEWS_CUTOFF_AFTER_DATE).tz_localize(None)
    days = pd.Series(days)
    known = days.notna().to_numpy()
    position = np.full(len(days), len(dates))
    position[known] = calendar.searchsorted(pd.DatetimeIndex(days[known]), "left")
    in_range = position < len(dates)
    out = pd.Series(pd.NaT, index=days.index, dtype=dates.dtype)
    out[in_range] = dates[position[in_range]]
    return out


def window_dates(available: pd.Series, trading_dates) -> pd.Series:
    """The trading date whose news window each availability time falls in.

    A trading date's window runs from the previous trading date's cutoff
    (exclusive) to its own (inclusive), so weekend and holiday news lands on
    the next trading date. A time after the last date's cutoff has no date
    yet (NaT); one before the first date's window lands on the first date.
    """
    out = day_window_dates(news_day(available), trading_dates)
    return out.set_axis(available.index)


def exact(value) -> Decimal:
    """A stored compound as the decimal it was written as (the column is
    NUMERIC(6, 4)): sums of these are exact, so they do not depend on the
    order the rows are added in, and the database's sum is the same number.
    Missing counts as nothing; news_days counts the present ones apart."""
    if pd.isna(value):
        return Decimal(0)
    if isinstance(value, Decimal):
        return value
    return Decimal(str(float(value)))


def total(values) -> Decimal:
    return sum(values, Decimal(0))


def news_days(articles: pd.DataFrame) -> pd.DataFrame:
    """The articles summed into a news-days summary (NEWS_DAY_COLUMNS), the
    frame the live run reads from the database (db.features.load_news_days).

    Per group: articles, positive and negative counts, how many have a
    compound, and the exact sum of those compounds.
    """
    if articles.empty:
        return pd.DataFrame(
            {
                "ticker": pd.Series(dtype=object),
                "available_day": pd.Series(dtype="datetime64[ns]"),
                "stored_day": pd.Series(dtype="datetime64[ns]"),
                "counts": pd.Series(dtype=bool),
            }
            | {column: pd.Series(dtype="int64") for column in NEWS_DAY_SUMS[:-1]}
            | {"compound_sum": pd.Series(dtype=object)}
        )
    sentiment = articles["sentiment"]
    compound = articles["compound"]
    rows = pd.DataFrame(
        {
            "ticker": articles["ticker"].to_numpy(),
            "available_day": news_day(available_at(articles)).to_numpy(),
            "stored_day": news_day(articles["inserted_at"]).to_numpy(),
            "counts": (~bad_rows(articles)).to_numpy(),
            "articles": 1,
            "positive": sentiment.eq("positive").astype("int64").to_numpy(),
            "negative": sentiment.eq("negative").astype("int64").to_numpy(),
            "compounds": pd.to_numeric(compound).notna().astype("int64").to_numpy(),
            "compound_sum": [exact(v) for v in compound],
        }
    )
    grouped = rows.groupby(NEWS_DAY_KEYS, sort=True)
    sums = grouped[NEWS_DAY_SUMS[:-1]].sum()
    sums["compound_sum"] = grouped["compound_sum"].agg(total)
    return sums.reset_index()[NEWS_DAY_COLUMNS]


def query_rules() -> dict:
    """The rules above, as db.features.load_news_days applies them in the
    database: one definition for the rows summed here and the sums read."""
    return {
        "source_delays": SOURCE_DELAY,
        "placeholder_sentiment": PLACEHOLDER_SENTIMENT,
        "placeholder_scores": PLACEHOLDER_SCORES,
        "rtol": CLOSE_RTOL,
        "atol": CLOSE_ATOL,
        "general_feed": GENERAL_FEED,
        "cutoff_after": NEWS_CUTOFF_AFTER_DATE,
    }


def dates_with_news(articles: pd.DataFrame, trading_dates) -> pd.DataFrame:
    """The (ticker, date) pairs whose window holds at least one article that
    counts: not a bad row, and available by that date's cutoff."""
    if articles.empty:
        return pd.DataFrame(columns=["ticker", "date"])
    good = articles[~bad_rows(articles)]
    dated = good.assign(date=window_dates(available_at(good), trading_dates))
    pairs = dated.dropna(subset=["date"])[["ticker", "date"]].drop_duplicates()
    return pairs.sort_values(["ticker", "date"]).reset_index(drop=True)
