"""Which stored news articles count, and from when.

The rules of docs/evaluation.md, section 4 (decisions D6 and D7). They live
here so one definition serves both the evaluation's news coverage report
and the features (features.engineering.build_features).

Articles are news_sentiment rows: ticker, source, url_domain, published_at,
inserted_at, sentiment, score_pos, score_neg, score_neu, compound.
"""

import numpy as np
import pandas as pd

from quantflow.utils.trading_calendar import news_cutoff

# NewsAPI's free tier serves articles about a day after publication, so the
# live run on day t has none of day t's NewsAPI articles. RSS arrives live.
SOURCE_DELAY = {"newsapi": pd.Timedelta(hours=24)}

# The neutral row stored for every headline while FinBERT could not load,
# removed in 29d1957. Real FinBERT output never lands on these exact scores.
PLACEHOLDER_SCORES = {"score_pos": 0.33, "score_neg": 0.33, "score_neu": 0.34}

# MarketWatch's general-market feed was credited to AAPL until 29d1957; its
# headlines are not about Apple.
GENERAL_FEED = ("AAPL", "marketwatch.com")


def bad_rows(articles: pd.DataFrame) -> pd.Series:
    """True for the rows no feature should use: FinBERT placeholders, and
    MarketWatch's general headlines credited to AAPL."""
    placeholder = articles["sentiment"].eq("neutral") & np.isclose(
        pd.to_numeric(articles["compound"]), 0.0
    )
    for column, value in PLACEHOLDER_SCORES.items():
        placeholder &= np.isclose(pd.to_numeric(articles[column]), value)
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


def window_dates(available: pd.Series, trading_dates) -> pd.Series:
    """The trading date whose news window each availability time falls in.

    A trading date's window runs from the previous trading date's cutoff
    (exclusive) to its own (inclusive), so weekend and holiday news lands on
    the next trading date. A time after the last date's cutoff has no date
    yet (NaT); one before the first date's window lands on the first date.
    """
    dates = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(trading_dates))))
    cutoffs = pd.DatetimeIndex([news_cutoff(d) for d in dates])
    times = pd.to_datetime(available, utc=True)
    known = times.notna()
    position = np.full(len(times), len(dates))
    position[known.to_numpy()] = cutoffs.searchsorted(times[known], side="left")
    in_range = position < len(dates)
    out = pd.Series(pd.NaT, index=available.index, dtype="datetime64[ns]")
    out[in_range] = dates[position[in_range]]
    return out


def dates_with_news(articles: pd.DataFrame, trading_dates) -> pd.DataFrame:
    """The (ticker, date) pairs whose window holds at least one article that
    counts: not a bad row, and available by that date's cutoff."""
    if articles.empty:
        return pd.DataFrame(columns=["ticker", "date"])
    good = articles[~bad_rows(articles)]
    dated = good.assign(date=window_dates(available_at(good), trading_dates))
    pairs = dated.dropna(subset=["date"])[["ticker", "date"]].drop_duplicates()
    return pairs.sort_values(["ticker", "date"]).reset_index(drop=True)
