"""Reads of the stored news the feature matrix is built from.

The live run builds its features in process (features.engineering
.build_features) from the daily bars and the stored news, the same code the
evaluation runs at every origin (decision D4, docs/evaluation.md). The
technical_indicators and anomalies tables are still written, for the
dashboard, but no feature reads them.

The live run reads the news summed per ticker and news day
(load_news_days), once per run, not every article once per ticker and stage:
on the 2026-11-06 snapshot the article rows came to 5.9 MB a read, read three
times a run, and grew with every article stored (Supabase's free tier allows
5 GB of egress a month).
"""

from decimal import Decimal

import pandas as pd
from sqlalchemy import text

from quantflow.db.connection import get_engine

# The host of an article's URL: enough to tell the general feed's rows
# apart, without copying headlines or links out of the database.
URL_DOMAIN = "substring(url from '^[A-Za-z][A-Za-z0-9+.-]*://([^/:?#]+)')"

# The columns every article read returns, used by the evaluation's snapshot
# export (db.snapshot).
ARTICLE_SELECT = f"""
    SELECT ticker, source,
           {URL_DOMAIN} AS url_domain,
           published_at, inserted_at, sentiment,
           score_pos, score_neg, score_neu, compound
    FROM news_sentiment
"""

# The score columns a placeholder rule may test.
_SCORE_COLUMNS = {"compound", "score_pos", "score_neg", "score_neu"}


def _news_day(column: str) -> str:
    """The news day a stored time falls in, as features.news.news_day dates
    it: the calendar date d with d + cutoff_after - 1 day < time <= d +
    cutoff_after, read in UTC whatever the session's time zone."""
    shifted = f"(({column} AT TIME ZONE 'UTC') - CAST(:cutoff_after AS interval))"
    return (
        f"(CAST({shifted} AS date)"
        f" + CASE WHEN {shifted} = date_trunc('day', {shifted}) THEN 0 ELSE 1 END)"
    )


def _news_days_query(
    source_delays: dict,
    placeholder_sentiment: str,
    placeholder_scores: dict,
    rtol: float,
    atol: float,
    general_feed: tuple[str, str],
) -> tuple[str, dict]:
    """The summary query and its parameters, from the rules in
    features.news: availability (source delay), the placeholder test
    (np.isclose in floats) and the general feed (bad_rows)."""
    params: dict = {"placeholder_sentiment": placeholder_sentiment}

    whens = []
    for i, (source, delay) in enumerate(source_delays.items()):
        whens.append(f"WHEN :source_{i} THEN CAST(:delay_{i} AS interval)")
        params[f"source_{i}"] = source
        params[f"delay_{i}"] = pd.Timedelta(delay).to_pytimedelta()
    delay_sql = f"CASE source {' '.join(whens)} ELSE interval '0' END"

    close = []
    for i, (column, value) in enumerate(placeholder_scores.items()):
        if column not in _SCORE_COLUMNS:
            raise ValueError(f"not a score column: {column!r}")
        # np.isclose(a, value): |a - value| <= atol + rtol * |value|.
        close.append(
            f"abs(CAST({column} AS double precision)"
            f" - CAST(:score_{i} AS double precision))"
            f" <= CAST(:tolerance_{i} AS double precision)"
        )
        params[f"score_{i}"] = float(value)
        params[f"tolerance_{i}"] = atol + rtol * abs(float(value))
    placeholder = (
        f"COALESCE(sentiment = :placeholder_sentiment AND {' AND '.join(close)}, false)"
    )

    ticker, domain = general_feed
    host = f"lower(COALESCE({URL_DOMAIN}, ''))"
    params |= {
        "feed_ticker": ticker,
        "feed_domain": domain,
        "feed_suffix": "." + domain,
        "feed_suffix_length": len(domain) + 1,
    }
    general = (
        f"COALESCE(ticker = :feed_ticker AND ({host} = :feed_domain"
        f" OR right({host}, :feed_suffix_length) = :feed_suffix), false)"
    )

    query = f"""
        WITH articles AS (
            SELECT ticker, sentiment, compound, inserted_at,
                   GREATEST(published_at + {delay_sql}, inserted_at) AS available_at,
                   NOT ({placeholder} OR {general}) AS counts
            FROM news_sentiment
        )
        SELECT ticker,
               {_news_day("available_at")} AS available_day,
               {_news_day("inserted_at")} AS stored_day,
               counts,
               count(*) AS articles,
               count(*) FILTER (WHERE sentiment = 'positive') AS positive,
               count(*) FILTER (WHERE sentiment = 'negative') AS negative,
               count(compound) AS compounds,
               COALESCE(sum(compound), 0) AS compound_sum
        FROM articles
        GROUP BY 1, 2, 3, 4
    """
    return query, params


def _decimal(value) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def load_news_days(
    *,
    source_delays: dict,
    placeholder_sentiment: str,
    placeholder_scores: dict,
    rtol: float,
    atol: float,
    general_feed: tuple[str, str],
    cutoff_after: pd.Timedelta,
) -> pd.DataFrame:
    """Every ticker's stored news summed per news day, in the shape
    features.news.news_days builds from article rows: ticker, available_day,
    stored_day, counts, then articles, positive, negative, compounds and the
    exact compound_sum. Its rules come from features.news
    (features.news.query_rules); a test on Postgres checks the two agree.

    A few kilobytes per stored day, where the article rows were about 150
    bytes per article.
    """
    query, params = _news_days_query(
        source_delays,
        placeholder_sentiment,
        placeholder_scores,
        rtol,
        atol,
        general_feed,
    )
    params["cutoff_after"] = pd.Timedelta(cutoff_after).to_pytimedelta()
    with get_engine().connect() as conn:
        # coerce_float would turn the exact sums into floats.
        rows = pd.read_sql(text(query), conn, params=params, coerce_float=False)
    out = pd.DataFrame(
        {
            "ticker": rows["ticker"].astype(object),
            "available_day": pd.to_datetime(rows["available_day"]),
            "stored_day": pd.to_datetime(rows["stored_day"]),
            "counts": rows["counts"].astype(bool),
        }
        | {
            column: rows[column].astype("int64")
            for column in ("articles", "positive", "negative", "compounds")
        }
        | {"compound_sum": rows["compound_sum"].map(_decimal).astype(object)}
    )
    # Sorted here rather than in SQL, whose order for text follows the
    # database's collation: the same order news_days gives.
    keys = ["ticker", "available_day", "stored_day", "counts"]
    return out.sort_values(keys, kind="stable").reset_index(drop=True)
