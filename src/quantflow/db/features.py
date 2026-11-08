"""Reads of the stored news the feature matrix is built from.

The live run builds its features in process (features.engineering
.build_features) from the daily bars and these article rows, the same
function the evaluation calls at every origin (decision D4,
docs/evaluation.md). The technical_indicators and anomalies tables are
still written, for the dashboard, but no feature reads them.
"""

import pandas as pd
from sqlalchemy import text

from quantflow.db.connection import get_engine

# The columns every article read returns, shared with the evaluation's
# snapshot export: the URL's host only, enough to tell the general feed's rows
# apart, without copying headlines or links out of the database.
ARTICLE_SELECT = """
    SELECT ticker, source,
           substring(url from '^[A-Za-z][A-Za-z0-9+.-]*://([^/:?#]+)') AS url_domain,
           published_at, inserted_at, sentiment,
           score_pos, score_neg, score_neu, compound
    FROM news_sentiment
"""

_ARTICLES = text(ARTICLE_SELECT + "WHERE ticker = :t ORDER BY inserted_at")


def load_articles(ticker: str) -> pd.DataFrame:
    """Every stored article for the ticker, in the order they arrived."""
    with get_engine().connect() as conn:
        return pd.read_sql(_ARTICLES, conn, params={"t": ticker})


_FETCH_TIMES = text("SELECT DISTINCT inserted_at FROM news_sentiment")


def load_fetch_times() -> pd.Series:
    """Every moment a fetch stored articles, for any ticker: a date whose
    news window holds none was not looked at (decision D6)."""
    with get_engine().connect() as conn:
        stored = pd.read_sql(_FETCH_TIMES, conn)
    return pd.to_datetime(stored["inserted_at"], utc=True)
