"""Writer for the news_sentiment table, and for trending_news, its twin.

trending_news holds the trending tier's articles, apart on purpose: the
core's news features read every stored news_sentiment row as a sign that a
fetch happened (docs/trending.md, sections 2 and 7.6).
"""

from quantflow.db.batch import insert_rows
from quantflow.db.connection import get_engine

# The tables scored headlines may go to. A name outside them is refused
# before anything is sent: it reaches the SQL as an identifier.
SENTIMENT_TABLES = ("news_sentiment", "trending_news")


def insert_sentiment(rows: list[dict], table: str = "news_sentiment") -> int:
    """Write scored headlines; return how many Postgres actually wrote.

    A headline already stored for the same (ticker, headline, published_at)
    is skipped and not counted, which is most of them: the RSS feeds return
    the same headlines on every run. Any other write error raises and rolls
    the whole call back. table is news_sentiment unless the trending tier
    passes trending_news, which has the same columns and key.
    """
    if table not in SENTIMENT_TABLES:
        raise ValueError(f"not a sentiment table: {table!r}")
    if not rows:
        return 0
    with get_engine().begin() as conn:
        return insert_rows(
            conn,
            table,
            rows,
            skip_conflicts=("ticker", "headline", "published_at"),
        )
