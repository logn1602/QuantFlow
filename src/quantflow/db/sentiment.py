"""Writer for the news_sentiment table."""

from quantflow.db.batch import insert_rows
from quantflow.db.connection import get_engine


def insert_sentiment(rows: list[dict]) -> int:
    """Write scored headlines; return how many Postgres actually wrote.

    A headline already stored for the same (ticker, headline, published_at)
    is skipped and not counted, which is most of them: the RSS feeds return
    the same headlines on every run. Any other write error raises and rolls
    the whole call back.
    """
    if not rows:
        return 0
    with get_engine().begin() as conn:
        return insert_rows(
            conn,
            "news_sentiment",
            rows,
            skip_conflicts=("ticker", "headline", "published_at"),
        )
