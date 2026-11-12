"""
features/sentiment.py
---------------------
Phase 6: News Sentiment Analysis Engine

Pulls financial news from two sources:
  1. NewsAPI       : Structured API, 100 req/day free tier
  2. RSS Feeds     : Yahoo Finance per-symbol headlines, free

Analyzes headlines using FinBERT:
  - ProsusAI/finbert — pre-trained on financial news
  - Returns: positive / negative / neutral + probabilities
  - Compound score = pos_prob - neg_prob (range: -1 to +1)

Stores results in news_sentiment table.
Sentiment scores can then be used as features in forecasting models.

Usage:
    python -m quantflow.features.sentiment                # every ticker
    python -m quantflow.features.sentiment --ticker AAPL  # one ticker only
    python -m quantflow.features.sentiment --show AAPL    # its latest sentiment
    python -m quantflow.features.sentiment --summary      # market overview
"""

import argparse
import sys
import time
import warnings
from datetime import UTC, datetime, timedelta

import feedparser
import pandas as pd
from sqlalchemy import text

from quantflow.config import NEWS_API_KEY, TICKERS
from quantflow.db.connection import get_engine
from quantflow.db.sentiment import insert_sentiment
from quantflow.utils.logger import get_logger
from quantflow.utils.text import storable_text as _clean_text

# Applied after imports rather than before them: nothing imported here
# warns at import time (transformers is loaded lazily in get_finbert),
# and an early call is what forced the imports below it to trip E402.
warnings.filterwarnings("ignore")

logger = get_logger("sentiment")


class FinBERTUnavailableError(RuntimeError):
    """FinBERT could not be loaded, so no headline can be scored."""


# ── FinBERT model ─────────────────────────────────────────────────────────────
# Pinned to an exact commit for REPRODUCIBILITY, not security. The RCE angle
# people usually cite for unpinned models does not apply here: trust_remote_code
# is unset and torch is >= 2.6, so weights load with weights_only=True.
#
# The real risk is quieter. Sentiment scores feed 6 of the 33 XGBoost features,
# so an upstream reweight of this checkpoint would silently shift every feature
# vector and therefore the MAPE figures this project publishes — with nothing in
# the repo changing. Pinning means a model change is a visible, deliberate diff.
#
# Resolved from the HF API on 2026-08-05; the checkpoint itself dates to
# 2023-05-23. Update deliberately after re-measuring.
FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"

# FinBERT reads at most 512 tokens, and the pipeline truncates to that itself
# (truncation=True, max_length=FINBERT_MAX_TOKENS). Headlines are also cut to
# FINBERT_INPUT_CHARS characters first: an older cap on the text handed to the
# tokenizer, not a token limit.
FINBERT_MAX_TOKENS = 512
FINBERT_INPUT_CHARS = 512

# Storage caps on feed text. The columns are TEXT, so these bound what a feed
# can make the pipeline store rather than enforce the schema.
MAX_HEADLINE_CHARS = 1000
MAX_URL_CHARS = 500

# ── RSS feed URLs per ticker ──────────────────────────────────────────────────
# Per-symbol feeds only. A general-market feed listed under one ticker credits
# every market headline to that ticker's sentiment features; MarketWatch's
# realtime headlines used to sit under AAPL that way.
RSS_FEEDS = {
    "AAPL": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US",
    ],
    "MSFT": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=MSFT&region=US&lang=en-US",
    ],
    "GOOGL": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GOOGL&region=US&lang=en-US",
    ],
    "AMZN": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AMZN&region=US&lang=en-US",
    ],
    "NVDA": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA&region=US&lang=en-US",
    ],
    "TSLA": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TSLA&region=US&lang=en-US",
    ],
    "META": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=META&region=US&lang=en-US",
    ],
    "JPM": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=JPM&region=US&lang=en-US",
    ],
}

# Company names for NewsAPI keyword search
TICKER_KEYWORDS = {
    "AAPL": "Apple stock",
    "MSFT": "Microsoft stock",
    "GOOGL": "Google Alphabet stock",
    "AMZN": "Amazon stock",
    "NVDA": "Nvidia stock",
    "TSLA": "Tesla stock",
    "META": "Meta Facebook stock",
    "JPM": "JPMorgan stock",
}


# ── FinBERT model ─────────────────────────────────────────────────────────────

_finbert_pipeline = None


def get_finbert():
    """Load FinBERT model once and cache it."""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        logger.info("Loading FinBERT model (first run takes ~2 min to download)...")
        try:
            from transformers import pipeline

            _finbert_pipeline = pipeline(
                "text-classification",
                model=FINBERT_MODEL,
                tokenizer=FINBERT_MODEL,
                revision=FINBERT_REVISION,
                top_k=None,  # return all 3 class scores
                device=-1,  # CPU (use 0 for GPU if available)
                truncation=True,
                max_length=FINBERT_MAX_TOKENS,
            )
            logger.info("FinBERT loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            return None
    return _finbert_pipeline


def _score(results: list[dict]) -> dict:
    """FinBERT's label scores for one headline as the stored fields.

    Compound score = positive probability - negative probability, and the
    dominant label is the stored sentiment.
    """
    scores = {r["label"].lower(): r["score"] for r in results}
    score_pos = scores.get("positive", 0.0)
    score_neg = scores.get("negative", 0.0)
    score_neu = scores.get("neutral", 0.0)
    return {
        "sentiment": max(scores, key=lambda label: scores[label]),
        "score_pos": round(score_pos, 4),
        "score_neg": round(score_neg, 4),
        "score_neu": round(score_neu, 4),
        "compound": round(score_pos - score_neg, 4),
    }


# A headline FinBERT could not score gets None, never a made-up neutral score.
# Stored as "neutral" with compound 0.0 it would be indistinguishable from news
# that really was neutral, and it would feed the model's sentiment features.


def analyze_headline(headline: str) -> dict | None:
    """
    Run FinBERT on a single headline.
    Returns dict with sentiment, score_pos, score_neg, score_neu, compound,
    or None when FinBERT is unavailable or fails on this headline.
    """
    model = get_finbert()
    if model is None:
        return None

    try:
        return _score(model(headline[:FINBERT_INPUT_CHARS])[0])
    except Exception as e:
        logger.warning(f"FinBERT failed on headline: {e}")
        return None


def analyze_batch(headlines: list[str]) -> list[dict | None]:
    """Run FinBERT on a batch of headlines (faster than one at a time).
    Entries are None for headlines that could not be scored."""
    if not headlines:
        return []
    model = get_finbert()
    if model is None:
        return [None] * len(headlines)

    try:
        clipped = [h[:FINBERT_INPUT_CHARS] for h in headlines]
        return [_score(results) for results in model(clipped)]
    except Exception as e:
        logger.warning(f"Batch FinBERT failed: {e}")
        return [analyze_headline(h) for h in headlines]


# ── News fetchers ─────────────────────────────────────────────────────────────


def fetch_rss(ticker: str, feeds: list[str] | None = None) -> list[dict]:
    """
    Fetch headlines from Yahoo Finance RSS for a ticker.
    Completely free, no API key needed.
    Returns list of {headline, url, published_at, source} dicts.

    feeds: the feed URLs to read. None, the default and the core's path,
    reads the ticker's RSS_FEEDS entry. The trending tier passes the feed it
    builds for a symbol not in RSS_FEEDS (docs/trending.md, section 7.5).
    """
    feeds = RSS_FEEDS.get(ticker, []) if feeds is None else feeds
    articles = []
    cutoff = datetime.now(UTC) - timedelta(days=7)

    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                # Parse published date
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    year, month, day, hour, minute, second = entry.published_parsed[:6]
                    published = datetime(
                        year, month, day, hour, minute, second, tzinfo=UTC
                    )
                else:
                    published = datetime.now(UTC)

                if published < cutoff:
                    continue

                headline = entry.get("title", "").strip()
                if not headline:
                    continue

                articles.append(
                    {
                        "headline": headline,
                        "url": entry.get("link", ""),
                        "published_at": published,
                        "source": "rss",
                    }
                )

        except Exception as e:
            logger.warning(f"RSS fetch failed for {ticker} ({feed_url}): {e}")

    logger.info(f"  RSS: {len(articles)} articles for {ticker}")
    return articles


def fetch_newsapi(ticker: str) -> list[dict]:
    """
    Fetch headlines from NewsAPI for a ticker.
    Free tier: 100 requests/day, last 30 days of news.
    Returns list of {headline, url, published_at, source} dicts.
    """
    if not NEWS_API_KEY:
        logger.warning("NEWS_API_KEY not set — skipping NewsAPI fetch")
        return []

    try:
        from newsapi import NewsApiClient

        client = NewsApiClient(api_key=NEWS_API_KEY)

        keyword = TICKER_KEYWORDS.get(ticker, ticker)
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        response = client.get_everything(
            q=keyword,
            from_param=from_date,
            language="en",
            sort_by="publishedAt",
            page_size=20,
        )

        articles = []
        for article in response.get("articles", []):
            headline = article.get("title", "").strip()
            if not headline or headline == "[Removed]":
                continue

            published_str = article.get("publishedAt", "")
            try:
                published = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
            except Exception:
                published = datetime.now(UTC)

            articles.append(
                {
                    "headline": headline,
                    "url": article.get("url", ""),
                    "published_at": published,
                    "source": "newsapi",
                }
            )

        logger.info(f"  NewsAPI: {len(articles)} articles for {ticker}")
        return articles

    except Exception as e:
        logger.error(f"NewsAPI fetch failed for {ticker}: {e}")
        return []


# ── Database write ────────────────────────────────────────────────────────────


def save_sentiment(
    articles: list[dict], ticker: str, table: str = "news_sentiment"
) -> int:
    """
    Save analyzed articles to news_sentiment table.
    Returns rows actually written: headlines already stored are skipped and
    not counted, and any other write error raises. An article whose headline
    is empty once cleaned (no title, or only NUL bytes) is not news, and is
    skipped with a log line.

    table: news_sentiment by default, the core's; the trending tier passes
    trending_news, its own twin of it (docs/trending.md, section 7.5).

    RSS and NewsAPI can return the same headline at the same time, so one
    batch can carry a key twice. ON CONFLICT DO NOTHING writes it once and
    counts it once on Postgres (tests/integration checks this), so no
    de-duplication is needed here.
    """
    rows = []
    empty = 0
    for article in articles:
        headline = _clean_text(article["headline"], MAX_HEADLINE_CHARS)
        if not headline:
            empty += 1
            continue
        rows.append(
            {
                "ticker": ticker,
                "source": article["source"],
                "published_at": article["published_at"],
                "headline": headline,
                "url": _clean_text(article.get("url"), MAX_URL_CHARS),
                "sentiment": article["sentiment"],
                "score_pos": article["score_pos"],
                "score_neg": article["score_neg"],
                "score_neu": article["score_neu"],
                "compound": article["compound"],
            }
        )

    if empty:
        logger.info(f"  {ticker}: skipped {empty} articles with no headline text")
    return insert_sentiment(rows, table=table)


# ── Display helpers ───────────────────────────────────────────────────────────


def show_sentiment(ticker: str, n: int = 15):
    """Print latest sentiment rows for a ticker."""
    engine = get_engine()
    query = text("""
        SELECT
            DATE(published_at)              AS date,
            LEFT(headline, 60)              AS headline,
            sentiment,
            ROUND(compound::numeric, 3)     AS compound,
            source
        FROM news_sentiment
        WHERE ticker = :ticker
        ORDER BY published_at DESC
        LIMIT :n
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker, "n": n})

    if df.empty:
        print(
            f"No sentiment data for {ticker}. Run: python -m quantflow.features.sentiment --ticker {ticker}"
        )
        return

    pos = len(df[df["sentiment"] == "positive"])
    neg = len(df[df["sentiment"] == "negative"])
    neu = len(df[df["sentiment"] == "neutral"])
    avg = df["compound"].mean()

    print(f"\n{'=' * 75}")
    print(f"  Sentiment for {ticker} — latest {n} headlines")
    print(
        f"  Positive: {pos} | Negative: {neg} | Neutral: {neu} | Avg compound: {avg:.3f}"
    )
    print(f"{'=' * 75}")
    print(df.to_string(index=False))
    print()


def show_summary():
    """Print market-wide sentiment overview."""
    engine = get_engine()
    query = text("""
        SELECT
            ticker,
            COUNT(*)                                            AS total_articles,
            ROUND(AVG(compound)::numeric, 3)                   AS avg_compound,
            COUNT(*) FILTER (WHERE sentiment='positive')       AS positive,
            COUNT(*) FILTER (WHERE sentiment='negative')       AS negative,
            COUNT(*) FILTER (WHERE sentiment='neutral')        AS neutral,
            DATE(MAX(published_at))                            AS latest_article
        FROM news_sentiment
        WHERE published_at >= NOW() - INTERVAL '7 days'
        GROUP BY ticker
        ORDER BY avg_compound DESC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        print("No sentiment data yet. Run: python -m quantflow.features.sentiment")
        return

    print(f"\n{'=' * 75}")
    print("  Market Sentiment Overview — last 7 days")
    print(f"{'=' * 75}")
    print(df.to_string(index=False))
    print()
    print("  Compound score: +1.0 = fully positive, -1.0 = fully negative")
    print()


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run(tickers: list[str] | None = None) -> dict:
    """
    Full pipeline: fetch news → run FinBERT → save to DB.
    Returns dict of {ticker: rows_inserted}.
    """
    tickers = tickers or TICKERS
    results = {}

    # Preload FinBERT once before the loop. Without it nothing is written:
    # storing unscored headlines as neutral would read as real neutral news.
    # It raises rather than returning zeros, so the CLI exits non-zero and the
    # scheduler logs a failed job instead of "0 rows saved".
    logger.info("Preloading FinBERT model...")
    if get_finbert() is None:
        raise FinBERTUnavailableError(
            "FinBERT could not be loaded, so no sentiment rows were written"
        )

    for ticker in tickers:
        logger.info(f"Processing sentiment for {ticker}...")

        # Fetch from both sources
        rss_articles = fetch_rss(ticker)
        newsapi_articles = fetch_newsapi(ticker)
        all_articles = rss_articles + newsapi_articles

        if not all_articles:
            logger.warning(f"  No articles found for {ticker}")
            results[ticker] = 0
            continue

        # Batch analyze with FinBERT
        headlines = [a["headline"] for a in all_articles]
        logger.info(f"  Running FinBERT on {len(headlines)} headlines...")
        scores = analyze_batch(headlines)

        # Merge scores back into articles; unscored headlines are not stored
        scored = [
            {**article, **score}
            for article, score in zip(all_articles, scores, strict=True)
            if score is not None
        ]
        if len(scored) < len(all_articles):
            logger.warning(
                f"  {ticker}: {len(all_articles) - len(scored)} headlines FinBERT "
                f"could not score were not stored"
            )

        # Save to DB
        n = save_sentiment(scored, ticker)
        results[ticker] = n
        logger.info(f"  {ticker}: {n} sentiment rows saved")

        # Small delay to be polite to APIs
        time.sleep(1)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Exits 1 when FinBERT cannot be loaded, so the daily refresh shows a
    failed run; its later steps still run, on the sentiment already stored."""
    parser = argparse.ArgumentParser(description="News sentiment analysis engine")
    parser.add_argument("--ticker", type=str, help="Single ticker (e.g. AAPL)")
    parser.add_argument("--show", type=str, help="Print sentiment for a ticker")
    parser.add_argument("--summary", action="store_true", help="Market-wide overview")
    parser.add_argument("--rows", type=int, default=15, help="Rows to show")
    args = parser.parse_args(argv)

    if args.summary:
        show_summary()
        return 0
    if args.show:
        show_sentiment(args.show.upper(), n=args.rows)
        return 0

    tickers = [args.ticker.upper()] if args.ticker else None
    if tickers is None:
        logger.info("Running sentiment pipeline for all tickers...")
    try:
        results = run(tickers)
    except FinBERTUnavailableError as e:
        logger.error(str(e))
        print(f"SENTIMENT FAILED: {e}", file=sys.stderr)
        return 1

    if tickers:
        print(f"\nDone: {results}")
        return 0
    total = sum(results.values())
    logger.info(f"Complete. Total sentiment rows saved: {total}")
    for ticker, n in results.items():
        print(f"  {ticker}: {n} rows")
    print("\nTo view results:  python -m quantflow.features.sentiment --show AAPL")
    print("Market overview:  python -m quantflow.features.sentiment --summary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
