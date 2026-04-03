"""
sentiment.py
------------
Phase 6 — News Sentiment Analysis Engine

Pulls financial news from two sources:
  1. NewsAPI       : Structured API, 100 req/day free tier
  2. RSS Feeds     : Yahoo Finance, Reuters, MarketWatch — completely free

Analyzes headlines using FinBERT:
  - ProsusAI/finbert — pre-trained on financial news
  - Returns: positive / negative / neutral + probabilities
  - Compound score = pos_prob - neg_prob (range: -1 to +1)

Stores results in news_sentiment table.
Sentiment scores can then be used as features in forecasting models.

Usage:
    python sentiment.py                    # fetch + analyze all tickers
    python sentiment.py --ticker AAPL      # one ticker only
    python sentiment.py --show AAPL        # print latest sentiment for AAPL
    python sentiment.py --summary          # market-wide sentiment overview
    python sentiment.py --analyze          # rerun FinBERT on unscored rows
"""

import sys
import os
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

import feedparser
import pandas as pd
from datetime import datetime, timedelta, timezone
from sqlalchemy import text

from config import TICKERS, NEWS_API_KEY
from db.connection import get_engine
from utils.logger import get_logger

logger = get_logger("sentiment")

# ── RSS feed URLs per ticker ──────────────────────────────────────────────────
RSS_FEEDS = {
    "AAPL":  [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US",
        "https://feeds.marketwatch.com/marketwatch/realtimeheadlines/",
    ],
    "MSFT":  [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=MSFT&region=US&lang=en-US",
    ],
    "GOOGL": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GOOGL&region=US&lang=en-US",
    ],
    "AMZN":  [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AMZN&region=US&lang=en-US",
    ],
    "NVDA":  [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA&region=US&lang=en-US",
    ],
    "TSLA":  [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TSLA&region=US&lang=en-US",
    ],
    "META":  [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=META&region=US&lang=en-US",
    ],
    "JPM":   [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=JPM&region=US&lang=en-US",
    ],
}

# Company names for NewsAPI keyword search
TICKER_KEYWORDS = {
    "AAPL":  "Apple stock",
    "MSFT":  "Microsoft stock",
    "GOOGL": "Google Alphabet stock",
    "AMZN":  "Amazon stock",
    "NVDA":  "Nvidia stock",
    "TSLA":  "Tesla stock",
    "META":  "Meta Facebook stock",
    "JPM":   "JPMorgan stock",
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
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                top_k=None,          # return all 3 class scores
                device=-1,           # CPU (use 0 for GPU if available)
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            return None
    return _finbert_pipeline


def analyze_headline(headline: str) -> dict:
    """
    Run FinBERT on a single headline.
    Returns dict with sentiment, score_pos, score_neg, score_neu, compound.
    """
    model = get_finbert()
    if model is None:
        return {"sentiment": "neutral", "score_pos": 0.33, "score_neg": 0.33,
                "score_neu": 0.34, "compound": 0.0}

    try:
        results = model(headline[:512])[0]   # truncate to 512 tokens

        scores = {r["label"].lower(): r["score"] for r in results}
        score_pos = scores.get("positive", 0.0)
        score_neg = scores.get("negative", 0.0)
        score_neu = scores.get("neutral",  0.0)

        # Compound score: positive signal - negative signal
        compound = round(score_pos - score_neg, 4)

        # Dominant sentiment
        sentiment = max(scores, key=scores.get)

        return {
            "sentiment": sentiment,
            "score_pos": round(score_pos, 4),
            "score_neg": round(score_neg, 4),
            "score_neu": round(score_neu, 4),
            "compound":  compound,
        }
    except Exception as e:
        logger.warning(f"FinBERT failed on headline: {e}")
        return {"sentiment": "neutral", "score_pos": 0.33, "score_neg": 0.33,
                "score_neu": 0.34, "compound": 0.0}


def analyze_batch(headlines: list[str]) -> list[dict]:
    """Run FinBERT on a batch of headlines (faster than one at a time)."""
    model = get_finbert()
    if model is None or not headlines:
        return [{"sentiment": "neutral", "score_pos": 0.33, "score_neg": 0.33,
                 "score_neu": 0.34, "compound": 0.0}] * len(headlines)

    try:
        # Truncate each headline
        truncated = [h[:512] for h in headlines]
        batch_results = model(truncated)

        output = []
        for results in batch_results:
            scores = {r["label"].lower(): r["score"] for r in results}
            score_pos = scores.get("positive", 0.0)
            score_neg = scores.get("negative", 0.0)
            score_neu = scores.get("neutral",  0.0)
            compound  = round(score_pos - score_neg, 4)
            sentiment = max(scores, key=scores.get)
            output.append({
                "sentiment": sentiment,
                "score_pos": round(score_pos, 4),
                "score_neg": round(score_neg, 4),
                "score_neu": round(score_neu, 4),
                "compound":  compound,
            })
        return output

    except Exception as e:
        logger.warning(f"Batch FinBERT failed: {e}")
        return [analyze_headline(h) for h in headlines]


# ── News fetchers ─────────────────────────────────────────────────────────────

def fetch_rss(ticker: str) -> list[dict]:
    """
    Fetch headlines from Yahoo Finance RSS for a ticker.
    Completely free, no API key needed.
    Returns list of {headline, url, published_at, source} dicts.
    """
    feeds = RSS_FEEDS.get(ticker, [])
    articles = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)

    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                # Parse published date
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                else:
                    published = datetime.now(timezone.utc)

                if published < cutoff:
                    continue

                headline = entry.get("title", "").strip()
                if not headline:
                    continue

                articles.append({
                    "headline":     headline,
                    "url":          entry.get("link", ""),
                    "published_at": published,
                    "source":       "rss",
                })

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
                published = datetime.fromisoformat(
                    published_str.replace("Z", "+00:00")
                )
            except Exception:
                published = datetime.now(timezone.utc)

            articles.append({
                "headline":     headline,
                "url":          article.get("url", ""),
                "published_at": published,
                "source":       "newsapi",
            })

        logger.info(f"  NewsAPI: {len(articles)} articles for {ticker}")
        return articles

    except Exception as e:
        logger.error(f"NewsAPI fetch failed for {ticker}: {e}")
        return []


# ── Database write ────────────────────────────────────────────────────────────

def save_sentiment(articles: list[dict], ticker: str) -> int:
    """
    Save analyzed articles to news_sentiment table.
    Skips duplicates. Returns rows inserted.
    """
    if not articles:
        return 0

    engine = get_engine()
    inserted = 0

    with engine.begin() as conn:
        for article in articles:
            try:
                conn.execute(
                    text("""
                        INSERT INTO news_sentiment
                            (ticker, source, published_at, headline, url,
                             sentiment, score_pos, score_neg, score_neu, compound)
                        VALUES
                            (:ticker, :source, :published_at, :headline, :url,
                             :sentiment, :score_pos, :score_neg, :score_neu, :compound)
                        ON CONFLICT (ticker, headline, published_at) DO NOTHING
                    """),
                    {
                        "ticker":       ticker,
                        "source":       article["source"],
                        "published_at": article["published_at"],
                        "headline":     article["headline"][:1000],
                        "url":          article.get("url", "")[:500],
                        "sentiment":    article["sentiment"],
                        "score_pos":    article["score_pos"],
                        "score_neg":    article["score_neg"],
                        "score_neu":    article["score_neu"],
                        "compound":     article["compound"],
                    }
                )
                inserted += 1
            except Exception as e:
                logger.warning(f"Row skipped: {e}")

    return inserted


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
        print(f"No sentiment data for {ticker}. Run: python sentiment.py --ticker {ticker}")
        return

    pos = len(df[df["sentiment"] == "positive"])
    neg = len(df[df["sentiment"] == "negative"])
    neu = len(df[df["sentiment"] == "neutral"])
    avg = df["compound"].mean()

    print(f"\n{'='*75}")
    print(f"  Sentiment for {ticker} — latest {n} headlines")
    print(f"  Positive: {pos} | Negative: {neg} | Neutral: {neu} | Avg compound: {avg:.3f}")
    print(f"{'='*75}")
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
        print("No sentiment data yet. Run: python sentiment.py")
        return

    print(f"\n{'='*75}")
    print("  Market Sentiment Overview — last 7 days")
    print(f"{'='*75}")
    print(df.to_string(index=False))
    print()
    print("  Compound score: +1.0 = fully positive, -1.0 = fully negative")
    print()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(tickers: list[str] = None) -> dict:
    """
    Full pipeline: fetch news → run FinBERT → save to DB.
    Returns dict of {ticker: rows_inserted}.
    """
    tickers = tickers or TICKERS
    results = {}

    # Preload FinBERT once before the loop
    logger.info("Preloading FinBERT model...")
    get_finbert()

    for ticker in tickers:
        logger.info(f"Processing sentiment for {ticker}...")

        # Fetch from both sources
        rss_articles     = fetch_rss(ticker)
        newsapi_articles = fetch_newsapi(ticker)
        all_articles     = rss_articles + newsapi_articles

        if not all_articles:
            logger.warning(f"  No articles found for {ticker}")
            results[ticker] = 0
            continue

        # Batch analyze with FinBERT
        headlines = [a["headline"] for a in all_articles]
        logger.info(f"  Running FinBERT on {len(headlines)} headlines...")
        scores = analyze_batch(headlines)

        # Merge scores back into articles
        for article, score in zip(all_articles, scores):
            article.update(score)

        # Save to DB
        n = save_sentiment(all_articles, ticker)
        results[ticker] = n
        logger.info(f"  {ticker}: {n} sentiment rows saved")

        # Small delay to be polite to APIs
        time.sleep(1)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="News sentiment analysis engine")
    parser.add_argument("--ticker",  type=str, help="Single ticker (e.g. AAPL)")
    parser.add_argument("--show",    type=str, help="Print sentiment for a ticker")
    parser.add_argument("--summary", action="store_true", help="Market-wide overview")
    parser.add_argument("--rows",    type=int, default=15, help="Rows to show")
    args = parser.parse_args()

    if args.summary:
        show_summary()
    elif args.show:
        show_sentiment(args.show.upper(), n=args.rows)
    elif args.ticker:
        results = run([args.ticker.upper()])
        print(f"\nDone: {results}")
    else:
        logger.info("Running sentiment pipeline for all tickers...")
        results = run()
        total = sum(results.values())
        logger.info(f"Complete. Total sentiment rows saved: {total}")
        for ticker, n in results.items():
            print(f"  {ticker}: {n} rows")
        print("\nTo view results:  python sentiment.py --show AAPL")
        print("Market overview:  python sentiment.py --summary")