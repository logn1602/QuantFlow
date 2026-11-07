"""Readers and the writer for the raw_prices table.

raw_prices holds two kinds of bar. Seeding writes one daily bar per trading
date; the 15-minute refreshes write intraday bars. Anything that wants one
row per trading date, which is every model and the dashboard, goes through
daily_bars, the single definition of how those two kinds combine:

  - a date with a daily bar uses it;
  - a date with only 15-min bars gets one bar built from them: first open,
    highest high, lowest low, last close, summed volume.

The loaders used to collapse a date to its first row by timestamp instead.
Seeded dates have a midnight daily bar, so they were fine, but every date
after the seed has only 15-min bars, and its "daily close" became the close of
the 09:30 bar. Building the day bar in pandas rather than SQL keeps the rule
testable without a database.

Both fetchers write through insert_bars, so the conflict rule for raw_prices
lives in one place too.
"""

import pandas as pd
from sqlalchemy import text

from quantflow.db.batch import insert_rows
from quantflow.db.connection import get_engine
from quantflow.utils.logger import get_logger
from quantflow.utils.trading_calendar import EXCHANGE_TZ

logger = get_logger(__name__)

# Column order of a raw_prices row as the fetchers write it.
RAW_PRICE_COLUMNS = ("ticker", "source", "ts", "open", "high", "low", "close", "volume")

# What daily_bars returns: one row per trading date.
DAILY_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
_PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]

# yfinance stamps 15-min bars with their start time on the regular-session
# grid: 09:30 to 15:45 New York, on the quarter hour. Daily bars sit exactly at
# a midnight: 00:00 UTC from today's yfinance, or 00:00 New York from older
# versions. A row that is neither is left out rather than guessed at.
_FIRST_BAR_MINUTE = 9 * 60 + 30
_LAST_BAR_MINUTE = 15 * 60 + 45
_BAR_MINUTES = 15


def _read(sql: str, params: dict, ticker: str) -> pd.DataFrame:
    """Run a raw_prices query and warn once if it returns nothing."""
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    if df.empty:
        logger.warning(f"No price data found for {ticker}")
    return df


def _now() -> pd.Timestamp:
    """The current time; a seam so tests can fix it."""
    return pd.Timestamp.now(tz="UTC")


def classify_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """bars plus three columns: date, is_daily and is_intraday.

    date is the bar's UTC calendar date as a naive midnight, which is what the
    pipeline's ts::date has always produced in Supabase's UTC sessions and the
    right trading date for every timestamp shape stored here.

    A row with both flags False is neither a daily bar nor a 15-min bar: a
    16:00 print, a bar stamped with seconds. Counted as a daily bar, as the
    first version of this rule did, it replaced the whole day.
    """
    ts = pd.to_datetime(bars["ts"], utc=True)
    local = ts.dt.tz_convert(EXCHANGE_TZ)
    minute = local.dt.hour * 60 + local.dt.minute
    on_grid = (
        (local.dt.second == 0)
        & (local.dt.microsecond == 0)
        & (minute % _BAR_MINUTES == 0)
    )
    return bars.assign(
        ts=ts,
        date=ts.dt.tz_localize(None).dt.normalize(),
        is_daily=(ts == ts.dt.normalize()) | (local == local.dt.normalize()),
        is_intraday=on_grid & minute.between(_FIRST_BAR_MINUTE, _LAST_BAR_MINUTE),
    )


def daily_bars(bars: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """One OHLCV row per trading date from a mix of daily and 15-min bars.

    bars needs ts, open, high, low, close and volume. Logs a warning naming
    how many dates had no daily bar, because those closes come from the last
    15-min bar rather than Yahoo's official close (they differ by a few
    hundredths of a percent, which is why the refresh now stores daily bars).
    """
    if bars.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)

    frame = classify_bars(bars).sort_values("ts", kind="stable")
    frame[_PRICE_COLUMNS] = frame[_PRICE_COLUMNS].astype(float)

    stray = frame[~frame["is_daily"] & ~frame["is_intraday"]]
    if not stray.empty:
        logger.warning(
            f"{ticker}: ignored {len(stray)} bars off the session grid "
            f"(neither 00:00 nor 09:30 to 15:45 New York on the quarter hour), "
            f"{stray['ts'].min()} to {stray['ts'].max()}"
        )

    daily = frame[frame["is_daily"]].drop_duplicates("date", keep="first")
    intraday = frame[frame["is_intraday"] & ~frame["date"].isin(daily["date"])]
    built = (
        intraday.groupby("date", sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            # min_count=1: a day whose bars all lack volume stays missing
            # rather than reading as zero shares traded.
            volume=("volume", lambda v: v.sum(min_count=1)),
        )
        .reset_index()
    )

    if not built.empty:
        logger.warning(
            f"{ticker}: {len(built)} of {len(daily) + len(built)} trading dates "
            f"have no daily bar; built from 15-min bars "
            f"({built['date'].min().date()} to {built['date'].max().date()})"
        )

    parts = [part[DAILY_COLUMNS] for part in (daily, built) if not part.empty]
    out = pd.concat(parts, ignore_index=True) if parts else daily[DAILY_COLUMNS]
    return out.sort_values("date").reset_index(drop=True)


def load_daily_bars(
    ticker: str, source: str = "yfinance", days: int | None = None
) -> pd.DataFrame:
    """One OHLCV row per trading date: date, open, high, low, close, volume.

    days limits the read to the trailing days that start at a UTC midnight.
    Cut at NOW() minus N days instead, the first date lost its 00:00 UTC daily
    bar but kept its later 15-min bars, and was built from part of a session.
    """
    if days is None:
        sql = """
            SELECT ts, open, high, low, close, volume
            FROM raw_prices
            WHERE ticker = :ticker AND source = :source
            ORDER BY ts ASC
        """
        params: dict = {"ticker": ticker, "source": source}
    else:
        sql = """
            SELECT ts, open, high, low, close, volume
            FROM raw_prices
            WHERE ticker = :ticker AND source = :source
              AND ts >= :since
            ORDER BY ts ASC
        """
        since = _now().normalize() - pd.Timedelta(days=days)
        params = {"ticker": ticker, "source": source, "since": since}

    bars = _read(sql, params, ticker)
    if bars.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    return daily_bars(bars, ticker)


def load_bar_timestamps(ticker: str, source: str = "yfinance") -> pd.DataFrame:
    """Every stored bar's timestamp for a ticker, as a ts column.

    Enough for classify_bars to tell which dates have a daily bar, which is
    all the daily-bar backfill needs to know.
    """
    return _read(
        """
        SELECT ts
        FROM raw_prices
        WHERE ticker = :ticker AND source = :source
        ORDER BY ts ASC
        """,
        {"ticker": ticker, "source": source},
        ticker,
    )


def _indexed_by_date(daily: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """daily_bars output indexed by each date's 00:00 UTC timestamp.

    Indicators and anomaly flags are stored against that timestamp, the same
    one daily bars carry, so a date's derived rows line up with its price.
    """
    index = pd.DatetimeIndex(daily["date"]).tz_localize("UTC").rename("ts")
    return daily.set_index(index)[columns]


def load_daily_ohlcv(ticker: str, source: str = "yfinance") -> pd.DataFrame:
    """Daily OHLCV, indexed by date at 00:00 UTC, all columns float.

    Used by the indicator engine, which needs every column to compute
    Bollinger Bands and MACD. Daily rather than per bar: computed over the
    raw table, the 14-, 26- and 20-period windows would span hours of 15-min
    bars on every date after the seed.
    """
    daily = load_daily_bars(ticker, source)
    if daily.empty:
        return daily
    return _indexed_by_date(daily, _PRICE_COLUMNS)


def load_daily_close_volume(ticker: str, source: str = "yfinance") -> pd.DataFrame:
    """Daily close and volume, indexed by date at 00:00 UTC.

    Used by anomaly detection, whose 30-row Z-score window is meant to be 30
    trading days.
    """
    daily = load_daily_bars(ticker, source)
    if daily.empty:
        return daily
    return _indexed_by_date(daily, ["close", "volume"])


def load_daily_close(ticker: str, source: str = "yfinance") -> pd.DataFrame:
    """One close per trading date, as columns ds and y.

    The ds/y naming is Prophet's required input contract, and every forecasting
    model consumes this shape.
    """
    daily = load_daily_bars(ticker, source)
    return daily.rename(columns={"date": "ds", "close": "y"})[["ds", "y"]]


def load_recent_ohlcv(ticker: str, days: int = 180) -> pd.DataFrame:
    """Daily OHLCV over a trailing window, as a date column rather than an index.

    Dashboard-facing: one candle per trading date. An intraday chart, if one
    is ever wanted, should get its own explicitly named loader rather than
    reusing this one.
    """
    return load_daily_bars(ticker, days=days)


def insert_bars(rows: list[dict]) -> int:
    """Write raw_prices rows; return how many Postgres actually wrote.

    A bar already stored for the same (ticker, source, ts) is skipped and not
    counted, which is the normal case: every fetch re-requests a window that
    overlaps the previous one. Any other write error raises and rolls the
    whole call back.
    """
    if not rows:
        return 0
    with get_engine().begin() as conn:
        return insert_rows(
            conn, "raw_prices", rows, skip_conflicts=("ticker", "source", "ts")
        )
