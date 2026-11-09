# Trending tier: design

Phase 5 of the work brief (sections 6 and 7), written 2026-11-09 before any trending
code exists. Nothing here is built yet: Checkpoint 5 decides it, and Phase 6 implements
what is approved, on `feat/trending`. Already decided by the owner and applied below:
the trending tier stores daily bars only (2026-11-07); its news is RSS only, like the
core's (D18); the split guard's jump check warns but never blocks for trending tickers
(2026-11-07); `TRENDING_PINNED` starts with `SPCX`; and this document states the storage
and egress the tier adds (section 9).

This version keeps the tier apart from the core by construction: its own workflow, its
own tables, and its prices under a source value of its own. The first draft shared the
core's tables and workflow and kept them apart with guards and filters; the review
(gstack `/plan-eng-review` and an independent second reader) showed those guards fail
on a wrong `TICKERS`, and that sharing the daily job's timeout and test step put the
forward tests at risk. Section 18 lists every choice left for the checkpoint, with the
first draft's alternative.

## 1. Goal and the two tiers

Surface the stocks people are watching and trading heavily right now, without
contaminating the evaluated core set.

| | Core watchlist | Trending tier |
|---|---|---|
| Which symbols | `TICKERS` in config, unchanged: the source of truth for CI, the workflow and Streamlit Cloud | Discovered every run from Yahoo's lists, plus `TRENDING_PINNED` |
| Prices | `raw_prices`, source `yfinance`: daily bars and 15-minute bars, never rewritten | `raw_prices`, source `trending`: daily bars only, about 6 months on first sight, refreshed from Yahoo every run |
| Indicators, anomaly flags | Stored by their stages, every run | Computed by the same functions; a summary stored, the rest computed when shown |
| News sentiment | RSS (D18), in `news_sentiment` | RSS, the same feed format, in `trending_news` |
| Forecasts, ensemble, backtest | Yes | **No** |
| Evaluated by the harness | Yes | No |
| Runs in | `daily_refresh.yml` | `trending.yml`, after it |

Why the trending tier is discovery and analytics only:
- **History.** Many trending names have little: new listings, recent spin-offs, names
  that only just crossed the lists' floors. The harness needs 306 training rows at its
  first validation origin, and the core has 648 trading dates; a 6-month history cannot
  be scored on the same windows.
- **Selection bias.** A stock trends *because* it just moved. A set chosen each day for
  its recent moves is selected on the outcome: models scored on it would measure the
  selection, not their skill, and any edge they showed would not carry to a set chosen
  in advance.
- **Churn.** The lists change daily, so a results table over "the trending stocks" would
  describe a different set every run, and no two runs could be compared.

What the lists measure, honestly: what moved most and traded most today, among large US
stocks. That is a fair proxy for "watched", but not attention itself; a mentions-based
source would measure that (section 3, not built).

## 2. Isolation: what the core never sees

The core is under two pre-registered forward tests (docs/evaluation.md, D16 to D19: 42
origins from 2026-11-09, scored once after 2027-09-15). The tier is designed so that
nothing it does can reach them:

| Core resource | Who reads it | How the tier stays out |
|---|---|---|
| `raw_prices`, source `yfinance` | Every price stage, the split guard, the backfill, the dashboard's latest closes and the snapshot export each read `source = 'yfinance'` only (`db/prices.load_daily_bars` and its wrappers default to it), as do the 1a diagnostics; the free-tier counts group by source, so trending rows show there as their own line | Trending bars are stored under source `trending`, and the tier's only delete matches `source = 'trending'`, so no setting, `TICKERS` included, can make it reach a core row |
| `news_sentiment` | The core features' news read and D6's "a fetch happened" signal (`engineering.load_features`), the snapshot export, the dashboard | Trending articles go to `trending_news`; nothing that reads `news_sentiment` changes |
| `technical_indicators`, `anomalies`, `forecasts`, `model_metrics`, `backtest_results` | The dashboard, the models | The tier writes none of them |
| `daily_refresh.yml`: its steps, inputs, exit codes, 45-minute job timeout, test step and FinBERT cache | The forward tests' daily news and forecasts | The tier runs in its own workflow (section 12); its tests are marked and left out of the daily test step |
| Core modules | Everything | Two optional parameters with defaults that keep today's behaviour, each proved by a test: the feed list of `sentiment.fetch_rss`, and the table that `sentiment.save_sentiment` passes on to `db.sentiment.insert_sentiment` (section 7.5). Nothing else in a core module changes |

So the core's news read, D6's fetch signal and the dashboard's core views need no
filter: trending data is never in the tables they read. A watchlist change is a
different matter (section 17).

## 3. Where the lists come from

### Interface

```python
@dataclass(frozen=True)
class TrendingItem:
    symbol: str  # validated (section 4.1) before anything else sees it
    source: str  # "yahoo_screener"
    list_name: str  # "most_actives", "day_gainers", "day_losers"
    rank: int  # 1-based position in that list
    pct_change: float | None
    volume: int | None
    market_cap: int | None
    price: float | None  # for the minimum price (section 4)
    quote_type: str | None  # "EQUITY", "ETF", ...
    short_name: str | None  # cleaned, at most 100 characters
    fetched_at: datetime  # UTC


class TrendingSource(Protocol):
    name: str

    def fetch(self) -> list[TrendingItem]:
        """Every item on the source's lists, or [] on any failure (logged)."""
```

These are the brief's fields plus `price`, which the minimum-price filter needs. A
source never raises: these are unofficial endpoints, so any failure (network, an HTTP
status, a malformed body, a field of the wrong type) logs a warning and returns `[]`. An
item whose symbol fails validation is dropped and logged; the rest of the list is kept.

### Yahoo's predefined screeners (implemented first)

The locked yfinance (1.7.0; `pyproject.toml` floors it at 1.5.2) exposes a screener API:
`yf.screen(name, count=25)`, with Yahoo's predefined lists named in
`yf.PREDEFINED_SCREENER_QUERIES`, among them `most_actives`, `day_gainers` and
`day_losers` (checked in the installed package's source, 2026-11-09). For a predefined
name, yfinance sends the list's id (`scrIds`) and the count to Yahoo's
predefined-screener endpoint, not a filter: Yahoo applies its own definition. It
returns the body's
`finance.result[0]` (expected to hold a `quotes` list, to be confirmed on a captured
response); it raises on an HTTP error status. `YahooScreenerSource` calls `yf.screen`
once per list (25 items each; Yahoo's cap is 250), maps each quote to a `TrendingItem`,
and catches everything. It goes through yfinance's own session, as every other Yahoo
call here does, so no second HTTP client or User-Agent is needed.

Yahoo's server-side definitions are not visible. yfinance's local copy of them suggests
market-cap floors of $2 billion on all three lists and a $5 price floor on gainers and
losers, but not on `most_actives`, and it is not what Yahoo applies. So the tier applies
its own price floor (`TRENDING_MIN_PRICE`, $5 by default, section 11).

The quote fields used (`symbol`, `shortName`, `quoteType`, `regularMarketPrice`,
`regularMarketChangePercent`, `regularMarketVolume`, `marketCap`) are confirmed on a
captured response in Phase 6, which becomes the test fixture. A missing field is `None`,
not a failure.

### `trending_tickers`: not in the first version

yfinance 1.7.0 has no trending-tickers call (searched in its source). Yahoo's trending
list sits on a separate unofficial endpoint and returns symbols without quote fields, so
using it means a second HTTP client (`requests`, with a timeout and a User-Agent naming
the project and its repository) and a quote lookup per symbol before the EQUITY filter
can apply. Recommended: leave it out of Phase 6 (decision T5).

### A social source (designed for, not built)

Reddit or StockTwits would measure attention rather than moves: mentions. What one would
need, to be checked against each provider's current terms before anything is built
(they change; nothing below was verified for this document):
- **Reddit**: an OAuth application (client id and secret, kept as secrets beside the
  database's), a descriptive User-Agent, the Data API's free-tier rate limit (100 queries
  a minute per client when last published) and its non-commercial terms; and symbol
  extraction from post text, which is noisy (tickers that are also words).
- **StockTwits**: an API key, if it accepts new applications at all, its published rate
  limits, and its own trending-symbols list.
- For either: mentions are a count, not a rank, so `TrendingItem.rank` would be the
  position in a mention-sorted list; the source name keeps them apart in the snapshots.

## 4. Filtering and limits

### 4.1 Symbol validation (untrusted input)

Symbols come from unofficial endpoints, so each is validated before it reaches the
database, a URL or a log line: upper-case ASCII letters, digits, `.` and `-` only,
starting with a letter, at most 10 characters (`raw_prices.ticker` is `VARCHAR(10)`).
Anything else is rejected and logged. `short_name` is cleaned as headlines are
(`sentiment._clean_text`: no NUL or control characters) and cut to 100 characters. SQL
is always parameterised; the RSS URL is built from a validated symbol only, and
URL-encoded.

### 4.2 The rules, in order

1. Validate (4.1); drop items that fail.
2. Keep `quote_type == "EQUITY"` only.
3. Drop core tickers (`TICKERS`). With the tier's storage apart (section 2), a wrong
   `TICKERS` here costs at most a duplicate entry in the Trending tab, never a core row.
4. Floors: `TRENDING_MIN_PRICE` ($5 by default) and `TRENDING_MIN_MARKET_CAP` (off by
   default).
5. Dedupe across lists: one entry per symbol, keeping every list name it appeared on and
   its best (lowest) rank; the other fields come from its best-ranked item.
6. Choose the active set (4.3).

### 4.3 The active set

Every run, from the deduped candidates and the stored `tickers` rows:
1. Every pinned symbol is active (section 5), outside the cap.
2. Then, up to `TRENDING_MAX_ACTIVE` (default 15) discovered symbols:
   - first, symbols on today's lists, by best rank, then higher volume, then symbol, so
     the order is deterministic;
   - then symbols active before, absent today, but seen within the last
     `TRENDING_INACTIVE_AFTER_DAYS` (default 5) fetched sessions, most recently seen
     first.
3. At most `TRENDING_MAX_NEW_PER_RUN` (default 5) symbols never seen before join per
   run: each costs a 6-month history download and its storage (section 9). A new symbol
   over the limit waits for a later run, if it is still listed.
4. An active symbol not chosen is deactivated (`active = false`); its prices, news and
   snapshots stay.

What this does in practice: the three lists carry about 60 distinct symbols after
dedupe (an estimate; Phase 6 measures it), so on a normal day the cap fills from today's lists, and the active set is
today's 15 best-ranked names, at most 5 of them new. The grace for absent symbols
matters only when the lists are short, or when one of them failed. Churn among the top
movers is high, so the new-symbol limit is the one that binds (section 9 plans on it).

Sessions, not runs: each fetch is dated by the session its lists describe, the latest
New York session closed at fetch time, because the workflow often starts after midnight
UTC and a second fetch of the same session must count once. The session date is
computed once, when the snapshot is written, and stored with it (section 6). Only
sessions whose fetch succeeded count: an outage, when every source returned `[]`,
deactivates nothing and counts towards nobody's absence, and the run fails visibly
(section 14). The selection is a pure function of its inputs, tested in Phase 6.

Why 15: every active symbol costs, each run, one Yahoo history download, an RSS fetch
and FinBERT on about 20 headlines on the runner's CPU. Fifteen plus the pinned symbols
add about twice the core's per-run work in those steps (16 symbols against 8), and stay
well inside yfinance's informal throttling. The brief's other reason, NewsAPI's 100
requests a day, no longer applies (D18).

## 5. Pinned tickers

`TRENDING_PINNED` (comma-separated, default `SPCX`; the default is kept in `config.py`,
`.env.example` and `trending.yml`, and a test keeps the three equal, as one does for
`TICKERS`): always tracked, even on days no list carries them, outside the cap and the
new-symbol limit. At most 10 are read, so the list cannot crowd out the tier's budget.
Each is validated like any other symbol (4.1); an invalid one is logged and ignored,
never fatal. Pinned symbols skip the EQUITY filter, since the owner chose them, but a
quote type other than EQUITY is logged.

On a symbol's first fetch, the log states its first stored bar date, and its `tickers`
row records it (`first_bar_date`).

**SPCX.** The symbol belonged to a Tuttle Capital ETF before the current company listed
on 2026-06-12, so its history must start at that listing, not with the old fund's
prices. A committed file, `config/trending_listing_dates.txt`, lists reused symbols in
the format of `config/split_jump_acknowledged.txt` (one `SYMBOL:YYYY-MM-DD` per line, a
reason required) and is read by the same parser (`config.split_jump_acknowledged`, which
takes a path). The fetch drops every bar before a listed date and logs how many it
dropped. The file starts with:

```
SPCX:2026-06-12  # the symbol was a Tuttle Capital ETF's before this listing
```

The first fetch also logs the first bar Yahoo returned before the cut, so a history that
already starts on 2026-06-12 is visible as such. A quote type of ETF for SPCX would mean
Yahoo still serves the old fund's metadata: logged.

## 6. Storage

Migration `sql/migrations/002_trending.sql`, idempotent (`CREATE TABLE IF NOT EXISTS`,
`CREATE INDEX IF NOT EXISTS`), mirrored in a new `sql/schema_trending.sql` and in `make
setup`. The owner runs it, in Supabase's SQL editor as `postgres`: the read-only role
reads tables `postgres` creates later through its default privileges
(`sql/schema_roles.sql`, step 4), so the dashboard reads them with no further grant. Run
as any other role, the migration needs a `GRANT SELECT ... TO quantflow_readonly`.

```sql
CREATE TABLE IF NOT EXISTS tickers (
    symbol            VARCHAR(10)  PRIMARY KEY,
    tier              VARCHAR(10)  NOT NULL DEFAULT 'trending'
                                   CHECK (tier IN ('core', 'trending')),
    short_name        VARCHAR(100),
    quote_type        VARCHAR(20),
    first_seen        TIMESTAMPTZ  NOT NULL,
    last_seen         TIMESTAMPTZ  NOT NULL,
    active            BOOLEAN      NOT NULL DEFAULT TRUE,
    pinned            BOOLEAN      NOT NULL DEFAULT FALSE,
    first_bar_date    DATE,
    first_trade_date  DATE,
    promoted_at       TIMESTAMPTZ,
    -- What the Trending tab's table shows, from the last run (section 7.4).
    summary_session   DATE,
    last_close        NUMERIC(14, 4),
    rsi_14            NUMERIC(8, 2),
    recent_flags      SMALLINT,
    big_move_date     DATE
);

CREATE TABLE IF NOT EXISTS trending_snapshots (
    snapshot_ts   TIMESTAMPTZ   NOT NULL,
    session_date  DATE          NOT NULL,
    symbol        VARCHAR(10)   NOT NULL,
    source        VARCHAR(20)   NOT NULL,
    list_name     VARCHAR(40)   NOT NULL,
    rank          SMALLINT      NOT NULL,
    pct_change    NUMERIC(10, 4),
    volume        BIGINT,
    market_cap    BIGINT,
    UNIQUE (snapshot_ts, symbol, source, list_name)
);
CREATE INDEX IF NOT EXISTS trending_snapshots_symbol_session
    ON trending_snapshots (symbol, session_date);

-- news_sentiment's columns, key and indexes (sql/schema_sentiment.sql), written out
-- rather than copied with LIKE, whose INCLUDING DEFAULTS would draw this table's ids
-- from news_sentiment's own sequence. Apart from it on purpose (section 2).
CREATE TABLE IF NOT EXISTS trending_news (
    id            BIGSERIAL PRIMARY KEY,
    ticker        VARCHAR(10)     NOT NULL,
    source        VARCHAR(30)     NOT NULL,
    published_at  TIMESTAMPTZ     NOT NULL,
    headline      TEXT            NOT NULL,
    url           TEXT,
    sentiment     VARCHAR(10)     NOT NULL,
    score_pos     NUMERIC(6, 4),
    score_neg     NUMERIC(6, 4),
    score_neu     NUMERIC(6, 4),
    compound      NUMERIC(6, 4),
    inserted_at   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE (ticker, headline, published_at)
);
CREATE INDEX IF NOT EXISTS idx_trending_news_ticker_ts
    ON trending_news (ticker, published_at DESC);
```

- `tickers` holds the trending tier's symbols only: the core list is always read from
  config, so there is no second copy of it to drift. The brief's columns, plus
  `quote_type` (the EQUITY filter's input, and the SPCX check), `first_bar_date` (the
  first fetch's log line, kept; also the replacement's guard, 7.2), `first_trade_date`
  (Yahoo's first trading date, for eligibility, section 8) and the summary the tab shows.
  `tier` becomes `'core'` for a promoted symbol once it is in `TICKERS` (section 8).
- `trending_snapshots` keeps every item fetched (after symbol validation, before the
  other rules), so any day's choice can be explained afterwards. "Days trending" is the
  number of distinct `session_date`s with an entry for the symbol, one grouped query.
- `trending_news` is `news_sentiment`'s twin, written by the same code (7.5); an
  integration test keeps their columns and key equal.
- Trending prices go in `raw_prices` with `source = 'trending'`: the same columns, the
  same daily-bar rows, and no new price table, since every core reader already filters
  on `source` (section 2).

Access goes through a new `db/trending.py`: batched writes through `db/batch.insert_rows`
(one multi-row insert per chunk, conflicts skipped and not counted), the price
replacement (7.2), and the reads the run and the dashboard need. No SQL outside `db/`
apart from the dashboard's display queries, which follow the existing pattern.

## 7. Pipeline

### 7.1 One run of the trending tier

```
 trending.yml, after the daily refresh has finished
 --------------------------------------------------
 Yahoo predefined screeners (yf.screen)         TRENDING_PINNED (SPCX)
   most_actives, day_gainers, day_losers               |
        |                                              |
        v                                              |
 validate symbols --> trending_snapshots (every item, session_date)
        |                                              |
        v                                              v
 EQUITY, not core, $5 floor, dedupe --> active set (15 + pinned, 5 new a run)
                                              |
                                              v
                                          tickers (active, first_seen, ...)
                                              |
        +-------------------------------------+------------------------+
        v                                     v                        v
 daily bars from Yahoo, replaced     compute_indicators,        RSS template feed,
 per symbol in raw_prices            detect_anomalies on        FinBERT,
 under source 'trending'      --->   the same bars: summary     trending_news
        |                            on tickers
        v
 jump check: warn and badge, never block

 Never touched: news_sentiment, the core's raw_prices rows (source 'yfinance'),
 technical_indicators, anomalies, forecasts, model_metrics, backtest_results.
```

1. Fetch every source; store every item in `trending_snapshots`, with its session date.
2. Apply the rules and choose the active set (section 4); write `tickers`.
3. Prices for every active symbol (7.2).
4. Indicators and anomaly flags for every active symbol (7.4).
5. Sentiment for every active symbol (7.5).

If the active set is empty, steps 3 to 5 are skipped: the stages treat an empty ticker
list as "every core ticker" (`tickers = tickers or TICKERS` in `indicators.run`,
`anomalies.run` and `sentiment.run`), so the tier never calls a stage's `run` at all
(7.4, 7.5) and never passes an empty list to anything.

### 7.2 Prices: daily bars only, re-fetched every run

Daily bars only, never 15-minute bars (2026-11-07): a trending name needs one row per
trading date for its indicators and its chart, and 15-minute bars would multiply its
storage by 27.

Each run, each active symbol's daily bars are downloaded from Yahoo (`auto_adjust=False`,
as the core's) from its `first_bar_date`, or 6 months back for a new symbol, shaped by
the Phase 1 daily-bar path (`yfinance_fetcher.download`, `as_daily_bars`,
`bars_to_rows`), and **replace** the symbol's `source = 'trending'` rows in one
transaction. Replacing rather than appending, as the core does, because a trending
name's prices are a cache of Yahoo's, not an evaluated history: a split or reverse split
(common among the names these lists carry) would otherwise leave the stored rows on the
old basis until rescaled by hand, while a re-fetch is on Yahoo's current basis every
day. It costs the same one download per symbol that appending would.

Guards: the delete matches `source = 'trending'` and the symbol, nothing else, so it
cannot reach a core row whatever `TICKERS` holds. The replacement is skipped, with a
warning, when the download does not reach back to the symbol's `first_bar_date`, so a
partial answer from Yahoo can never shorten a history; `first_bar_date` comes from
`tickers`, so no stored bars are read to check. The replaced rows leave dead tuples that
autovacuum reclaims: about 4,000 rows a run, under 1 MB.

### 7.3 The split guard on trending names

- **The jump check warns, never blocks** (decided 2026-11-07). It runs on each symbol's
  fresh bars in the trending run. Since those bars are Yahoo's, re-adjusted on every
  fetch, a split almost never shows as a jump in them: a split-sized move left in them is
  a real move, or Yahoo's own error. So the dashboard shows it as a "big move" badge
  (`big_move_date`), and nothing stops.
- **The basis check does not apply**: it compares stored closes with Yahoo's current
  ones, and the stored closes *are* Yahoo's current ones each day (7.2).
- The core's split guard is untouched: it reads `source = 'yfinance'` for `TICKERS` only,
  so a trending symbol can never fail it.

### 7.4 Indicators and anomaly flags

The same functions as the core's stages, `indicators.compute_indicators` and
`anomalies.detect_anomalies`, run on the bars the run has just downloaded (no read back
from the database). Six months is enough for MACD (26 rows), the Bollinger bands (20)
and the anomaly z-score's 30-row window. The run stores, on each symbol's `tickers` row,
what the tab's table shows: the session, the last close, RSI(14), the number of anomaly
flags in the last 10 trading days, and the latest big move. The full indicator and flag
series are computed again when a visitor opens a symbol's chart (section 10), from its
stored bars, by the same functions. Nothing is written to `technical_indicators` or
`anomalies`, and the stages' `run` functions are not called: they read and write the
core's tables.

Each symbol is processed on its own: one that raises is logged and the rest go on.

### 7.5 Sentiment: RSS only, in `trending_news`

News is RSS only (D18). The brief's NewsAPI parts (the `short_name` keyword fallback,
the 100-requests quota, the top-N limit on NewsAPI calls) are superseded. A trending
symbol's feed is Yahoo's per-symbol RSS feed, built from the template the core's URLs
already follow:

```
https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US
```

The run calls the sentiment module's parts, not its `run` (which would also call
`fetch_newsapi` and fall back to the core tickers on an empty list):
- `fetch_rss(symbol, feeds=[template URL])`: a new optional `feeds` parameter; without
  it, `fetch_rss` reads `RSS_FEEDS` exactly as today;
- `analyze_batch` (FinBERT, loaded once by `get_finbert`);
- `save_sentiment(scored, symbol, table="trending_news")`: a new optional `table`
  parameter, passed through to `db.sentiment.insert_sentiment`; without it, both write
  `news_sentiment` exactly as today.

Tests prove both defaults unchanged, and that the template reproduces each of the eight
core URLs in `RSS_FEEDS` exactly, so a trending feed is the same kind of feed.

### 7.6 Why the core's news read needs no change

Under D6, a date whose window saw no fetch at all has every sentiment feature missing,
and "a fetch happened" is read from every stored article's `stored_day`, for any ticker
(`engineering.load_features`; the harness reads the same from the snapshot). Trending
articles stored beside the core's would make a day with only a trending fetch count as
fetched for the core, so their features would read 0 articles instead of missing. With
`trending_news` apart, that cannot happen, and `engineering.py`, which its own docstring
calls leakage-critical, is not touched.

### 7.7 Deactivation and reactivation

A symbol absent from every list long enough is deactivated unless pinned (4.3); its
history stays. If it returns, it is active again and the next re-fetch (7.2) fills the
gap from its `first_bar_date`, with no new 6-month download.

## 8. Promotion to core (manual only)

**In Phase 6, eligibility only.** A trending symbol is shown as promotion-eligible when
it has been on the lists on at least `TRENDING_PROMOTION_MIN_DAYS` distinct sessions
(default 20, about a month) or is pinned, **and** has at least
`PROMOTION_MIN_HISTORY_DAYS` trading days of history (proposed default 504, two years).
History counts from the later of Yahoo's `first_trade_date` and the symbol's date in
the listing-date file, since Yahoo's metadata can still be a former holder's (SPCX).

Why 504, and the trade-off:
- The core's models need history: Prophet's yearly seasonality is on only from about two
  years (Phase 1, 1h); the harness's first validation origin trains on 306 rows; the
  core's 648 trading dates are what a ticker needs to be scored on every validation and
  test origin like the others.
- Lower (252, one year) marks more names sooner, but each would sit longer in the
  short-history row, with models fitted on fewer market regimes.
- Higher (648) means a promoted ticker is scored like the core at once, but few new
  names ever qualify: SPCX, listed 2026-06-12, would be eligible from about June 2028 at
  504 and June 2027 at 252. Long-listed names that trend (a large cap on `most_actives`)
  qualify after their 20 sessions whatever the threshold.

**`make promote` and the short-history row: deferred (decision T6).** Promotion changes
`TICKERS`, so during the forward window (to 2027-09-15) it is the watchlist decision of
section 17, and building it now would also change the harness's report code mid-window.
When it is built, it must:
- check eligibility (refusing otherwise unless `--force`), set `promoted_at`, and print,
  without editing anything, the `TICKERS=` line for `.env`, the workflow and Streamlit's
  secrets, and the history backfill to run (the core stores two years of daily bars
  under source `yfinance`, then 15-minute bars);
- run the split guard's jump check on that backfilled history first and list its
  split-sized moves for acknowledgement: the core's guard scans a ticker's whole stored
  history (`split_guard.check`), names that trend often moved 30% or more in a day, and
  an unacknowledged move would fail the daily refresh for every ticker;
- leave `trending_news` where it is: the promoted ticker's core news starts at its first
  core fetch, so D6 reads the days before as "not collected", not as quiet days, which is
  right, since its feed was only read on its active days;
- in results tables, put a ticker whose history does not cover the window's first origin
  in a separate short-history row, out of the headline mean until it has full history.

## 9. Storage and egress the tier adds

Estimated from the 2026-11-09 measurements (`docs/architecture.md`, "Running on free
tiers"): a price row takes about 246 bytes stored (8.9 MB for 36,144 rows, indexes
included) and about 90 bytes on the wire (a 4,418-row read was 0.40 MB); an article about
494 bytes stored (19 MB for 38,466); the core's RSS feeds bring about 11 articles a
ticker a day (August to October). Assumed: 15 discovered symbols plus SPCX active, 250
runs a year, and new symbols at the cap of 5 a run, since churn keeps it binding (4.3).
Phase 6 measures the real figures on a local copy through the byte-counting proxy, as
Phase 4 did.

### Storage, first year

| What | Size | First year |
|---|---|---|
| First-sight histories | 126 bars x 246 B = 31 KB a new symbol | 5 a run: 1,250 symbols, 39 MB; fewer later, as the names the lists carry repeat |
| Daily bars of the active set | 16 x 250 bars | 1 MB |
| `trending_news` | 16 symbols x 250 days x 494 B | 22 MB at the core's 11 a day; 40 MB at 20 a day, a full feed daily |
| `trending_snapshots` | 75 rows a run, about 160 B each | 3 MB |
| `tickers` | one row a symbol ever seen | under 0.3 MB |
| Indicator and anomaly series | not stored (7.4) | 0 |
| Dead tuples from the daily price replacement | about 4,000 rows a run, reused by autovacuum | under 1 MB, not growing |
| **The tier** | | **about 65 to 85 MB** |

The database is 51 MB of 500 MB, and the core adds about 3.6 MB a month (43 MB a year).
With the tier it would be about 160 to 180 MB after a year and, at first-year rates,
reach 500 MB in about three and a half to four years, against about ten without it;
longer as new names thin out. The two large lines are the articles and the first-sight
histories: the new-symbol limit bounds the second, and history is kept by the brief, so
neither shrinks.

### Egress, per run

| Read | Bytes |
|---|---|
| The run's reads: `tickers` and the last few sessions of snapshots | under 0.1 MB |
| Prices, indicators, anomaly flags | none: downloaded from Yahoo and computed in the run; the replacement's guard reads `first_bar_date` from `tickers` |
| Sentiment | writes only |
| The daily refresh | unchanged: it reads nothing of the tier's |
| **The tier** | **under 0.1 MB a run, about 2 MB a month** |

Against about 33 MB a run (0.7 GB a month) for the daily refresh today: negligible.

**The dashboard's Trending tab**, cached for an hour like every loader: the table (the
active symbols' `tickers` rows, their snapshot aggregates and their 7-day news
aggregate, about 20 to 30 KB) and, per symbol a visitor opens, its bars (about 250 rows,
22 KB), from which the chart's indicators and flags are computed. Worst case, every
active symbol opened every hour: 16 x 30 KB x 24, about 12 MB a day; at the dashboard's
real traffic, a few MB a month. All of it sits well inside the free plan's 5 GB a month.

## 10. Dashboard: the Trending tab

A seventh tab, "Trending", captioned **"Discovery only: not forecast or backtested."**
(the brief's wording, with a colon in place of its dash). It shows:
- a table of the active symbols: best rank, the lists it is on, % change, volume, days
  trending, 7-day sentiment (mean compound and article count, from `trending_news`), the
  anomaly flags of the last 10 trading days, and badges: pinned, promotion-eligible, big
  move (7.3);
- for the selected row, the price and indicator charts the core tabs use
  (`price_chart`, `rsi_macd_chart`), on its bars from `raw_prices` (source `trending`),
  with indicators and flags computed by `compute_indicators` and `detect_anomalies`.

Data loading is testable, which `app.py` itself is not (Known problem 5): the queries
and the table's shaping live in an importable `dashboard/trending_view.py` that takes a
connection, tested offline with a fake one. The table is one grouped query, not one per
row. `app.py` wraps each loader in `dashboard/caching.cached` like every loader, so
nothing empty is cached. When the tables are empty or missing (before the migration),
the tab shows an empty state ("No trending data yet") instead of an error. Symbols and
names are shown as text only, never as HTML. The six core tabs are unchanged.

## 11. Configuration

All in `config.py`, read from the environment, blank-safe (a blank value takes the
default) and validated where used, not at import, so a typo cannot take the dashboard
down (as `SPLIT_GUARD_THRESHOLD` is handled today):

| Variable | Default | Meaning |
|---|---|---|
| `TRENDING_PINNED` | `SPCX` | Always tracked, outside the cap; at most 10 read |
| `TRENDING_LISTS` | `most_actives,day_gainers,day_losers` | Yahoo's predefined lists to read |
| `TRENDING_MAX_ACTIVE` | `15` | Discovered symbols tracked at once |
| `TRENDING_MAX_NEW_PER_RUN` | `5` | Symbols never seen before that may join in one run |
| `TRENDING_INACTIVE_AFTER_DAYS` | `5` | Fetched sessions absent before deactivation, cap permitting |
| `TRENDING_MIN_PRICE` | `5` | Floor, dollars; `0` turns it off |
| `TRENDING_MIN_MARKET_CAP` | off | Optional floor, dollars |
| `TRENDING_PROMOTION_MIN_DAYS` | `20` | Distinct sessions on the lists, for eligibility |
| `PROMOTION_MIN_HISTORY_DAYS` | `504` | Trading days of history, for eligibility |

`config/trending_listing_dates.txt` (section 5) is a committed file, not a variable, for
the reason `split_jump_acknowledged.txt` is: a commit carries its list everywhere.

## 12. Scheduling

### Its own workflow (decision T1)

`.github/workflows/trending.yml`:
- **Trigger**: `workflow_run` on "QuantFlow Daily Data Refresh", `types: [completed]`,
  whatever its result, plus `workflow_dispatch` (the owner's runs). It therefore starts
  only once the daily refresh has finished, never beside it, and stops if GitHub ever
  disables the daily refresh's schedule (60 days without repository activity), which the
  owner would notice there first.
- **Job**: its own `timeout-minutes: 30` and concurrency group; the same database
  secrets; the FinBERT model from the same `actions/cache` key the daily refresh saves.
- **Steps**: install from `uv.lock`; the offline test suite; the run (7.1), with its
  sentiment step last.
- **Failure**: a failed step fails this workflow, so GitHub's failure notice goes out and
  the daily refresh stays green. A red daily refresh still means a core data problem, a
  red trending run a trending one, as the dependency audit's own workflow already does
  for security updates.

In `daily_refresh.yml`, one change: its test step leaves out the tests marked `trending`
(`-m "not eval and not integration and not trending"`), so a trending test can never
stop the core's data steps on a forward-test day. CI still runs every test. Trending
tests never read `config.TICKERS`: CI sets two tickers and the daily refresh eight, and
a test that read them could pass in one and fail in the other.

### The laptop scheduler (decision T7)

The brief asks for a `run_trending_job` that nothing in the model chain depends on. The
scheduler is unused (the owner relies on the daily workflow; `docs/architecture.md`
says so). Recommended: add it as the brief says, as small as possible (one job, once a
day after the close, calling the same entry point; it logs a failure and never raises),
so the scheduler stays a complete local alternative. It reads no core prices, so the
split guard's pause does not apply to it.

### Make

`make trending` (one run of the tier; `--cap N` for a local check, Phase 6 step 5),
listed in `make help`. `make promote` waits with the rest of promotion (section 8).

## 13. Code layout

### What already exists, and is reused

| Need | Existing code | Reused as |
|---|---|---|
| Yahoo's lists | `yf.screen`, `yf.PREDEFINED_SCREENER_QUERIES` (yfinance 1.7.0) | The Yahoo source; no new HTTP client |
| Daily bars from Yahoo | `ingestion/yfinance_fetcher.download`, `as_daily_bars`, `bars_to_rows` | The price fetch (7.2) |
| Reading bars by source | `db/prices.load_daily_bars(ticker, source=...)` | The tab's chart, with `source="trending"` |
| Batched writes | `db/batch.insert_rows` | Every trending write |
| Indicators, anomaly flags | `features/indicators.compute_indicators`, `features/anomalies.detect_anomalies` | The run's summary and the tab's chart |
| Sentiment | `features/sentiment.fetch_rss`, `get_finbert`, `analyze_batch`, `save_sentiment`, `_clean_text`; `db/sentiment.insert_sentiment` | The run's sentiment step, with the two new parameters (7.5) |
| The jump check | `ingestion/split_guard.split_sized_jumps` | On each symbol's fresh bars, warn only |
| A committed list with reasons | `config.split_jump_acknowledged` | The listing-date file (section 5) |
| Charts | `dashboard/app.price_chart`, `rsi_macd_chart` | The Trending tab |
| Caching | `dashboard/caching.cached` | The tab's loaders |
| Test support | `fake_db`, the integration `pg` fixture, the byte-counting proxy (Phase 4) | Phase 6's tests and its egress measurement |

### New modules

| Module | Role |
|---|---|
| `ingestion/trending_sources.py` | `TrendingItem`, `TrendingSource`, `YahooScreenerSource` |
| `ingestion/trending_rules.py` | Pure functions: validation, filters, dedupe, session dates, the active set, deactivation, eligibility |
| `db/trending.py` | Reads and batched writes for `tickers`, `trending_snapshots`, `trending_news`; the price replacement |
| `pipelines/trending.py` | The run (7.1) and the CLI behind `make trending` |
| `dashboard/trending_view.py` | The tab's queries and shaping, testable |

Boundaries: `models/` and `evaluation/` never import a trending module, the trending
modules never import `models/`, and the only core-module changes are the two parameters
of 7.5 (a test, beside the existing boundary tests).

## 14. Failure modes

| What fails | What happens | What the owner sees |
|---|---|---|
| A Yahoo list (HTTP error, timeout, malformed) | `[]` for that list; the others still count | A warning in the trending run's log |
| Every list at once (Yahoo down, or its format changed) | Nothing deactivated or counted; prices, analytics and sentiment still run for the active set; then the run exits 1 | A failed trending run and GitHub's failure notice, every day it lasts |
| One symbol's history download, or a partial one | Its stored bars stay; it stays active | A warning |
| One symbol's indicators, flags or sentiment | That symbol is skipped for that step; the rest go on | A warning |
| An RSS feed | No articles for that symbol that run | A warning |
| A database error | The run fails | A failed trending run; the core is unaffected |
| The migration not yet applied | The run fails; the tab shows its empty state | A failed trending run until the owner runs `002_trending.sql` |
| The run hangs (an RSS fetch has no timeout of its own) | The job's 30-minute timeout ends it | A failed trending run; the daily refresh had already finished |
| An invalid symbol, pinned or listed | Dropped | A warning naming it |
| A big move on a trending name | Logged, a badge on the tab | The badge |

## 15. Tests (Phase 6)

Offline, as every unit test here, marked `trending`:
- Sources: saved responses for success, an empty list, a malformed body and an HTTP
  error; a source never raises.
- Rules: validation (good, too long, bad characters, lower case), EQUITY and core
  exclusion, the floors, dedupe (list names and best rank), session dating (a fetch
  after the close, after midnight UTC, before the next open; two fetches of one session
  count once), the active set (cap, the new-symbol limit, pinned outside both and at most
  10, absent-but-recent kept only with room, outage sessions not counted, deterministic
  ties), deactivation, eligibility (the later of `first_trade_date` and the listing
  date).
- The listing-date file through the shared parser: a required reason, bars before the
  date dropped.
- `db/trending.py` with the fake database: batched writes counted as written; the price
  replacement deleting only `source = 'trending'` rows of its symbol, and skipped for a
  download that does not reach `first_bar_date`.
- The run: an empty active set calls no stage; a run with nothing from every source
  deactivates nothing and exits 1; one failing symbol does not stop the others.
- The core unchanged: `fetch_rss` and `save_sentiment` without the new parameters behave
  exactly as today; the feed template reproduces the eight `RSS_FEEDS` URLs; nothing in
  the tier writes `news_sentiment` or a `source = 'yfinance'` row.
- The workflows: `daily_refresh.yml` unchanged but for the test step's `not trending`;
  `trending.yml` triggered by the daily refresh's completion and dispatch, with its own
  timeout; trending tests do not read `config.TICKERS`.
- The dashboard view with a fake connection, including the empty state for missing
  tables; the core tabs' loaders unchanged.
- Configuration: defaults, blank values, the three copies of `TRENDING_PINNED` equal.

On a throwaway Postgres (integration): the migration twice (idempotent), the new tables
readable by the read-only role, `trending_news` matching `news_sentiment`'s columns and
key, the batched writes and the price replacement on real types.

## 16. Order of work (Phase 6)

The brief's section 8, adjusted for the decisions above, on `feat/trending`, tests green
after every step:

1. Config, `TrendingItem`, `TrendingSource` and the Yahoo source, with saved responses
   (success, empty, malformed, HTTP error); one real response captured as the fixture.
2. Migration `002_trending.sql`, `sql/schema_trending.sql`, `make setup`,
   `db/trending.py`; mocked-database tests, then the integration tests on a throwaway
   Postgres.
3. The rules (validation, filters, dedupe, session dates, the active set, deactivation,
   eligibility) as pure functions, fully tested.
4. The two sentiment parameters (7.5), with the tests that the core path is unchanged.
5. `pipelines/trending.py` and `make trending`, verified against a throwaway Postgres
   loaded with the snapshot, with a cap of 3. The brief's check "against my DB" writes to
   the live database, so it is the owner's to run, or a one-off go-ahead.
6. `trending.yml`, the daily refresh's test-step marker, and the scheduler job, with the
   workflow tests.
7. The Trending tab, then gstack `/qa` on it.
8. gstack `/cso` on the trending path; the egress measured through the proxy.

Steps 1 to 4 touch different modules and could run in parallel worktrees; 5 needs 1 to
4; 6 and 7 need 5. Run in order, the diff stays reviewable one step at a time, which
matters more here than the hours saved.

## 17. The forward tests and the watchlist

D16, D17 and D19 were written for the eight core tickers ("at least 5 of the 8"). The
tier does not touch their cells (section 2). A change to `TICKERS` would, in two ways:
- a ticker added or promoted during the forward window adds cells the rules did not
  foresee, and the headline means over the core change;
- the forward test's snapshot export reads `TICKERS` by default
  (`evaluate export`, unless `--tickers` is given), and the harness takes D6's fetch
  times from every article in the snapshot. A snapshot holding a ninth ticker would turn
  days when only its fetch stored anything from "no fetch" into "fetched, no news" for
  the original eight, changing their features.

The owner's question at Checkpoint 5: whether to change the core watchlist at all (for
example, keep the eight and add a few diversifiers outside technology, communications and
banks). Whatever is decided, recommended: the forward tests score the original eight as
pre-registered, their snapshot is exported with `--tickers` set to those eight (written
into D16 now, before any forward date is scored), and any other core ticker is reported
beside them, not in them.

## 18. Decisions for Checkpoint 5

| | Decision | Recommended | The alternative |
|---|---|---|---|
| T1 | Where the tier runs | Its own workflow, `trending.yml`, after the daily refresh finishes; the daily test step leaves trending tests out (section 12) | Steps at the end of `daily_refresh.yml` with `continue-on-error` (the first draft): they share its 45-minute timeout, a trending failure sends no failure notice, and a trending test can stop the core |
| T2 | Trending storage | Its own: prices under source `trending` in `raw_prices`, news in `trending_news`, a summary on `tickers`; no core table written (section 2) | The core's tables with guards and filters (the first draft): a wrong `TICKERS` defeats the guards, and the core's news read in `engineering.py` must be filtered mid-window |
| T3 | Trending prices | Re-fetched from Yahoo and replaced every run (7.2), so the basis check does not apply and the jump check marks big moves (7.3) | Appended like the core's, with a warn-only basis check and hand rescales after splits |
| T4 | Limits | 15 discovered symbols plus up to 10 pinned, 5 new a run, deactivation after 5 fetched sessions absent when the cap has room, a $5 price floor; eligibility at 20 sessions and 504 trading days | Other values; no price floor (the brief's default) |
| T5 | Yahoo's `trending_tickers` | Not in Phase 6 (section 3) | A second HTTP client and a quote lookup per symbol |
| T6 | Promotion | Phase 6 shows eligibility only; `make promote` and the short-history row wait until a promotion is wanted, at the latest after the forward window (section 8) | Build both in Phase 6, as the brief lists them |
| T7 | The laptop scheduler | A minimal job, as the brief asks (section 12) | No job: the scheduler is unused |
| W | The core watchlist, and the forward tests' snapshot | The owner's choice; recommended either way: the forward tests stay on the eight, exported with `--tickers` (section 17) | |

## 19. Not in this design

- Forecasts, the ensemble, the backtest or the harness for trending names (section 1).
- 15-minute bars for trending names.
- NewsAPI, or any news source but RSS (D18).
- A social source (designed in section 3, not built).
- Automatic promotion; and, until decision T6 says otherwise, `make promote` and the
  short-history row (section 8).
