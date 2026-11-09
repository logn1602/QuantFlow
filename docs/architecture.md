# Architecture

Design decisions, the reasoning behind them, and what is wrong with them. The
[README](../README.md) covers what the system does; this covers why it is built
this way and where it would break.

---

## Package boundaries

```
config ──> db ──> ingestion
             │
             ├──> features ──> models ──> evaluation
             │                    ▲            │
             └────────────────────┴────────────┘
                     pipelines · dashboard
```

The one rule enforced by the layout: **`models` depends on `evaluation`, never
the reverse.** Scoring code must not be able to reach into training internals,
because that is the mechanism by which an evaluation quietly starts measuring
something other than generalisation. `tests/unit/test_package_boundaries.py`
parses every module in `evaluation/`, imports inside functions included, and
fails on any import of `quantflow.models`. The backtest run, which needs the
ensemble, lives in `pipelines/backtest.py` for that reason.

That constraint has a visible cost. `evaluation.out_of_fold` needs a fitting
function, and the only fitter in the project is the NNLS meta-learner in
`models.ensemble`. Importing it would create a cycle. So the loop takes
`fit_fn` as a parameter and `models.ensemble` binds it:

```python
# evaluation/out_of_fold.py — generic
def expanding_window_predictions(stacked_df, feature_cols, fit_fn, warmup): ...


# models/ensemble.py — bound to this project's meta-learner
def out_of_fold_meta_predictions(stacked_df, warmup=META_WARMUP_DAYS):
    return expanding_window_predictions(stacked_df, BASE_MODELS, _fit_nnls, warmup)
```

This looks like unnecessary indirection until you notice it is the only thing
stopping the dependency arrow from reversing.

---

## Key decisions

### The train/test split lives in one module

Previously the split was written out at six sites across three modules, each
carrying its own `HOLDOUT_DAYS = 30`. Six copies of a constant that *must*
agree is a latent bug of a specific and nasty kind: if one drifts, every model
still trains, still reports a MAPE, and the MAPEs are simply no longer
comparable. Nothing fails. The dashboard shows numbers. They are wrong.

`evaluation/splits.py` is now the only definition, and the only place a future
change to the evaluation protocol has to be made.

### NNLS instead of Ridge for the meta-learner

Ridge with an intercept can assign negative coefficients to base models and use
the intercept to compensate. That fits well in-sample and extrapolates badly:
when live prices drift outside the range the meta-learner was fitted on, a
negative weight amplifies rather than dampens.

NNLS constrains `w_i >= 0`, and normalising to `sum(w) == 1` makes the ensemble
a convex combination. It therefore cannot predict outside
`[min(base_preds), max(base_preds)]` for any given day — a hard bound on how
wrong it can be relative to its inputs. The trade is expressiveness: a convex
combination cannot learn that one base model is systematically biased and
correct for it.

### Out-of-fold scoring, and what it cost

An earlier version fitted the meta-learner on all 30 holdout days and scored it
on the same 30. That comparison cannot lose, and it reported the ensemble as
the best model on all eight tickers.

The expanding-window loop fits on days `[0, i)` and predicts day `i`. Honest,
and the honest answer is that the ensemble wins on **one** ticker out of eight.
Keeping the ensemble anyway is a deliberate choice: it helps on AAPL, and the
learned weights are a useful per-ticker diagnostic of which model family the
data favours. Presenting it as a general improvement would not be.

### Metrics rounded in one place

`regression_metrics` rounds before returning. Rounding at each call site
instead means the value written to `model_metrics`, the value logged to MLflow,
and the value printed to the console can disagree in the last digit — which is
exactly the kind of discrepancy that costs an hour when two dashboards
disagree and neither is wrong.

### Price data: one row per trading date, stored raw

`raw_prices` holds two kinds of bar: daily bars, stored at their exchange
date's 00:00 UTC, and 15-minute bars on the 09:30 to 15:45 New York grid. A
row that is neither, such as a stray 16:00 print, is left out with a warning.
Both kinds are stored only once final, because `ON CONFLICT DO NOTHING` never
corrects a stored row: today's daily bar after 17:00 New York, and a 15-minute
bar two minutes after its quarter hour ends. Yahoo serves both while they are
still trading, with the latest price as the close and the volume so far.
Everything that wants one row per trading date (every model, the indicator
and anomaly stages, the dashboard's candles) goes through
`db.prices.daily_bars`:

- a date with a daily bar uses it;
- a date with only 15-minute bars gets one bar built from them: first open,
  highest high, lowest low, last close, summed volume. Each load warns, per
  ticker, how many earlier dates took this path. Today's session, which has
  no daily bar until 17:00 New York by design, is logged at INFO instead.

The loaders used to keep each date's first row by timestamp instead. Seeded
dates were fine, since their midnight daily bar sorts first, but every date
after the seed had only 15-minute bars, so its "close" was the close of the
09:30 bar. The daily refresh now also stores Yahoo's daily bar after every
close, and `python -m quantflow.ingestion.backfill` gives every NYSE trading
day without a daily bar its official one, from each ticker's first stored date
to the last completed session: the dates with only 15-minute bars, and the
trading days with no bars at all (a gap between the seed and the first
15-minute refresh, or days the refresh did not run). Run it with `--dry-run`
first, which lists the empty days it would fill; a real run prints an undo
query that deletes exactly the rows it wrote.

**Prices are stored raw.** Both fetchers request `auto_adjust=False`: bars
are split-adjusted, as Yahoo serves everything, but not dividend-adjusted.
Yahoo never dividend-adjusts 15-minute bars, so adjusted daily bars sat on a
different basis from the 15-minute bars of the same date, and a stored value
depended on when it was fetched. Three consequences follow:

- Raw prices drop on each ex-dividend date by roughly the dividend, a
  fraction of a percent per quarter for the dividend payers here (AMZN and
  TSLA pay none). The models see that as an ordinary down move.
- Backtest returns exclude dividends, for the strategy and for buy-and-hold
  alike. The comparison between the two stays like-for-like, but neither
  figure is a total return.
- History stored before this change was seeded with `auto_adjust=True`, so
  it is dividend-adjusted as of the seed date. The bars just before the seed
  carry an adjustment factor of 1, so that segment joins the raw rows after
  it without a jump; within it, earlier prices sit slightly below what
  actually traded. A fresh seed is raw throughout, so a new install's
  numbers can differ slightly from the live database's.

**Splits are guarded, not absorbed.** Yahoo split-adjusts as of fetch time
and stored rows are never rewritten, so a split after rows were stored would
read as a crash. `quantflow.ingestion.split_guard` runs after each daily
fetch and fails the refresh when stored daily closes differ from Yahoo's raw
closes by more than `SPLIT_GUARD_THRESHOLD` (default 15%; dividend drift stays
far below it, any split far above) on three or more consecutive dates. A split
shifts every row stored before it, a long unbroken run; one or two isolated
mismatches are a bad print on one side, and are logged as warnings instead of
stopping every stage. A ticker whose check raises (the database or Yahoo
misbehaving) is logged and the others are still checked, so it never hides
another ticker's break. The command fails on a break, on an error, and on a
ticker Yahoo returned no reference prices for: a guard that could not check
a ticker does not report a pass. The fix after a split is a scoped rescale of
that ticker's earlier rows, run by hand, after which the guard passes again
by itself.

The guard also fails on a split-sized one-day move: a stored daily close at
or below 0.733 times, or at or above 1.35 times, the stored close before it.
Those limits are 3-for-2 and its reverse, the smallest common splits, moved
10% toward no move at all because the split day's own move compounds with the
split; everything beyond them is flagged whatever its ratio (7-for-1,
50-for-1, 1-for-8, a 2-for-1 on a day the stock also fell 20%), since a missed
split costs more than a false alarm. That check needs nothing from Yahoo, so
it still catches a split on a day the reference download fails, when the
scheduler would otherwise reuse the previous trading day's pass (below). It
excuses no one-day spike as a bad print: one row stored on the new basis
among rows on the old one, such as a gap filled after a split, looks the same.

A real move of that size (any fall of 26.7% or more, or rise of 35% or more)
fails the guard too. Once it is confirmed real, its date and the reason go in
`config/split_jump_acknowledged.txt`, a committed file rather than a setting,
so the daily workflow reads the list of the commit it runs on and a local
run the list in its checkout. An entry silences only this check, only for
that ticker and date, and only on a run where the comparison with Yahoo ran
and Yahoo's raw closes agree with the stored ones on that date and the date
before. A move Yahoo does not show was made by a stored row, and a move
nothing could check is not confirmed, so both still fail. The comparison runs
on every date either way, so an entry cannot hide a split. The command's
failure output lists Yahoo's split events for each failing ticker since its
first stored date; a rescale uses that ratio, not the nearest common split the
guard names, which a big move on the split day can mislead (TSLA's 5-for-1 of
2020-08-31 sits nearer 4-for-1).

In the daily workflow a failing guard fails its step, so nothing after it
runs; `make all` runs it in the same place. The per-stage commands do not run
it. The scheduler has no steps, so it remembers the guard's last answer: the
guard runs at startup, at 09:45 and 16:20 New York time, and after the 17:05
daily bars. The 09:45 run pauses the intraday price jobs on the day a split
takes effect, before the session's first 15-minute bar is final; the 16:20
run comes before the model jobs. These cron jobs run however late they
start, so a laptop that wakes after 16:20 checks before any model job still
due.

The jobs that read prices (indicators, anomalies, both model jobs, the
ensemble and the backtest) then follow the guard's last answer:

- After a run that checked every ticker and found nothing, they run, and
  that run's time is kept as the last full pass.
- After a run that found a possible unadjusted split (a basis break, or a
  split-sized move not acknowledged and confirmed by Yahoo), they log why and
  skip, until a run checks every ticker and finds none.
- While the guard cannot read its settings (the threshold, or
  `config/split_jump_acknowledged.txt`, which it reads again on every run so
  an acknowledgement needs no restart), they skip the same way: neither check
  can run until the settings are fixed.
- After a run that could not check every ticker (Yahoo returned nothing for
  one, a check raised, or the run failed), they keep running only if the last
  full pass is from the current or previous NYSE trading day, in New York
  time: a Monday-morning hiccup reuses Friday's pass. Otherwise they log why
  and skip until a run checks every ticker. A pause already in force stays
  either way.

The last full pass is kept in memory, so after a restart the price jobs wait
until a run checks every ticker, normally the one at startup. Ingestion and
sentiment keep running throughout.

**Forecast dates follow the NYSE calendar.** `utils/trading_calendar.py`
builds it from pandas' holiday rules, so it needs no calendar package, and
follows the exchange rather than the federal list: Good Friday and
Juneteenth are closed, Columbus Day and Veterans Day are open, and a
Saturday holiday closes the Friday before, except New Year's Day. Tests
check it against the exchange's published lists for 2025 to 2027. One-off
closures, such as the national day of mourning on 2025-01-09 or a weather
closure, cannot be known in advance and are not covered: a forecast made
just before one still lists that day.

### SQL in `db/`

Every read of `raw_prices`, the feature matrix's read of the stored articles,
and every write except the anomaly flags live under `quantflow.db`. The feature
matrix is built in process from the daily bars and the articles
(`features.engineering.build_features`, the function the evaluation calls at
every origin); the indicator and anomaly tables are written for the dashboard
and read by no model. The other
exceptions are display queries: the command-line `--show` and `--compare`
helpers beside each stage, and the dashboard. `raw_prices` has
one definition of a trading-day bar, `daily_bars`, and every reader is a
projection of it: daily OHLCV indexed by each date's 00:00 UTC timestamp for
the indicator engine, close and volume for anomaly detection, a `ds`/`y`
series for the forecasters, and a windowed frame with a date column for the
dashboard. Projecting one shape is cheap at this size. What must exist exactly
once is the rule for combining daily and 15-minute bars, because two copies of
it would let the models and the dashboard disagree about a day's close.
`load_bar_timestamps` serves the backfill, which only needs to know which dates
already have a daily bar.

---

## Running on free tiers

Everything runs on free plans. Each limit below was checked against the provider's own
pages on 2026-11-09.

| Service | Free-plan limits | What it means here |
|---|---|---|
| Supabase (Postgres) | 500 MB database; 5 GB egress a month, plus 5 GB cached egress; a project pauses after 1 week of inactivity; no automatic backups | The daily refresh keeps the project active on weekdays. Backups are by hand (`pg_dump` to a folder outside the repo), as before the 2026-11-07 backfill. |
| GitHub Actions | Free on standard runners in a public repository; scheduled workflows are disabled after 60 days without repository activity; a scheduled run can start late when GitHub is busy | The daily refresh stops if the repository sees no activity for 60 days, which would also stop the news forward test (docs/evaluation.md, section 4). It often starts hours after 22:30 UTC; the 12:00 UTC news cutoff (D7) allows for that. |
| Streamlit Community Cloud | An app with no traffic for 12 hours goes to sleep; a visit wakes it | A first visitor may wait for the app to wake; the README carries a screenshot so the dashboard can be seen without it. |
| NewsAPI (Developer plan) | 100 requests a day, no more available; articles appear with a 24-hour delay; searches go back one month; the plan is for development and testing only, not production | Not used from 2026-11-09: the daily refresh passes no key, and the features count RSS articles only, history included (docs/evaluation.md, section 4, D18). The articles NewsAPI supplied before stay stored and never count. |

### Database size (measured 2026-11-09)

Measured with `scripts/diagnostics/free_tier_usage.sql`, one read-only transaction
through the diagnostics runner (`run_1a_diagnostics.py --sql ... --live`, which now
logs in as the SELECT-only role of `sql/schema_roles.sql`):

| | Size | Rows (Postgres's estimate) |
|---|---|---|
| Whole database (Supabase's own schemas included) | 51 MB of 500 MB | |
| `news_sentiment` | 19 MB | 38,466 |
| `raw_prices` | 8.9 MB | 36,144 (per ticker: 648 daily bars and 3,770 15-minute bars, plus 100 Alpha Vantage rows) |
| `anomalies` | 6.1 MB | 765 |
| `technical_indicators` | 4.5 MB | 4,984 |
| `model_metrics` | 0.7 MB | 2,618 |
| everything else | under 0.3 MB each | |

The last 30 days added 5,336 price rows (1,176 of them the 2026-11-07 backfill), 4,355
articles and 680 metrics rows, about 3.6 MB a month at the current rates. At that pace
the database reaches 500 MB in roughly ten years. The anomaly and indicator tables are
replaced every run, so their files hold dead rows (6.1 MB for 765 anomaly rows);
autovacuum reuses that space, so it does not grow without bound.

### Egress (estimated 2026-11-09)

Measured on a local copy, not the live database: a throwaway Postgres loaded with the
2026-11-06 snapshot to the live row counts (4,418 price rows a ticker, 37,949
articles), the daily stages run against it through a proxy counting every byte the
server sent (wire bytes before TLS, which adds under 1%):

| Daily refresh step | Bytes sent by the database |
|---|---|
| Indicators | 3.2 MB |
| Anomalies | 3.2 MB |
| Models (every model, all 8 tickers) | 13.0 MB |
| Backtest | 9.8 MB |
| Split guard (not run locally; 8 reads of 0.40 MB each, by the measured read) | about 3.2 MB |
| **One daily run** | **about 33 MB** |

That is about 0.7 GB a month at 22 runs. Nearly all of it is price reads: every stage
reads each ticker's whole `raw_prices` history, which grows by about 570 rows a ticker
a month (a daily bar and 26 15-minute bars a trading day), so the monthly figure grows
with it, to about 1.6 GB by September 2027 (the
news forward test's end). The fetch and sentiment steps, backups (one `pg_dump`, about
the database's size) and the laptop scheduler (unused) add little. Supabase's usage
page holds the measured monthly total; the owner checks it there.

**The dashboard.** Measured before Phase 4's caching: a first render for one ticker
read 0.33 MB (329,727 bytes) and a repeat render a minute later 0.31 MB (306,574 bytes),
because the price loader, the largest read, was not cached and so every rerun read the
ticker's prices again; a public "Refresh Data" button also cleared the cache for every
visitor.

**Caching (decision E), done in Phase 4.** Every loader, prices included, now keeps
what it read for an hour (`CACHE_SECONDS` in `dashboard/caching.py`), and the Refresh
button is gone: the data changes once a day, and the page shows each refresh within
the hour. Repeat views inside the hour read nothing, so the page reads at most once an
hour for each loader, ticker and lookback setting, whatever the traffic. A visitor on
the default lookback costs about 0.33 MB an hour per ticker at most, under 8 MB a
ticker a day. The daily run's whole-history price reads are the larger line, and the
one the trending tier would multiply, so the Phase 5 design states the reads it adds.

**What a visit can find mid-run (fixed 2026-11-09).** The daily run deleted every
ticker's forecasts before fitting anything and wrote each model's as it finished, so a
visit during those minutes found some of a ticker's models or none, and the cache then
kept that for the hour. Now the run holds its forecasts until the last model is done
(`db.forecasts.ForecastSet`) and replaces each ticker's in one transaction
(`replace_forecasts`): a visitor finds the previous set or the new one. Run for two
tickers on the snapshot, a second connection polling every quarter second found no
model for either ticker at some point on the old code, and all five throughout on the
new, whose stored forecasts were identical. A ticker the run made nothing for keeps
its previous set. And no loader keeps an empty result: it is read again on the next
visit.

---

## Known problems

Ordered by how much they would change the reported results. Updated 2026-11-09
(Phase 4): items 1, 2, 6 and 8 are fixed; 8 to 13 were found in Phase 4.

### 1. The base-model comparison was not like-for-like (fixed in Phase 2)

ARIMA and Prophet were scored on a 30-day forecast made blind, XGBoost and LightGBM on
30 one-step predictions each given the true previous day, so the boosters' lower MAPE
mostly measured the easier task. The rolling-origin evaluation (docs/evaluation.md,
section 2) now scores every model on the task the dashboard publishes, a forecast 1 to
7 trading days ahead from each of 42 validation and 25 test origins, the boosters
recursively as the live run does, beside a naive "no change" baseline. The 30-day
holdout figures the daily run still records (`model_metrics`) keep the old mismatch and
are no longer shown anywhere.

### 2. The displayed MAPE did not describe the published forecast (fixed in Phase 4)

The dashboard showed the 30-day holdout MAPE (a 20-day out-of-fold figure for the
ensemble) beside a 7-day forecast. It now shows the 7-day error the evaluation measured,
per ticker, with "no change" beside every model (`dashboard/accuracy.py`, reading
`published_accuracy.csv`). What remains: those figures are as of the evaluation's
snapshot (data to 2026-11-06) and change only when the evaluation is rerun and published
again (item 10).

### 3. The backtest sample is far too small

17 trades across eight tickers over a 20-day out-of-fold window (a local run on the
2026-11-06 data, 2026-11-09). Annualised
return, Sharpe and alpha are computed and stored, and are meaningless at this sample
size: scaling a 20-day window by 252/20 blows small moves up into extreme figures (in the
data of 2026-08-04, a -0.24% return became an alpha of -1485%). The
README no longer reports backtest figures; the dashboard shows them, labelled as a
demonstration of the harness, not an edge.

### 4. The scheduler has no dependency graph

`pipelines/scheduler.py` orders stages by wall-clock offset: models at 16:45, ensemble at
17:00, backtest at 17:15. If the model job overruns by fifteen minutes, the ensemble
runs against stale base forecasts and reports a number that looks fine. There is no
completion signal, no retry, and an in-memory job store that loses all state on
restart. Production does not use it: the daily GitHub Actions workflow runs the stages
in order, each after the last has finished. Its model jobs also still delete their
models' forecasts before fitting, so a visit during a job finds those models missing;
the daily run replaces each ticker's set in one transaction instead.

### 5. The dashboard connects at import time

`dashboard/app.py` opens a database connection at module scope, so it cannot be
imported without live credentials and therefore cannot be unit tested. Its coverage is
0% and is excluded from the coverage report rather than counted as a gap. The accuracy
it shows is computed in `dashboard/accuracy.py`, which reads no database and is tested.

### 6. Duplicated work in the backtest (fixed in Phase 2)

The backtest called `tune_and_train_meta` for its log lines and threw the
results away, running the meta-learner's whole out-of-fold loop twice per
ticker. It now runs once (`tests/unit/test_backtest_pipeline.py`).

### 7. Anomaly table duplication

The live `anomalies` table held many copies of each `(ticker, ts)` flag because
`UNIQUE (ticker, ts)` was added to `schema.sql` after the table already existed, and
`CREATE TABLE IF NOT EXISTS` cannot retrofit a constraint, so the `ON CONFLICT DO
NOTHING` in `save_anomalies` had nothing to detect. The write path now replaces each
ticker's flags inside a transaction, which is correct regardless of constraint state,
so each run leaves one copy of each of its flags (about 765 rows on 2026-11-09).
`sql/migrations/001_dedupe_anomalies.sql`,
which adds the constraint, has not been run against the live database. The replace
leaves dead rows behind: 6.1 MB on disk for those 765 rows (see "Running on free
tiers"). Fresh installs are unaffected.

### 8. The dashboard did not set the dark theme it is styled for (fixed in Phase 4)

Found taking the README screenshot (2026-11-09). With no `.streamlit/config.toml`, the
dashboard followed each viewer's light or dark setting, but its charts use Plotly's dark
template and its metric cards a dark background: in the light theme the forecast
chart's white "Actual" line vanished against the white page and the cards' labels were
dark on dark. `.streamlit/config.toml` now sets `[theme] base = "dark"`, so every viewer
sees it as designed (`tests/unit/test_dashboard_settings.py`).

### 9. The live ensemble's weights are fitted on a different task

The live ensemble's NNLS weights come from a 30-day stack in which ARIMA forecasts 30
days blind, Prophet predicts the 30 dates, the boosters predict one step ahead from the
true previous day, and naive's column is one constant; live, every column is a forecast
1 to 7 days ahead (docs/evaluation.md, section 17). The rolling weights fit the right
task but did not score better on the validation dates (D2); the forward test (D17)
compares the two again.

### 10. The live run keeps no forecast history

`run_models` replaces the previous run's forecasts with the new ones, so nothing
records what was forecast for a date once it has passed, and the live 7-day error
cannot be measured from the dashboard's own forecasts. The published accuracy comes
from the evaluation instead, and goes stale until the next rerun.

### 11. XGBoost is fitted four times per ticker per run

The scored holdout fit and the live fit (models step), the ensemble's 30-day stack, and
the backtest's holdout predictions each fit XGBoost, whose tuned settings (600 trees,
depth 6, one thread) take about 7 times as long as the earlier ones. On the manual run
of 2026-11-09 the models step took 110 s and the backtest 39 s, against 30 s and 14 s on
2026-11-07. Fitting fewer times must leave every forecast unchanged.

### 12. Every stage rereads each ticker's whole price history

`load_daily_bars` reads every `raw_prices` row of a ticker, 15-minute bars included,
and keeps one row per date; the indicators, anomalies, split guard, models and backtest
each call it. That is most of the daily run's database egress (about 33 MB a run, see
"Running on free tiers"), and it grows with the stored bars, about 570 rows a ticker a
month. Reading only the window a stage needs, or the daily bars only where they exist,
would cut it.

### 13. A push may not reach the live dashboard

Streamlit Cloud installs `quantflow` from `requirements.txt`, whose `.` line makes an
ordinary (not editable) install, and `dashboard.py` runs that installed copy. By
Streamlit's documentation, the app copies committed files but needs a reboot for a fresh
build after a change to a file it does not monitor; the installed package is such a
case. So a change to the package, the dashboard's code or its published accuracy table,
reaches the live app when `requirements.txt` also changes (which rebuilds the
environment) or after a reboot from the app's workspace. Not verified on the live app
yet. Fixes for the owner to weigh: an editable install (`-e .`), or `dashboard.py`
putting the checkout's `src/` first on the import path.

---

## What would change at scale

The current design targets eight tickers, daily retraining, and a single
Postgres instance. Each of these breaks at a different point.

**Ingestion.** Writes go out as one multi-row `INSERT ... ON CONFLICT` per
chunk of up to 1,000 rows (`db/batch.py`), comfortable for eight tickers and
well beyond. At a few hundred tickers, `COPY` into a staging table with a merge
is the standard next step. It is a contained change because almost every write
goes through `db/`; the anomaly flags are the exception.

**Feature computation.** Every stage recomputes indicators over full history on
every run. At eight tickers this is seconds. The shape that survives is
incremental computation over a watermark, or materialised views maintained by
Postgres.

**Training.** Currently sequential, one ticker at a time, in the scheduler
process. Model training is embarrassingly parallel across tickers; the reason
it is not parallelised is that nothing here takes long enough to justify it.
Past a few dozen tickers, per-ticker training becomes a job queue.

**Orchestration.** The wall-clock offsets are the first thing to go. Airflow,
Dagster or Prefect give completion-signalled dependencies, retries with
backoff, and backfill — all of which the current scheduler simulates with
`sleep`-equivalents and does not actually provide.

**Model management.** MLflow is used for tracking only. At the point where a
model is *served* rather than regenerated nightly, the Model Registry becomes
the missing piece: versioned artefacts, stage transitions, and the ability to
answer "which model produced this prediction" after the fact.

**Storage.** One Postgres instance holds raw prices, indicators, sentiment,
forecasts and backtest results. The natural split is a time-series store for
`raw_prices` and indicators — where the access pattern is append-heavy and
range-scanned — leaving relational storage for the low-volume result tables.

**Evaluation.** The rolling-origin evaluation refits every model at every origin:
about 16 minutes for eight tickers and 42 origins with 3 worker processes on an 8 GB
laptop, growing with tickers times origins times models. Past a few dozen tickers it
belongs on a machine with more memory, or in a scheduled job of its own; the code
already runs tickers in parallel processes and gives the same cells at any worker
count.
