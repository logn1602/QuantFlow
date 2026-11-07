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
something other than generalisation.

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
close, and `python -m quantflow.ingestion.backfill` fills in the dates
between the seed and that change. Run it with `--dry-run` first; a real run
prints an undo query that deletes exactly the rows it wrote.

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

In the daily workflow a failing guard fails its step, so nothing after it
runs; `make all` runs it in the same place. The per-stage commands do not run
it. The scheduler has no steps, so it remembers the guard's last answer: the
guard runs at startup, at 09:45 and 16:20 New York time, and after the 17:05
daily bars. The 09:45 run pauses the intraday price jobs on the day a split
takes effect, before the session's first 15-minute bar is final; the 16:20
run comes before the model jobs. These cron jobs run however late they
start, so a laptop that wakes after 16:20 checks before any model job still
due. While the last answer is a break, the jobs that read prices
(indicators, anomalies, both model jobs, the ensemble and the backtest) log
why and skip. Ingestion and sentiment keep running. A ticker the guard could
not check changes nothing there: a break stays, and no pass is confirmed.

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

Every read of `raw_prices` and most writes live under `quantflow.db`. The
exceptions sit beside their callers: the anomaly-flag and backtest-result
writes, and some feature, display and evaluation queries. `raw_prices` has
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

## Known problems

Ordered by how much they would change the reported results.

### 1. The base-model comparison is not like-for-like

The most significant flaw in the project, and it is a measurement flaw, not a
code defect.

- ARIMA and Prophet: fit on train, forecast 30 steps. Day 30 is predicted 30
  days blind.
- XGBoost and LightGBM: predict on holdout rows that carry the **actual**
  lagged close and indicators for each day. Thirty independent one-step-ahead
  predictions, each given a true yesterday.

The boosters are solving a far easier problem. Their ~3% mean MAPE against
ARIMA's 5.48% mostly measures the difference in horizon, not in model quality.

Fixing it properly means walk-forward refitting: at each holdout day, refit on
everything before it, predict one step, advance. That is 30 refits per model
per ticker instead of one, which is why it has not been done — but it is the
only way the four columns become comparable. A cheaper partial fix is to score
the boosters' existing recursive rollout (`generate_forecast`) against actuals,
which at least measures the same artefact the dashboard publishes.

### 2. Reported MAPE does not describe the published forecast

The dashboard shows a 7-day forecast built by feeding each prediction back in
as the next day's lag. Errors compound across those seven steps. The MAPE shown
beside it is a one-step-ahead number. They are different quantities and the
README now says so, but a reader glancing at the dashboard will still connect
them.

### 3. The backtest sample is far too small

Eleven trades across eight tickers over 20 days. Annualised return, Sharpe and
alpha are computed and stored, and are meaningless at this sample size —
scaling a 20-day window by 252/20 turns a −0.24% move into an alpha of −1485%.
The window is 20 days rather than 30 because the out-of-fold warmup consumes
ten, so lengthening it means either a longer holdout or a shorter warmup.

### 4. The scheduler has no dependency graph

`pipelines/scheduler.py` orders stages by wall-clock offset — models at 16:45,
ensemble at 17:00, backtest at 17:15. If the model job overruns by fifteen
minutes, the ensemble runs against stale base forecasts and reports a number
that looks fine. There is no completion signal, no retry, and an in-memory job
store that loses all state on restart.

### 5. The dashboard connects at import time

`dashboard/app.py` opens a database connection at module scope, so it cannot be
imported without live credentials and therefore cannot be unit tested. Its
coverage is 0% and is excluded from the coverage report rather than counted as
a gap.

### 6. Duplicated work in the backtest

`evaluation/backtest.py` calls `tune_and_train_meta`, discards both return
values, and then calls `out_of_fold_meta_predictions` — which repeats the
entire expanding-window loop that `tune_and_train_meta` just ran internally.
The first call survives only for its logging side effect. That is the full
meta-learner fit done twice per ticker.

### 7. Anomaly table duplication

The live `anomalies` table holds many copies of each `(ticker, ts)` flag
because `UNIQUE (ticker, ts)` was added to `schema.sql` after the table already
existed, and `CREATE TABLE IF NOT EXISTS` cannot retrofit a constraint, so the
`ON CONFLICT DO NOTHING` in `save_anomalies` had nothing to detect. The write
path now replaces each ticker's flags inside a transaction, which is correct
regardless of constraint state. `sql/migrations/001_dedupe_anomalies.sql`
collapses the existing duplicates and adds the constraint; it has not been run
against the live database. Fresh installs are unaffected.

---

## What would change at scale

The current design targets eight tickers, daily retraining, and a single
Postgres instance. Each of these breaks at a different point.

**Ingestion.** Writes go out as one multi-row `INSERT ... ON CONFLICT` per
chunk of up to 1,000 rows (`db/batch.py`), comfortable for eight tickers and
well beyond. At a few hundred tickers, `COPY` into a staging table with a merge
is the standard next step. It is a contained change because almost every write
goes through `db/`; the anomaly flags and the backtest results are the
exceptions.

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

**Evaluation.** The single 30-day holdout is a point estimate. Rolling-origin
evaluation across several windows would give a distribution instead, which is
what is actually needed to say whether the ensemble's one-ticker win is signal
or noise. At the current sample size, it is not possible to tell.
