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

### SQL confined to `db/`

Every statement in the project lives under `quantflow.db`. `raw_prices` has
four readers rather than one god-function, because the four callers genuinely
need different shapes — full OHLCV indexed by timestamp, close-and-volume,
a deduplicated daily `ds`/`y` series, a windowed frame with a date column.
Merging them would mean either passing shape flags or returning the widest
shape and making callers discard columns. Four named functions sharing one
private helper is the smaller cost.

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

**Ingestion.** Row-by-row `INSERT ... ON CONFLICT` is fine for eight tickers
and fails at a few hundred. `COPY` into a staging table with a merge is the
standard fix, and it is a contained change because all writes are already in
`db/`.

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
