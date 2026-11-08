# Evaluation design

Status: the Phase 2 (honest measurement) design, with the decisions taken at
Checkpoint 2a on 2026-11-07 (section 15). Nothing here changes a number in the README;
Phase 4 regenerates those from the runs this design produces.

The goal is one question, answered for each model with every ticker reported: **does it
forecast the next seven closes better than "every future close equals today's close"?** Daily prices are close
to a random walk, so a small, consistent edge over that naive forecast is a good
result, and no edge is a result worth reporting too.

---

## 1. Why the current numbers cannot answer it

- **Four different tasks.** ARIMA and Prophet forecast 30 days blind from one
  origin. XGBoost and LightGBM make 30 one-step predictions, each given the true
  previous close and indicators. The ensemble is scored on 20 out-of-fold days of a
  stack built from those mixed predictions. None of these is the task the dashboard
  publishes: a 7-day forecast from today, with the boosters feeding each prediction
  back in as the next day's lag (Known problems #1 and #2 in `architecture.md`).
- **No baseline.** A 3% MAPE means little without knowing what "no change" scores on
  the same days.
- **One window.** A single 30-day holdout gives one number per model and ticker, with
  no idea of its spread.
- The published 2026-08-04 figures were also scored against 09:30-bar prices. The
  backfill fixed the data, and the snapshot checks it rather than assuming it: any date
  built from 15-min bars is listed, and a run refuses one at an origin or a target
  (section 6).

---

## 2. The task: rolling-origin 7-day forecasts (2b)

### Origins

An **origin** is a trading date `o`. At `o` a model sees only rows dated on or before
`o`, refits from scratch, and forecasts the closes of the next 7 trading dates, exactly
as the live run does after the close of `o` (the daily workflow runs at 22:30 UTC, after
the close, so `o`'s close is known).

The grid (decisions D1 and D1a), on the snapshot pinned at 2026-11-06 (section 6; 648
trading dates per ticker from 2024-04-10):

| | Origins | Dates | Notes |
|---|---|---|---|
| Test origins | 25, every 5 trading days | 2026-05-07 to 2026-10-28 | Target dates 2026-05-08 to 2026-11-06, about six months. 521 training rows at the first origin, 641 at the last. Reported, never used to choose (below). |
| Warmup origins | 12, every 5 trading days | 2026-02-10 to 2026-04-30 | Only to fit the ensemble's first weights for the test origins (section 5). Never scored. |
| Validation origins | 42, every 5 trading days | 2025-06-30 to 2026-04-23 | Every keep-or-drop decision is made here (below). The last one's day-7 target is 2026-05-04, before the first test target (2026-05-08), so no choice sees a test-date close. 306 training rows at the first. Prophet's yearly seasonality is on at only the last 2, so Prophet settings chosen here may not carry over. |
| Validation warmup | 12, every 5 trading days | 2025-04-02 to 2025-06-23 | Only to fit the ensemble's first weights for the validation origins, so the ensemble is scored on all 42 like every other model. 246 training rows at the first. Never scored. |

That is 25 forecasts per ticker, model and horizon, 200 per model and horizon across
the 8 tickers.

Every 5 trading days follows the brief's example. At horizons 6 and 7, the windows of
two consecutive origins overlap by one or two days, so their errors are correlated.
Spread is therefore reported as quartiles, not as a standard error, and the significance
test in section 3 allows for the overlap. Alternatives: every 7 trading days (18
origins, no overlap), or twelve months of origins (50, about twice the runtime).

### The rule for the test dates

Decided by the owner on 2026-11-07, and binding on Phases 2 to 4:

- **Every keep-or-drop decision uses only the 42 validation dates.** That covers
  Phase 3's experiments, whether the news features stay, and the Checkpoint 2b decision
  on switching the live ensemble to the new weights (D2, settled by the rule in section
  15 before the test dates are scored).
- **Test-date results are reported, never used to choose.** If something kept looks
  worse on the test dates, it stays kept, and the README says so.
- **The README states how many times the test dates were scored, and in which
  phases.** Every scoring is counted: the CLI scores the validation dates unless it is
  given `--window test` with a reason, tags such a run `window=test` in MLflow, and each
  one is entered in the ledger below. One scoring is one run of the harness on the test
  dates, however many models or variants it holds.
- **Phase 2 scores the test dates twice**, both after every Phase 2 choice is settled
  on the validation dates: the current system (R0) and the Phase 2 result.
- **The 2d corrections are not choices.** Counting news from when it reached the
  database, keeping missing values missing, and ignoring the bad rows ship because they
  make training match what the live run sees, whatever either window shows. Their
  effects are measured on the validation dates and reported, not used to keep or drop
  them.

Test-date ledger:

| # | Date | Phase | Commit | MLflow run | What was scored |
|---|---|---|---|---|---|
| 1 | 2026-11-08 | 2 | `7f9133c` | `9a069158` | T1: the current system (R0's commit): naive, drift, ARIMA, Prophet, XGBoost, LightGBM, the ensemble with the old weights. 25 origins x 8 tickers. |
| 2 | 2026-11-08 | 2 | `cff181e` | `ed9be5e8` | T2: the final Phase 2 commit, after D2 was settled on R7: the same models plus the ensemble with the new weights, scored beside the old (12 test-warmup origins supply the new weights' first rows). |
| 3 | 2026-11-08 | 3 | `7beb3ff` | `93bdf518` | T3: the final Phase 3 model set, after every Phase 3 choice was settled on the validation dates (section 16): every model and both ensembles. |

Kept, and worse on the test dates (for the README, rule above): the live ensemble
kept the old weights by D2 on the validation dates (3.381% against 3.389% MAPE);
on the test dates the new weights score 3.298% and the old 3.676%. The old weights
stay. Phase 3 (T3 against T2, section 16): XGBoost's tuned settings (E3) and the live
stack's naive column (E6) were kept on the validation dates, and on the test dates
XGBoost went from 3.440% to 3.538% and the live stack from 3.676% to 3.711%. Both
stay.

### How a run flows

```
eval_snapshots/2026-11-06/          one read-only export (section 6)
  prices.csv, articles.csv
        |
        v
for each ticker, for each origin o (37 of them):
        |
        +--> cut:  prices dated <= o,  articles in the database by the cutoff after o
        |
        +--> forecasters, passed in (evaluation/ never imports models/):
        |      naive     the close of o, 7 times
        |      arima     fitted on closes <= o, 7 steps
        |      prophet   fitted on history <= o, the next 7 trading dates
        |      xgboost   features built from the cut data, fitted, rolled forward 7
        |      lightgbm  the same
        |
        +--> check: 7 finite numbers, dated the next 7 dates in the data
        |
        v
cells   (ticker, model, origin, h, origin close, prediction, actual)
        |
        +--> ensemble: NNLS on earlier origins' cells (section 5), more cells
        |
        v
tables, report.md, one MLflow run (section 7)
```

### What a forecaster receives and returns

The harness cuts the data; forecasters never see anything past the cut.

- **Prices**: the snapshot's daily bars dated on or before `o`.
- **Articles**: those inserted into the database by `o`'s cutoff, 00:00 UTC on the
  next calendar day (section 4, decision D7). An article not stored by then could not
  have reached any run that day. The feature function then applies the full
  availability rule, so an article stored in time but not yet available (publish time
  plus its source's delay still ahead) lands in no row of the cut history.
- **The cutoff** lives in `utils/trading_calendar.py`, next to the calendar, because
  both `features/` and `evaluation/` need it. A test checks it against the workflow's
  cron line (`30 22 * * 1-5`): the run starts after the latest close (21:00 UTC in
  winter) and before the cutoff.
- **The contract**: a forecaster returns exactly 7 numbers for the next 7 trading
  dates. An exception or a non-finite number is a failed cell (recorded, section 7). A
  wrong length is a bug in the adapter, not a model failure, so it raises.

### What is scored

One **cell** per (ticker, model, origin, horizon h = 1..7): the origin's close, the
prediction, the actual close and the error. The forecast dates must be the next 7
dates in the data. If the trading calendar and the data disagree (a one-off closure such
as 2025-01-09, which the calendar cannot know), the run fails loudly rather than
comparing a forecast with the wrong day. No such closure falls inside the proposed
window.

### How each model forecasts at an origin

The same code as the live run, called with the history cut at `o`:

- **Naive** (new): the origin's close, for every horizon.
- **ARIMA**: `ARIMA_ORDER` (5,1,0) fitted on the closes up to `o`, 7 steps ahead.
- **Prophet**: `prophet_settings` fitted on the history up to `o`, predicted on the
  next 7 trading dates. Yearly seasonality needs 730 days of history, reached on
  2026-04-10, so it is off at the first 9 warmup origins, on at the last 3 (2026-04-16,
  04-23 and 04-30) and on at every test origin, as it was in the live run on those
  days. The ensemble's first weights therefore mix Prophet fits with and without it;
  each cell records the flag.
- **XGBoost, LightGBM**: features rebuilt from the rows up to `o` (below), the live
  model fitted on every row whose target is known by `o`, then `generate_forecast` from
  row `o`, feeding each prediction back as the next lag. `generate_forecast` updates only
  the lags, returns, rolling statistics and Bollinger features; everything else stays
  frozen at row `o`'s values, as it does live: open, high, low, volume, the day's range,
  volume change, the indicators, sentiment and anomaly features. Open, high and low
  track the price level, so every horizon's forecast is anchored near day `o`'s price;
  a booster scoring close to naive may be partly built that way rather than learned,
  which is why section 3 reports how far each model's forecasts move. This is the live
  model, not the train-split model (which was handed the holdout as `eval_set` until
  Phase 3's F3, section 16).
- **Ensemble**: section 5.

A dense 1-step-ahead task (every model given the true history each day) is not
proposed. The h = 1 cells already give a 1-step number at every origin. A daily 1-step
task would need a Prophet refit per day, about 125 x 8 x 0.8 s, roughly 13 minutes
plus the boosters; it can be added later if the h = 1 numbers raise a question.

### Features at an origin are rebuilt, not read from the tables

The stored anomaly scores look ahead. `detect_iqr_anomalies` sets its IQR fences, and
the z-score it stores on flagged days, from the whole price history
(`features/anomalies.py`, the quantiles and the return mean and standard deviation).
A stored flag for a date in May therefore reflects returns through November. The live
run recomputes anomalies from the data up to its own day, so rebuilding them at each
origin from the data up to `o` is what the live run saw that day. RSI, MACD and the
Bollinger bands are trailing or exponentially weighted, so recomputing them gives the
stored values; it costs about 40 ms per origin.

So the features at an origin come from one pure function that takes daily bars and
sentiment articles and returns the 33-column frame (plus `has_news`, section 4). The
live path will use the same function instead of reading `technical_indicators` and
`anomalies` (decision D4). The tables are still written for the dashboard. This should
change no live feature value; the snapshot carries the stored rows so that is checked
once, not assumed.

### The live run's training rows look ahead too (decision D14)

Rebuilding at each origin makes the measurement faithful to the live run, but the live
run has the same flaw inside its own training rows: on day T it fits the IQR fences,
and the return mean and standard deviation behind an IQR-flagged row's z-score, on
every return up to T, then trains the boosters on rows dated long before T. A training
row's `zscore` and `is_anomaly` therefore carry information from after that row. The
effect of any one later return is small, but the training rows are not built the way
the newest row, the one the forecast starts from, is.

Fixed on `feat/evaluation-live`, as a correction rather than a choice (section 2):
**point-in-time anomaly scores** for the features. Each row's IQR fences and return
mean and standard deviation come from the returns up to that row only, after a
warmup (rows before it get a missing score, not 0); the Z-score detector is a trailing
30-day window and already point-in-time. It ships with the in-process features (D4),
since the features stop reading the `anomalies` table then; the table keeps its
full-history flags for the dashboard, which shows them in hindsight. It is measured as
its own run (R5, section 11), and a failing-first test reverses the one `8f9fad2`
pinned: a later price must no longer move an earlier anomaly score.

The README's integrity table says the features are backward-looking, citing the
perturbation test, which covers `engineer_features` but not the anomaly scores it is
handed; Phase 4 corrects that claim (and `engineering.py`'s docstrings, with D14).

---

## 3. Metrics (2e)

All defined once in `evaluation/metrics.py`; the existing `mape`, `mae` and `rmse`
stay as they are.

- **MAPE**, kept for continuity with the README.
- **MAE**, in price units, per ticker.
- **Directional accuracy**: the share of cells where the prediction and the actual
  move the same way from the origin's close. Cells where either equals the origin's
  close are left out, so naive has none (reported as n/a). Reported next to the share
  of up moves in the window, which is what a forecast that always says "up" would
  score; 60% means little in a window where 60% of moves were up.
- **Skill** = 1 - MAE(model) / MAE(naive), over the same cells, per ticker and horizon
  and over all horizons. Positive means the model beats naive. Across tickers:
  the mean and median skill and how many of the 8 tickers are positive (MAE is not
  pooled across tickers, whose prices differ by an order of magnitude).
- **One primary test per model, fixed before any run** (decision D8). For each test
  origin, the model's absolute percentage error minus naive's, averaged over the 8
  tickers and 7 horizons: 25 numbers. Their mean, with a Newey-West variance (lag 1,
  since consecutive origins' day-6 and day-7 windows overlap) and Student t with 24
  degrees of freedom, gives one p-value per model; a Holm correction across the models
  controls for testing several. Averaging over tickers first matters: the 8 large caps
  move together, so 8 per-ticker wins in one rising window can be a single bet on the
  market. A model "beats naive" only if its Holm-adjusted p is below 0.05. Every
  per-ticker and per-horizon figure is descriptive, and the report says so. (A
  per-ticker, per-horizon test would be about 280 tests per run, with about 14 false wins
  expected at 5%.)
- **A second baseline, drift**: a random walk with drift, the origin's close grown by
  the training window's mean daily log return for h days. Reported beside naive, so a
  model that only rides the trend in a rising market shows up as such. Skill stays
  against naive, as the brief defines it.
- **Prediction bands** (decision D10): the share of actual closes inside each model's
  published band, and the band's width, per horizon: the boosters' fixed 2% band, the
  ensemble's weighted bounds, ARIMA's and Prophet's 95% intervals. No extra fits; the
  cells carry the bounds.
- **Spread**: per ticker, model and horizon, the quartiles of the absolute percentage
  error across origins.
- **How far forecasts move**: per model and horizon, the median absolute distance
  between the forecast and the origin's close. A model whose day-7 forecast barely
  leaves the origin's close is close to naive by construction (see the boosters in
  section 2).

---

## 4. Does the news help? (2d)

### Coverage

From the snapshot, per ticker and source (RSS, NewsAPI): first and last publish time,
article count, and the share of trading dates with at least one article under the
point-in-time rule below, both in the training history at the first test origin and in
the test window. Also two checks of the rule against reality: the spread of
`inserted_at - published_at` for NewsAPI rows (does the 24-hour delay show?), and how
many articles reached the database after the cutoff of the row they belong to.

### Bad rows already stored

- **FinBERT placeholders**: rows written while FinBERT could not load, `neutral` with
  scores exactly 0.33 / 0.33 / 0.34 and compound 0.0 (the old placeholder, removed in
  `29d1957`).
- **MarketWatch general headlines credited to AAPL**: RSS rows under AAPL with a
  `marketwatch.com` URL (the feed was removed in `29d1957`).

Counted per ticker with their date ranges. The feature function ignores both, live and
in the harness, so nothing depends on deleting them. A scoped DELETE comes with the
report for you to run if you want them gone.

### Point-in-time alignment

Today, `load_features` averages articles by UTC publish date and left-joins that onto
the trading dates. Four problems:

1. **Training rows see articles the live run could not.** The run at 22:30 UTC on day
   t sees nothing published later that day; day t's training row includes it.
2. **NewsAPI's free tier delays articles by about 24 hours.** On day t the live run has
   none of day t's NewsAPI articles; the training row has all of them.
3. **Weekend and holiday news is dropped.** Its date has no price row to join to.
4. **Late arrivals are back-dated.** Each fetch reaches 7 days back, so after an outage
   the next run stores days-old articles, which then count in rows the live run had
   already forecast from without them.

The rule (decision D7): an article becomes **available** at the later of its publish
time plus its source's delay (24 hours for NewsAPI, none for RSS) and its `inserted_at`,
the moment it actually reached the database. Row t gets the articles that became
available after the previous trading date's cutoff and at or before t's, with the
cutoff at **12:00 UTC on the calendar day after each date**. Weekend and holiday news
lands on the next trading day; an article fetched late counts from when it arrived, in
training exactly as live.

Why noon the next day and not the 22:30 cron time or midnight (D7 as first decided
said midnight; amended 2026-11-08): GitHub starts the scheduled run late. Of the 141
fetches in the 2026-11-06 snapshot, 63 began between 23:00 and 23:59 UTC and 73 between
00:00 and 06:45 UTC (the latest, 8 hours late). The sentiment step runs before the
models, so with a midnight cutoff a late run's articles would land in the next date's
row: about half the live forecasts would not use the news their own run fetched, and 15
of 135 trading dates (2026-04-21 to 10-30) would have no fetch in their window, against
5 at noon, all real outages. Noon holds every run seen and is still before the next
session opens (13:30 UTC at the earliest). The cost: as a training row, day t can also
hold articles another fetcher (the laptop scheduler, a manual run) stores between the
workflow's run and noon; none of the 141 fetches started between 07:00 and 12:00 UTC.

The 24-hour figure was checked against the stored `inserted_at` values on the
2026-11-06 snapshot: no NewsAPI article was stored sooner than 24.0 hours after its
publish time (median 31 hours, 95th percentile 60). NewsAPI's current documentation is
checked when D7 is wired (second branch). With `inserted_at` in the rule, the delay only
matters if an article was ever stored before it had passed, and none was.

**A historical archive is a different kind of source.** If a trusted archive is ever
added (for example a paid news API's back catalogue, imported once), its articles reach
the database months or years after they were published, so counting from arrival would
put every one of them on the day of the import. For such a source an article counts
from its **publish time plus that source's delay**: how long the provider's live feed
takes to show an article, so a training row holds what a live fetch from it could have
seen. This is a per-source rule, set where the source's delay is (`SOURCE_DELAY` in
`features/news.py`), not a switch for every source: RSS and NewsAPI keep counting from
arrival, because their `inserted_at` is when the live run could first see them.
"Trusted" means the archive's publish times have been checked against the articles the
live fetchers stored for the same days, before any of its rows count.

### Missing is not neutral (also Phase 1 review findings F6 and F7)

Today a day without news inherits the last known sentiment (forward fill), days
before any news was collected read 0, and the boosters fill every remaining missing
value with 0 (`boosting.py`, in the scored fit, the live fit and the live row). So a
feed outage freezes the sentiment features, and "no data" reads as neutral news.

Proposed (decision D6):
- A `has_news` flag (0 or 1). Article, positive and negative counts are the true counts,
  0 when the row's window had no article. No forward fill.
- `sentiment_compound`: the mean compound of the row's articles, missing when it had
  none. `sentiment_3d`: the mean compound of every article in the windows of the last
  three trading dates (t - 2 to t), missing when all three were empty.
  `sentiment_change`: today's compound minus the previous trading date's, missing when
  either is.
- Before a ticker's first collected article, every sentiment feature is missing: "not
  collected" is not "no news".
- A row whose window saw no fetch at all (no article stored for any ticker between the
  two cutoffs) gets every sentiment feature missing, counts included: nothing was
  looked for, so 0 would be a guess. `inserted_at` records when fetches stored
  something, which is enough to tell.
- The boosters get missing values as missing, not 0, for every feature, in training,
  in the live row and in the harness. XGBoost and LightGBM both handle them natively.

Limits that stay:
- A fetch that ran but stored nothing for any ticker looks like no fetch; a fetch that
  failed for one ticker only looks like a quiet day for it.
- Counts depend on how often news was fetched (NewsAPI returns at most 20 articles per
  call, and the laptop scheduler fetches every 6 hours on top of the workflow), so
  `article_count` partly measures fetch frequency.
- At serving time the newest row holds what arrived by the run; as a training row it
  can also hold what arrived between the run and midnight. A run long before the
  cutoff (the laptop scheduler during the day) sees a smaller part of the window.

### How the live run reads the news (`perf/news-egress`, 2026-11-08)

The features need only each window's counts and compound sum, and every window is a
run of whole news days (from one cutoff to the next). So the live run reads the news
summed per ticker and news day, in one query for every ticker
(`db.features.load_news_days`), once in the model run and once in the backtest. It
used to read every article once per ticker in each of three stages: 18.7 MB a run on
the 2026-11-06 snapshot, growing with every article; the sums are 89 KB a read. The
evaluation sums the snapshot's article rows the same way (`features.news.news_days`),
and a test on Postgres checks the query's sums equal those, at every edge of the rules.

The daily mean is the exact sum of the stored compounds (they are decimals), rounded
once, over their count. pandas' mean of the rows as floats, used before, differs from
it in the last bit in 370 of the snapshot's 1,056 news windows (at most 1.1e-16). No
forecast changed with it: not the live boosters' on the snapshot, nor the boosters'
or the ensemble's holdout stack at any validation origin, so R7 still stands.

### News coverage comes before any news decision (decision D15)

Every harness run reports news coverage per window: the share of (ticker, date) rows
on the validation dates and on the test dates that have at least one real article
(available by the rule above, bad rows left out), per ticker and in all, and the same
share across the training history at each window's first origin. The `coverage`
command reports it from the snapshot alone, with no model fitted, so it is known
before any news run.

If fewer than half of the (ticker, date) rows on the validation dates have a real
article, the news ablation is reported as "not enough data", and no news feature is
kept or dropped on it, in Phase 2 or Phase 3 (Phase 3's news-variant experiments do not
run). The threshold is fixed here, before the coverage is known. The 2d corrections
still ship (section 2): they make training match the live run whatever the coverage.

**Outcome on the 2026-11-06 snapshot** (digest `f0c3957b`, 2026-11-08): the stored news
starts in April 2026 (first article published 2026-04-02, first stored 2026-04-10), so
only two validation origins, 2026-04-09 and 04-23, have a real article: 16 of the 336 validation (ticker, origin) rows, 4.8%, and none of the 2,440
training rows at the first validation origin. The test origins have 200 of 200. (With
the midnight cutoff first decided: 2.4% and 80%.) So the news ablation reads
**"not enough data yet"**: R6 is reported and decides nothing, Phase 3's news-variant
experiments do not run, and the sentiment features stay in the boosters as they are,
with the 2d corrections. The test dates' 80% cannot stand in: they never choose
(section 2).

### Forward test of the news (after Phase 2)

No free archive can fill in the history: Alpha Vantage's NEWS_SENTIMENT, the candidate
checked, is a premium endpoint (one request with the free key, 2026-11-08). So the news
gets its test on dates still to come, with the plan fixed before any of them exist:

- **Window**: forward origins every 5 trading days from 2026-11-09, the first trading
  date after the snapshot, horizon 7. **42 origins** (confirmed by the owner on
  2026-11-08), as many as the validation window every other decision uses: the last
  origin is 2027-09-03 and its day-7 target 2027-09-15, 213 trading days after the
  snapshot by the project calendar. Scored once, when that close is in; no interim
  look, since an early look that misses invites another.
- **Snapshot**: a new one, exported then on the owner's go-ahead (D5). The 2026-11-06
  snapshot and its test dates stay as they are.
- **Gate**: D15 on the forward origins: at least half the (ticker, origin) rows with a
  real article, or "not enough data" again. Every test origin had one, so a working
  daily fetch clears it with room. It depends on the daily workflow fetching every
  trading day until then: GitHub disables a scheduled workflow after 60 days without
  repository activity (it did on 2026-11-03), and under D6 a day without a fetch has
  every sentiment feature missing.
- **Comparison**: the boosters with and without the sentiment features (R6's pair), on
  the code of that day, against each other with the primary test of section 3
  (per-origin APE difference, Newey-West, Holm across the two boosters), and both
  against naive.
- **Rule**: written down and approved before the forward window is scored, on Phase 3's
  bar (kept only if it improves the held-out results across tickers).
- **In the same scoring**: the live ensemble's current weights against the new ones,
  with the rule already written (section 5, "Forward test of the ensemble weights",
  decision D17).

### Ablation

The boosters with and without the sentiment features (the six plus `has_news`), on the
same snapshot and validation origins, per ticker and horizon, against each other and
against naive. Reported whichever way it goes, or as "not enough data" under the
coverage rule above.

---

## 5. The ensemble, trained on the task it is used for (2c)

Today the NNLS weights are fitted on the 30-day holdout stack, where the boosters
had true lags, and then applied to recursive 7-day forecasts.

Proposed:
- **Stack rows** are the base models' cells: one row per (origin, horizon) with the
  four base forecasts and the actual.
- **At origin `o`**, the weights are fitted on the rows of the 12 grid origins before
  `o` (`o - 5` back to `o - 60` trading days) whose target date is on or before `o`.
  Origin `o - 5`'s horizons 6 and 7 fall after `o`, so they are left out. That is about
  80 rows for 4 weights, shared across horizons; separate weights per horizon would
  leave about 12 rows each, which is a Phase 3 candidate.
- **Live, the same rule** with `o` = today, if the switch is approved (decision D2:
  settled on the validation dates by the rule in section 15, before the test dates are
  scored): the ensemble step refits the four base models at the 12 grid origins before
  today and fits the weights on those rows. About 26 seconds per ticker, so the daily
  workflow takes about 3.5 minutes longer. It ships with the second merge (decision
  D11).
- **Scored on the same 25 test origins and cells** as every other model. The warmup
  origins supply its first rows, so there is no shorter window and no separate warmup
  inside the test period.
- `evaluation/out_of_fold.py` gains this rolling-origin form of its loop and keeps
  taking the fitter as a parameter.

Which rows the weights at origin `o` may use:

```
trading days   o-60 ... o-15   o-10           o-5                 o          o+7
grid origins     *   ...   *     *              *                 *
o-10 targets                     |-- h1 .. h7 --|                       all <= o: used
o-5 targets                                     |-- h1 .. h5 --| h6 h7  h6, h7 after o: left out
o forecast                                                        |-- h1 .. h7 --|  scored
```

When the window is too thin, the ensemble fails rather than guessing:
- Rows count only when all four base models have a forecast for that origin.
- At least 8 of the 12 origins must be complete. Otherwise that ticker gets no
  ensemble forecast: live, the other tickers carry on and the run exits non-zero, as
  `ForecastAlignmentError` already does; in the harness, those cells are recorded as
  failed.
- An all-zero NNLS solution is an error, not a silent fall-back to equal weights
  (`_fit_nnls` falls back today, `models/ensemble.py`).

Not proposed now: storing weights from a periodic run (a migration and a scheduled
workflow), or fitting them on the stored live forecasts, which would cost nothing but
are deleted on every run (`run_models` clears old forecasts), so that is a retention
decision for later.

### Forward test of the ensemble weights (decision D17, pre-registered)

D2 kept the live ensemble's current weights on the validation dates (3.381% against
3.389%), and on the test dates the new weights looked better (3.298% against 3.676%);
by the rule in section 2 that changes nothing. The question gets a fresh window
instead, with the rule written on 2026-11-08, before any forward date exists:

- **Window and scoring**: the forward test of section 4 (D16): 42 origins every 5
  trading days from 2026-11-09, the last target 2027-09-15, scored once on a new
  snapshot after that close, no interim look. The ensemble comparison does not depend
  on the news gate: it is decided even if the news reads "not enough data" again.
- **The two weightings**, fixed now: the current weights (`ensemble_stack`, NNLS
  refitted at each origin on its 30-day holdout stack, as the live run does) and the
  new weights (`ensemble_rolling`, this section's rolling-origin weights: the 12 grid
  origins before each origin, at least 8 complete, only targets known by the origin).
  Both combine the same four base forecasts at each origin, from the base models live
  on the day of the scoring, so the comparison isolates the weights. The new weights'
  first rows come from the 12 grid origins before 2026-11-09, which are fitted on, not
  scored. If Phase 3 changes which weighting is live, the rule still decides between
  these two.
- **The rule**: the live ensemble uses the new weights if, over every forward cell both
  are scored on (all tickers, horizons and origins; a cell either one fails is left out
  of both), the new weights' MAPE is no higher than the current weights'; a tie goes to
  the new weights. Otherwise it keeps the current weights. This is D2's rule, repeated
  on data neither window has seen. The primary test, per-ticker and per-horizon figures
  are reported and do not decide.
- **No decision** if either weighting fails on more than 10% of its forward cells (the
  harness's failure rule): the live weights then stay as they are, and the README says
  why.
- **If the new weights win**, the live switch ships as D2 described it: the ensemble
  step refits the four base models at the 12 grid origins before each day and fits the
  weights with `out_of_fold.rolling_origin_fit` (about 3.5 minutes more per daily run),
  and D3 applies (the 30-day dashboard figure for the ensemble stops being written).

### What stays on the 30-day stack until Phase 4 (decision D3)

The backtest (Known problem #3) keeps using `collect_holdout_stacks`, with its current,
documented meaning, and the base models keep writing their 30-day holdout figures. If
the live ensemble switches to the new weights (D2), its dashboard figure from the
30-day stack stops being written, since it would describe weights the live run no
longer uses; the dashboard keeps showing the last one, dated, until Phase 4 replaces
every displayed figure with the 7-day rolling-origin error. The backtest decision goes
with Phase 4 too.

---

## 6. Data snapshot and reproducibility

- **One read of the live database per snapshot**, on your go-ahead, in a
  `REPEATABLE READ, READ ONLY` transaction checked before any query (the diagnostics
  pattern). `load_daily_bars` opens its own connection and has no as-of date, so the
  export reads the raw bars on the transaction's connection, up to the as-of date, and
  applies the same pure `daily_bars()` the loader uses; the manifest lists, per ticker,
  every date built from 15-min bars, and a run refuses one at an origin or a target. It
  also reads the sentiment articles (ticker, source, URL domain, publish and insert
  times, scores; no headline text), failing if `inserted_at` is missing from the live
  table, and the stored indicator and anomaly rows, for the one-off check in section 2.
  Well under 10 MB of egress.
- **Written to `eval_snapshots/<as-of>/`** (git-ignored) with a manifest: as-of date,
  export time, row counts and a SHA-256 per file.
- **The 2026-11-06 snapshot** (digest `f0c3957b314290dd5a1a1541ce7f73a896724b76365a5cd8896e31081f26e05f`)
  was exported on 2026-11-08 under the midnight cutoff, so it holds the articles stored
  by 2026-11-07 00:00 UTC. Under the amended D7 the as-of date's window runs to 12:00;
  that only touches the 2026-11-06 row, a target date, while every origin's window
  closes by 2026-10-29 12:00 (the last test origin is 2026-10-28). So it holds every
  article any origin can use, and it is not re-exported.
- **Kept off the public repository.** The snapshot holds Yahoo prices and per-article
  scores, publish and insert times and URL hosts (no headline text, no full URLs), so it
  is never committed or published; reports cite its digest. A copy of it, and of every
  Phase 2 run's outputs (R0 to R7, T1, T2 and the three checks), is kept outside the
  repository on the owner's laptop in `C:\QuantFlow-backups\`
  (`eval_snapshots\2026-11-06\`, and `eval_runs\<run>\` with
  `eval_runs\<run>.sha256`), checked on 2026-11-08 against the manifest's hashes and
  each run's file hashes.
- **Every run reads only the snapshot.** MLflow records the snapshot hash and the git
  commit. All Phase 2 and Phase 3 runs use the 2026-11-06 snapshot, so they compare
  like for like, and repeated runs cost the database nothing.
- **Known caveats**, the same for every model and for naive: prices before 2026-04-10
  are dividend-adjusted as of that date, later ones raw (`architecture.md`, "Price
  data"); news coverage is whatever section 4's report finds.
- **The test dates follow the rule in section 2**: reported, never used to choose,
  every scoring counted in the ledger and in the README.

---

## 7. Outputs

- **`cells.csv`**, one row per ticker, model, origin and horizon, is the single source.
  Every table is derived from it by pure, unit-tested functions.
- **Tables**: per ticker, model and horizon (MAPE, MAE, directional accuracy, skill,
  quartiles, how far forecasts move, band coverage and width); per ticker and model over
  all horizons; the headline across tickers (mean MAPE, mean and median skill, tickers
  with positive skill, the primary test's Holm-adjusted p per model), with naive and
  drift beside every model; failures.
- **MLflow** experiment `rolling_origin_evaluation`, one run per invocation. Params:
  as-of date, snapshot hash, git commit, horizon, step, origins, tickers, models, feature
  set. Metrics: the headline per model and the per-ticker figures. Artifacts: the cells,
  the tables and a Markdown report. A run that raises is logged as FAILED with its
  traceback. The store is the project's default, `mlflow.db` in the repo root (MLflow
  resolves it from the working directory, so `make evaluate` runs from there), and the
  report prints the store and the run ID.
- **An MLflow failure fails the run.** The model modules log MLflow errors as
  "non-critical" warnings (`models/statistical.py`, `_log_mlflow`); for the harness an
  unlogged experiment is a failure, so the cells and tables are still written to disk
  and the command exits non-zero.
- **Failed fits** are recorded per cell and reported. Comparisons use the cells where
  every compared model produced a forecast. A model that fails on more than 10% of its
  cells fails the run rather than being compared on a convenient subset.
- **Entry point**: `python -m quantflow.pipelines.evaluate` and `make evaluate`. It
  lives in `pipelines/` because it wires the models into the harness; `evaluation/`
  never imports `models`.
- **Windows**: `--window validation`, the default, or `--window test --reason "..."`.
  A test run without a reason is refused before any data is read. The window and the
  reason are MLflow tags, and the report's header names the window.

---

## 8. Code layout

| Module | Change |
|---|---|
| `evaluation/splits.py` | Adds the origin grid next to the 30-day split, so both schemes live in one module. |
| `evaluation/rolling_origin.py` | New: the loop (cuts the history at each origin, calls each forecaster, checks dates, builds cells, records failures) and the two baselines, naive and drift. Forecasters are passed in as functions. |
| `evaluation/out_of_fold.py` | Adds the rolling-origin stack loop (section 5); the fitter stays a parameter. |
| `evaluation/metrics.py` | Adds directional accuracy, skill, band coverage and width, and the primary test (Newey-West variance, Holm correction). |
| `evaluation/report.py` | New: the tables, from the cells. |
| `evaluation/snapshot.py` | New: writes and reads snapshot files and the manifest. |
| `evaluation/backtest.py` | Keeps the simulation and its metrics; the orchestration that imported `models` moves out. |
| `features/engineering.py` | The pure feature function; its SQL moves to `db/`. |
| `features/anomalies.py` | A pure function for the combined z-score and IQR detection, used by `run()` and the features. |
| `db/` | The article query and the snapshot's read-only reads; the backtest's queries. |
| `models/statistical.py` | One forecast function each for ARIMA and Prophet, used by the holdout score, the live forecast, the ensemble's stack and the harness (decision C). |
| `models/boosting.py` | One recursive-forecast function (fit, then roll forward), used live and by the harness. Missing values stay missing. |
| `models/ensemble.py` | Rolling-origin weights, live and in the harness; NNLS unchanged. |
| `models/forecasters.py` | New: adapts each model to the harness's forecaster signature. |
| `pipelines/evaluate.py` | New: the CLI. |
| `pipelines/backtest.py` | New home of the backtest CLI; the workflow, Makefile and scheduler point at it. |

---

## 9. Cleanup in this phase

- **Decision C**: the ARIMA and Prophet holdout code duplicated in `statistical.py`
  and `ensemble.py` merges into the forecast functions above; the ensemble's hardcoded
  (5,1,0) becomes `ARIMA_ORDER`. `collect_holdout_stacks` takes the price and feature
  inputs instead of loading them itself (today it calls `load_prices` and
  `load_features` inside), so R0 can replay the old ensemble at each origin; the live
  wrapper does the loading.
- **Docstrings and diagrams this change makes wrong**, updated in the same commits:
  the architecture block in `models/ensemble.py`'s module docstring (its stack is the
  30-day holdout), `evaluation/out_of_fold.py`'s docstring ("20 days rather than 30"),
  the note on the holdout metric in `models/boosting.py`, and the Package boundaries
  diagram in `architecture.md`.
- **`evaluation` imports `models`** in `backtest.py`: removed, plus a test that parses
  every module in `evaluation/` and fails on any import of `quantflow.models`, lazy
  ones included. Known problem #6 (the backtest runs the meta-learner's out-of-fold loop
  twice) is fixed on the way.
- **Inline holdout slices** in `ensemble.py` and the second `HOLDOUT_DAYS` in
  `backtest.py` go through `evaluation/splits.py`.
- **2d**: the bad sentiment rows, the frozen forward fill (F6) and the boosters'
  missing-as-0 (F7), as in section 4.
- **SQL into `db/`** where this phase touches it: the feature queries and the
  backtest's.

---

## 10. Tests

Unit tests stay offline (no database, network or training); fakes stand in for the
models.

**Failing first.** Each fix to existing behaviour gets a regression test that is shown
failing on the code before it, as in Phase 1:
- F7: a missing feature reaches the booster as missing, not 0 (live row and training).
- F6: after a ticker's last article, rows read `has_news` 0 and a missing compound, not
  the last value.
- Point-in-time: a Saturday article lands on Monday instead of vanishing; a NewsAPI
  article lands a day later; an article stored days after it was published counts from
  when it was stored, not on its publish date; an article stored by a run that GitHub
  started late (06:50 UTC) counts that day, one stored at 12:05 UTC the next.
- Bad rows: a placeholder row and a MarketWatch row under AAPL change no feature.
- Decision C: changing `ARIMA_ORDER` changes the order the ensemble's stack fits.
- Known problem #6: the backtest runs the out-of-fold loop once.
- Live ensemble: a thin window or an all-zero NNLS solution fails loudly.

**New code**, each branch covered:
- The grid: spacing, count, every test origin's horizon-7 target inside the data,
  warmup origins before test origins, too little data refused.
- The loop: a spy forecaster never receives a price after its origin or an article
  stored after the cutoff; target dates are the next 7 data dates; a calendar and data
  mismatch fails the run; an exception and a non-finite number each become a failed
  cell; a wrong length raises; the 10% rule fails the run.
- Baselines: naive's predictions equal the origin's close and its skill against itself
  is 0; drift's equal the close grown by the training mean log return, using no return
  after the origin; an oracle forecaster scores skill 1.
- Metrics, against hand-computed values: directional accuracy (ties left out, all
  cells left out gives none), skill (naive error of 0 gives none), band coverage and
  width, the primary test (the per-origin average over tickers and horizons, the
  Newey-West lag, Holm's ordering and adjusted p-values, identical errors, fewer than 3
  origins gives none).
- Missing news: a window with no fetch for any ticker gives missing counts, not 0.
- The ensemble's stack loop: the fitter never sees a row whose target is after the
  origin; the 12-origin window and the 8-origin minimum hold; the live weights and the
  harness weights come from the same function for the same cells.
- The report: tables from a hand-built cells frame, the intersection rule (a model's
  failed cell removes that cell from every comparison), the headline arithmetic.
- The snapshot: a round trip; a file whose hash does not match the manifest is refused;
  the export refuses a non-local database without `--live` before connecting, and checks
  `transaction_read_only` (the diagnostics tests, reused); plus an integration test on a
  throwaway local Postgres.
- The CLI: params and metrics reach a faked MLflow; an exception ends the run as FAILED;
  an MLflow failure exits non-zero with the files written; the window defaults to
  validation; `--window test` without a reason is refused before any data is read; the
  window and reason are tagged.
- The grid's validation origins: every day-7 target falls before the first test
  target.
- Features: given a history, the pure feature function gives what the live run reads
  for it (an identity test against the stored-table path); a later price never moves
  an earlier indicator, but it can move an earlier anomaly score, whose IQR fences use
  the whole history the function is given, which is why the harness cuts the history
  at each origin before building features (both pinned by tests); "not collected"
  versus "no news"; the three sentiment definitions.
- The live path: the boosters' live forecast is built from the same feature function
  and recursive-forecast function the harness calls; `load_features` no longer reads the
  indicator and anomaly tables.
- The backtest move: the workflow, Makefile and scheduler call the new module
  (`test_workflows.py`, `test_scheduler.py`); the import guard; the cutoff against the
  workflow's cron line (the run starts after the latest close and before the cutoff).
- One eval-marked test (`tests/evals`) runs the whole harness on a synthetic random walk
  with fast fake models.

---

## 11. Order of work and the runs reported at Checkpoint 2b

Each run is tied to a commit, so a "before" number is produced by the code as it was,
not by a legacy flag kept around for the purpose. Two branches (decision D11), so the
measurement can merge without changing anything the daily workflow does:

**`feat/evaluation`: measurement, no change to the live run**
1. Refactors, each with tests showing identical behaviour: decision C, the forecast
   functions, the pure feature and anomaly functions (the harness uses them; the live
   path keeps reading the tables until D4 ships), the backtest's new entry point and the
   import guard, the splits.
2. The harness, baselines, metrics, report, snapshot, CLI and MLflow logging.
3. Verify on 1 ticker, then 3, on the validation dates of the snapshot (taken on your
   go-ahead, D5); then the coverage report.
4. **R0 on the validation dates: the current code under the harness's cut.** All
   models, the old ensemble weights (refitted at each origin on its own 30-day stack),
   naive and drift. Its news features still use publish dates, so it slightly flatters
   today's boosters.

**`feat/evaluation-live`: the live changes, branched from `feat/evaluation`**
5. One change per commit and one booster run per change on the validation dates
   (decision D12): **R1** bad sentiment rows ignored; **R2** news availability (D7);
   **R3** no forward fill, `has_news`, and the missing-news rules (D6); **R4** missing
   as missing for every feature (F7); **R5** point-in-time anomaly scores (D14); **R6**
   the ablation, R5 without the sentiment features, or "not enough data" (D15). These
   measure the corrections; they do not decide them (section 2).
6. The rolling-origin ensemble weights and the in-process live features (D4). **R7 on
   the validation dates**: every model on the final commit, the new weights beside the
   old ones on the same cells. D2 is settled here, by its rule.
7. **The two test-date scorings**, entered in the ledger: **T1**, R0's commit (the
   current system); **T2**, the final commit, with the new and old ensemble weights both
   reported. Nothing is chosen after this point in Phase 2.

As built (2026-11-08): the in-process live features (D4) went first on the second
branch, as their own commit. They change no feature value (on the 2026-11-06 snapshot
the rebuilt indicators and anomaly scores equal the stored ones exactly), and with them
in place each of R1 to R5 reaches the live run and the harness in the same commit,
instead of being built twice. Each run executes from a temporary git worktree at its
commit, so it uses exactly the committed code. The new weights are scored as
`ensemble_rolling`, the old as `ensemble_stack`; R6's pair is `xgboost_no_news` and
`lightgbm_no_news`, run beside the full boosters. Combiners are scored at the scored
origins only; the warmup origins supply their earlier cells.

Checkpoint 2b shows the validation runs (R0 to R7, with the D2 outcome) and the test
scorings (T1, T2): every ticker, every model, naive and drift included. On approval,
`feat/evaluation` is pushed first; `feat/evaluation-live` follows once that is merged,
with the live ensemble switch as D2 settled.

### Runtime

Measured on this laptop on a synthetic 648-day series (one origin, one ticker): features
0.04 s, ARIMA 0.11 s, Prophet 0.81 s, XGBoost about 1 s (its first call is the slowest),
LightGBM 0.17 s; about 2.2 s for everything. The old ensemble's 30-day stack adds about
2 s per origin where it is replayed.

| Run | Window | Work | Estimate |
|---|---|---|---|
| 1 and 3 ticker checks | validation | 54 origins | about 2 and 6 minutes |
| R0 | validation | 54 origins x 8 tickers, plus the old stack at 42 origins | about 27 minutes |
| R1 to R6 | validation | boosters only, 42 origins x 8 tickers, each | about 4 minutes each |
| R7 | validation | 54 origins x 8 tickers, plus the old stack at 42 origins | about 27 minutes |
| T1, T2 | test | 37 origins x 8 tickers, plus the old stack at 25 origins, each | about 18 minutes each |

About two hours of runs in all, none longer than half an hour. Memory stays well under
1 GB. The real runs replace these estimates in the checkpoint. The daily workflow gains
about 3.5 minutes from the live ensemble refits, if D2 approves them (section 5).

---

## 12. Not in this phase

| Item | Why not now |
|---|---|
| Running tickers in parallel processes | A full run is minutes; worth it in Phase 3, when the harness runs many times. |
| A dense daily 1-step task | The h = 1 cells cover 1-step; see section 2. |
| Separate ensemble weights per horizon | About 12 rows each; a Phase 3 candidate. |
| The backtest on the new harness | Known problem #3; with the dashboard figures in Phase 4. |
| The dashboard's accuracy figures | Phase 4, by the brief. |
| Weights from stored live forecasts | Needs a retention decision (`run_models` clears them). |
| A manual GitHub workflow for the evaluation | Decision D9. |
| Deleting the bad sentiment rows | The features ignore them; the DELETE is offered, yours to run. |
| Social-media news sources | Phase 5 designs for them; not built. |
| Deciding whether the news helps | Too little news on the validation dates (D15); a forward test after 2027-09-15 (section 4). |

---

## 13. What already exists and is reused

| Existing | Reused for |
|---|---|
| `evaluation/out_of_fold.py`, fitter as a parameter | The rolling-origin stack loop, same pattern. |
| `evaluation/splits.py`, `evaluation/metrics.py` | The grid, and the new metrics next to the old ones. |
| `boosting.generate_forecast`, `fit_live_model` | The boosters' recursive forecast at every origin. |
| `statistical.prophet_settings`, `ARIMA_ORDER` | Every Prophet and ARIMA fit. |
| `indicators.compute_indicators`, `anomalies.detect_*` | The per-origin feature rebuild. |
| `trading_calendar.next_trading_days` | Forecast dates, checked against the data. |
| `db/prices.load_daily_bars` | The snapshot's prices: the one row per date every model reads. |
| `scripts/diagnostics/run_1a_diagnostics.py` | The read-only transaction and local-only guard for the export. |
| `ensemble.check_forecast_dates`, `ForecastAlignmentError` | The live ensemble's fail-loudly policy. |

---

## 14. What this resolves and what it leaves

Resolves Known problems #1 (not like-for-like) and #2 (the reported error does not
describe the published forecast), the single-holdout limitation, and #6 (the duplicated
meta-learner loop). Leaves #3 (the backtest, Phase 4), #4 (the scheduler), #5 (the
dashboard import) and #7 (the anomaly duplicates; migration 001 is still yours to run).

---

## 15. Decisions (Checkpoint 2a, 2026-11-07)

Decided by the owner; the body above applies them.

| | Decision |
|---|---|
| D1 | 25 test origins every 5 trading days (2026-05-07 to 2026-10-28), 12 warmup origins, h = 1..7, snapshot pinned at 2026-11-06. |
| D1a | 42 validation origins (2025-06-30 to 2026-04-23, every day-7 target before the first test target) for every keep-or-drop decision, with 12 validation-warmup origins (2025-04-02 to 2025-06-23) for the ensemble. As first decided the window ended 2026-04-30, whose targets reach 2026-05-08 and 05-11, two test dates; moved back one origin on 2026-11-07; approved 2026-11-08. |
| D2 | The rolling-origin weights are built now. The live run switches to them if, on the 42 validation dates, their mean absolute percentage error over every cell (all tickers and horizons) is no higher than the old weights'; a tie goes to the new weights, which are fitted on the task they are used for. Settled on the validation run (R7) before the test dates are scored; Checkpoint 2b shows the comparison. Approved as a written rule on 2026-11-08, rather than decided after seeing results. **Settled 2026-11-08 on R7** (commit `5799ce9`, MLflow run `f64ca084`), before any test date was scored: over the same 2,352 validation cells the new weights score 3.3885% and the old 3.3807%, so the live ensemble keeps the old weights (and D3's 30-day figure keeps being written). |
| D3 | The backtest stays on the 30-day stack until Phase 4. If the live ensemble switches, its 30-day dashboard figure stops being written. |
| D4 | Live features built in-process through the harness's function (second branch). |
| D5 | One read-only snapshot export per snapshot, on the owner's go-ahead, into git-ignored `eval_snapshots/`; the coverage report runs on it first. |
| D6 | Missing stays missing for the boosters; no forward fill; `has_news`; sentiment missing before collection began and on rows whose window saw no fetch. |
| D7 | An article is available at the later of publish time plus its source's delay and `inserted_at`; rows cut at 12:00 UTC on the calendar day after each trading date (first decided as 00:00 UTC; amended 2026-11-08 because GitHub starts the 22:30 run up to 8 hours late, section 4). A trusted historical archive, if one is ever added, counts from publish time plus its delay, not from arrival: a per-source rule (added 2026-11-08, section 4). |
| D8 | One primary test per model (per-origin difference from naive averaged over tickers and horizons, Newey-West, Holm across models); everything else descriptive; drift baseline beside naive. |
| D9 | No manual GitHub workflow for the evaluation now. |
| D10 | Prediction-band coverage and width reported per horizon. |
| D11 | Two branches: `feat/evaluation` (measurement, no live change) merges first, then `feat/evaluation-live`. |
| D12 | One change per commit and per run on the second branch, on the validation dates. |
| D13 | The rule for the test dates (section 2): every keep-or-drop decision on the validation dates only, D2 included; test results reported, never used to choose, and anything kept stays kept if it looks worse there, as the README says; every test scoring counted (ledger, MLflow tag) and the count and phases stated in the README. Phase 2 scores the test dates twice (T1, T2). |
| D14 | Point-in-time anomaly scores for the features, on `feat/evaluation-live` with D4: the live training rows stop looking ahead (section 2). A correction, measured as R5. The README's "features are backward-looking" claim is corrected in Phase 4. |
| D15 | News coverage per window is reported before any news run; under half of the validation (ticker, date) rows with a real article means the ablation is "not enough data" and no news feature is kept or dropped on it (section 4). |
| D16 | The news gets a forward test: 42 origins every 5 trading days from 2026-11-09, scored once after the last target (2027-09-15) on a new snapshot, with the D15 gate and a keep-or-drop rule written before scoring (section 4). Confirmed 2026-11-08. |
| D17 | Pre-registered 2026-11-08, before any forward date is scored: the forward test (D16) also compares the live ensemble's current weights with the new rolling-origin weights on the same cells. The live ensemble uses the new weights if their MAPE over every forward cell is no higher than the current weights' (a tie goes to the new weights, as in D2); either failing on more than 10% of its cells means no decision (section 5). |

Phase 3 candidates raised in review, not part of Phase 2: naive's forecast (the
origin's close) as a fifth stack column, so the ensemble can lean on it; not freezing
open, high and low during the boosters' rollout.

---

## 16. Phase 3: accuracy experiments

Approved by the owner on 2026-11-08, with changes: the correctness fixes are separated
from the experiments and kept whatever they score; tickers run in parallel, at most 3
workers; Prophet tuning (E5) is skipped. Written before any Phase 3 run. Everything
runs on `feat/accuracy`, one change per commit, each run from a git worktree at its
commit on snapshot `f0c3957b`, on the validation dates, logged to MLflow.

### Order of work

1. **Setup, no forecast changes.**
   - **S1**: the interval score (below) beside band coverage and width, per horizon;
     the keep-or-drop rule as code (`evaluation/decision.py`, the `compare` command),
     so every decision is computed the same way, from two runs' cells.
   - **S2**: tickers in parallel worker processes, at most 3 (8 GB of RAM), each model
     on one thread inside its worker. A test checks that parallel cells equal
     sequential cells exactly, and a full parallel run is checked against the
     sequential run of the commit before it.
2. **Fixes, kept whatever they score**, like the 2d corrections (section 2): each its
   own commit with a test shown failing first, measured on the validation dates and
   reported, not used to keep or drop anything.
   - **F1, one thread per booster.** XGBoost's row subsampling (`subsample` 0.8) draws
     different rows for different thread counts, so the same data gives a different
     forecast on this laptop (8 threads, where R7 ran) than on GitHub's runner: up to
     about $3 apart at one origin of the snapshot. With one thread the published
     forecast is the evaluated one on any machine. LightGBM's forecasts do not depend
     on the thread count; it is pinned too, so each worker runs one thread per model.
   - **F1b, Prophet's band from a fixed seed** (found building S2). Prophet simulates
     its 95% interval from numpy's global random state, unseeded, so its band, and the
     ensembles' bands that combine it, changed on every run, and worker processes could
     not reproduce a sequential run. The band is drawn from a fixed seed, and the
     caller's random state is put back. No point forecast changes.
   - **F2, the rollout's Bollinger features as training builds them.** Training gets
     the bands from `ta`, with the population standard deviation, rounded as stored;
     the rollout used the sample standard deviation (its `bb_width` about 2.6% wider)
     and clipped `bb_position` to [0, 1], which training never does. Both now come
     from one definition, and a feature is missing where the band has no width, as in
     training.
   - **F3, no holdout in the scored fit.** The scored XGBoost fit received the holdout
     as `eval_set`. Without early stopping that changes no prediction, but it would
     leak the holdout the moment early stopping or tuning read it.

   The run after F3, **B0**, is every model's incumbent.
3. **Experiments**, in this order, each one hypothesis against the incumbent and naive:

| # | Hypothesis | Decided on |
|---|---|---|
| E1 | The boosters predict the next day's log return, turned back into a price, instead of the price itself: a tree cannot predict above the highest target it was trained on. | each booster |
| E2 | The rollout stops freezing open, high and low: each step's open is the previous close, and high and low keep the last day's ratios to the close. | each booster |
| E3 | Booster hyperparameters tuned on training data only (below). | each booster |
| E4 | ARIMA's order chosen by AIC on each fit's own data, p 0 to 5, q 0 to 2, d 1, instead of a fixed (5, 1, 0). | ARIMA |
| | Base set: every model and both ensembles, after E1 to E4: the ensembles' incumbent. | |
| E6 | Naive as a fifth stack column, in the live 30-day stack and the rolling weights. | each ensemble |
| E7 | Separate rolling-ensemble weights per horizon. | the rolling ensemble |
| I1 | The boosters' bands (below). | the band rule |

   Not run: Prophet tuning (E5, skipped by the owner), and the news variants (D15).
4. **T3**: the test dates scored once, with the final model set (ledger entry 3),
   reported, never used to choose. Phase 4 reuses T3 if the code has not changed.

### The keep-or-drop rule (E1 to E7)

- **Cells**: one model's cells at the 42 validation origins x 8 tickers x horizons 1
  to 7, in the candidate's run and the incumbent's, both on snapshot `f0c3957b`.
  Compared on the cells both versions forecast.
- **Incumbent**: the model as of the last kept change (B0 to start).
- **Keep only if both**: (1) the candidate's MAPE over all those cells is strictly lower
  than the incumbent's; (2) its MAPE is lower on at least 5 of the 8 tickers.
  Otherwise drop it; a tie drops (the incumbent is simpler or already there).
- **Dropped regardless** if it fails more cells than the incumbent, or more than the
  harness's 10% limit.
- A change to both boosters is decided per booster; ARIMA's on ARIMA's cells; E6 per
  ensemble, E7 on the rolling ensemble. Other models' numbers are reported, not
  deciding.
- **Reported, not deciding**: MAPE by horizon, MAE, skill against naive, directional
  accuracy, interval score and band coverage, a paired test against the incumbent
  (per-origin difference in absolute percentage error, Newey-West lag 1), and the D8
  primary test against naive, still the only basis for "beats naive".

### Tuning on training data only (E3)

Tuning origins sit on the grid's 5-day step before the validation warmup, each one's
day-7 target before the first warmup origin (2025-04-02): 24 origins, 2024-10-01 to
2025-03-19 on the snapshot (the last target 2025-03-28), each training on 121 to 236
rows. One is left out: 2025-01-03, whose forecast days include 2025-01-09, a one-off
closure the trading calendar does not know (section 2), so the harness refuses it. The
first tuning run stopped on it before scoring anything; the command now leaves such an
origin out and names it. 23 origins are scored. No tuning target is a
validation or test date. For each booster the candidates are the current settings and a
grid of 12 (max depth 3, 4 or 6; learning rate 0.03 or 0.1; 200 or 600 trees; the
sampling settings unchanged), each scored by its 7-day recursive MAPE over every
tuning cell; the lowest is E3's candidate, then judged on the validation dates by the
rule above. If the current settings score lowest, E3 has no candidate and does not
run. Settings chosen on 121 to 236 training rows may suit the validation origins' 306
to 511 less well; that is the cost of tuning on training data only. No early stopping.

### The band rule (I1)

The boosters publish a fixed plus or minus 2% band, which covered 44% (XGBoost) and 42%
(LightGBM) of actual closes on the validation dates (R7). Two candidates, after E1 to E3 so they describe
the final boosters:

- (a) a random-walk band: forecast x exp(plus or minus 1.96 sigma sqrt(h)), with sigma
  the standard deviation of the last 60 daily log returns up to the origin;
- (b) per-horizon 2.5% and 97.5% quantiles of the booster's own log errors at the 12
  grid origins before this one whose targets are known by it (the rolling weights'
  pattern; the live run would replay those 12 forecasts, about 1 to 2 minutes).

A candidate is kept only if its 95% band covers 92.5% to 97.5% of each booster's
validation cells and its mean interval score is below the plus or minus 2% band's; of
two that pass, the lower interval score. If neither passes, the boosters' bands stop
being presented as 95% bands, and the ensemble stops combining them as if they were.
The dashboard labels the bands "XGBOOST band" and so on, with no level, and the README
claims none; Phase 4 adds the measured coverage to the label.

The **interval score** of a 95% band [l, u] for an actual y is (u - l) + 40 (l - y) if
y < l, + 40 (y - u) if y > u (Gneiting and Raftery, 2007, alpha 0.05), reported as a
percentage of y so tickers can be pooled. Narrow bands that miss are penalised as
well as wide ones; lower is better.

### Results on the validation dates (2026-11-08)

Every run used snapshot `f0c3957b` and the 42 validation origins (54 with the warmup,
where an ensemble or a band variant needed earlier origins). Each ran from a git
worktree at its commit and was logged to MLflow (experiment `rolling_origin_evaluation`);
the outputs are in `eval_runs/<run>/`. Each decision's per-ticker and per-horizon tables
are in the candidate run's `compare-<incumbent>.md`. MAPE in %.

**Fixes** (kept whatever they scored):

| Fix | Run, commit, MLflow | Effect |
|---|---|---|
| F1, one thread | F1, `e970153`, `75b0de04` | XGBoost 3.2151 to 3.2177: every forecast moved, by up to $18.27. LightGBM, ARIMA, naive and drift identical; the ensembles move with XGBoost (stack 3.3807 to 3.3747, rolling 3.3885 to 3.3782). Prophet's band also changed on every cell between R7 and F1, with no change to Prophet: its unseeded sampling (F1b). |
| F1b, Prophet's seed | F1b, `ecc289d`, `189faf38` | No point forecast changed; only Prophet's and the ensembles' bands. |
| S2, workers | S2, `310f1e2`, `b86066ca` | All 22,848 cells equal F1b's exactly, every column but timing; 338 s against 1,069 s. |
| F2, Bollinger | F2, `4200240`, `31845c57` | XGBoost 3.2177 to 3.2138, LightGBM 3.3484 to 3.3465 (days 2 to 7 only: day 1 starts from the true row). |
| F3, no `eval_set` | B0, `6ed36e5`, `72a7633c` | Identical to F2 everywhere. B0 is every model's incumbent. |

**Experiments** (the rule above; "lower" counts the tickers on which the candidate's
MAPE was lower):

| | Run, commit, MLflow | Model | Incumbent | Candidate | Lower | Decision |
|---|---|---|---|---|---|---|
| E1, log-return target | E1, `f096371`, `4fa403e6` | XGBoost | 3.2138 | 3.6804 | 1 of 8 | drop |
| | | LightGBM | 3.3465 | 3.6416 | 2 of 8 | drop |
| E2, open, high, low rolled | E2, `a80d5fa`, `670028c5` | XGBoost | 3.2138 | 3.5849 | 0 of 8 | drop |
| | | LightGBM | 3.3465 | 3.6468 | 1 of 8 | drop |
| E3, tuned settings | E3, `94ea0ec`, `9f0f8d8c` | XGBoost | 3.2138 | 3.1426 | 6 of 8 | **keep** |
| | | LightGBM | 3.3465 | 3.3834 | 3 of 8 | drop |
| E4, ARIMA by AIC | E4, `6549dff`, `599cdc76` | ARIMA | 2.8833 | 2.8827 | 3 of 8 | drop |
| E6, naive column | E6, `73a47b5`, `fbff7c63` | ensemble_stack | 3.3740 | 3.3709 | 7 of 8 | **keep** |
| | | ensemble_rolling | 3.3913 | 3.3823 | 6 of 8 | **keep** |
| E7, per-horizon weights | E7, `01da7e8`, `a230ad3f` | ensemble_rolling | 3.3823 | 3.3692 | 4 of 8 | drop |

- **E1** did better at days 1 and 2 (XGBoost 1.56% against 1.76% at day 1) and worse
  from day 3 (5.52% against 4.30% at day 7): the predicted returns compound over the
  rollout.
- **E2** left day 1 unchanged, by construction, and did worse every later day. Frozen,
  open, high and low pull each step toward the starting price, which on these dates
  helps.
- **E3**'s tuning (run E3-tuning, `4836f40`, MLflow `85c14595`, 23 origins, 1,288
  cells): XGBoost depth 6, rate 0.03, 600 trees, at 3.918% against the current
  settings' 4.013%; LightGBM depth 4, rate 0.1, 600 trees, at 4.322% against 4.346%. On
  the validation dates XGBoost's was lower at every horizon (paired p 0.002); LightGBM's
  did worse. A check run (`0ee3abc`) and the base set below reproduce each decision
  commit's forecasts exactly.
- **E4**: lower overall by 0.0006 points (paired p 0.95), on 3 tickers. AIC chose
  (0, 1, 0), the naive forecast, in 139 of the 336 fits.
- **Base set** (`8da5c62`, MLflow `163eac9e`, every model and both ensembles after E1
  to E4): the ensembles' incumbent. The better XGBoost made both ensembles slightly
  worse (stack 3.3721 to 3.3740, rolling 3.3789 to 3.3913).
- **E6**: small gains (stack -0.003 points, paired p 0.21; rolling -0.009, p 0.053).
  Naive has no band, so its bounds are its forecast, and the ensembles' bands narrow:
  over the validation cells the rolling ensemble's coverage fell from 81.3% to 61.9%
  and the stack's from 59.7% to 56.6%. Bands are not part of the rule; see the open
  points below.
- **E7** did much better at day 1 (1.64% against 2.16%) and worse from day 4 (4.83%
  against 4.46% at day 7).

**Bands (I1; run I1, `16416cd`, MLflow `c80674c8`)**, over each booster's 2,352
validation cells:

| Booster | Band | Coverage | Width | Interval score |
|---|---|---|---|---|
| XGBoost | plus or minus 2% (current) | 45.7% | 4.0% | 69.2 |
| | random walk | 91.4% | 14.4% | 24.5 |
| | error quantiles | 80.8% | 12.1% | 32.8 |
| LightGBM | plus or minus 2% (current) | 41.4% | 4.0% | 75.6 |
| | random walk | 89.5% | 14.4% | 26.6 |
| | error quantiles | 80.1% | 12.1% | 35.2 |

Neither candidate covers 92.5% to 97.5%, so by the band rule neither is kept, and the
boosters keep the plus or minus 2% band. The random-walk band comes close (91.4% and
89.5%, under-covering most at days 1 to 3) at about a third of the current band's
interval score. Quantiles of only 11 or 12 earlier errors per horizon cannot reach 95%
coverage (the range of 12 draws holds a 13th about 85% of the time).

**The final model set** is the code at `16416cd`: XGBoost with E3's settings; LightGBM,
ARIMA (5, 1, 0) and Prophet as they were; both ensembles weighing naive as a fifth
column; the boosters' plus or minus 2% band. Its validation results are run E6's: run I1
reproduces E6's naive and booster forecasts and bands exactly at `16416cd`, and nothing
after E6 changes the ensembles (E7 was turned back off).

| Model | R7 MAPE | Final MAPE | Final mean skill | Tickers with positive skill | Holm p against naive |
|---|---|---|---|---|---|
| naive | 2.8798 | 2.8798 | 0 | | |
| ARIMA | 2.8833 | 2.8833 | -0.003 | 4 | 0.79 |
| drift | 2.9280 | 2.9280 | -0.011 | 2 | 0.42 |
| XGBoost | 3.2151 | 3.1426 | -0.096 | 1 | 0.0006 (worse) |
| LightGBM | 3.3484 | 3.3465 | -0.177 | 1 | under 0.0001 (worse) |
| ensemble_stack | 3.3807 | 3.3709 | -0.185 | 0 | under 0.0001 (worse) |
| ensemble_rolling | 3.3885 | 3.3823 | -0.170 | 0 | 0.0004 (worse) |
| Prophet | 5.5150 | 5.5150 | -0.906 | 0 | under 0.0001 (worse) |

Nothing beats naive. Phase 3 narrowed XGBoost's gap to naive by about a fifth (0.335 to
0.263 points of mean APE difference); every other model is within 0.01 points of where
Phase 2 left it.

**Open, for the owner** (no rule here decides them):

- The band rule says that if neither candidate passes, the boosters' bands stop being
  presented as 95% bands (the dashboard and README never label them so) and the
  ensemble stops combining them as if they were. How the ensemble's band is then built
  is a design choice not made here.
- Naive's zero-width band in the ensembles (E6) narrows their bands, as above.

**Deviations from the plan**, all decided before any affected number was seen:

- The fixes include F1b (Prophet's seed), found building S2, and F2 covers the
  Bollinger bands' standard deviation and the no-width case as well as the clip.
- Tuning left out one origin (2025-01-03) over the unknown closure, as above; the first
  tuning run stopped on it before scoring anything (MLflow `3911a000`, failed).
- Run times: the base set (14 minutes) and E6 (17.5 minutes) ran past my re-estimates
  of 8 and 14. E6 ran a minute past the 1.5x stop (16.5 minutes) and was let finish; the
  cause was XGBoost's 600 deeper trees, fitted for the forecast and again in the 30-day
  stack, and my own test runs competing for the CPU. A first F1 run blocked on a stderr
  pipe and was stopped (MLflow `101edc3c`, killed).

### The test dates, scored once (T3, 2026-11-08)

T3 (`7beb3ff`, MLflow `93bdf518`, ledger entry 3): the final model set on the 25 test
origins (12 test-warmup origins for the rolling weights), 1,400 cells per model, none
failed. Reported, not used to choose anything (D13).

| Model | T2 MAPE (Phase 2's final) | T3 MAPE (Phase 3's final) | Mean skill | Tickers with positive skill | Holm p against naive |
|---|---|---|---|---|---|
| naive | 3.1481 | 3.1481 | 0 | | |
| ARIMA | 3.1658 | 3.1658 | -0.007 | 1 | 0.41 |
| drift | 3.1887 | 3.1887 | -0.010 | 1 | 0.28 |
| ensemble_rolling | 3.2978 | 3.2852 | -0.041 | 1 | 0.041 (worse) |
| XGBoost | 3.4398 | 3.5377 | -0.170 | 0 | 0.0004 (worse) |
| LightGBM | 3.5579 | 3.5566 | -0.174 | 1 | under 0.0001 (worse) |
| ensemble_stack | 3.6756 | 3.7106 | -0.228 | 0 | under 0.0001 (worse) |
| Prophet | 6.5771 | 6.5771 | -1.096 | 0 | under 0.0001 (worse) |

Nothing beats naive on the test dates either. Between T2 and T3 XGBoost changed by the
fixes F1 and F2 as well as E3's settings, and the ensembles by XGBoost and E6, so no
single change can be read off this table; the per-change measurements are the
validation runs above. The test dates have now been scored three times (T1 and T2 in
Phase 2, T3 in Phase 3); Phase 4 reuses T3 if the code has not changed.

After T3, the pre-checkpoint review led to three commits (`5913d75`, `fa2e175`,
`60ecf03`) that change no forecast: the compare and tune commands' handling of failed
cells and uncommitted runs, a check that the live ensemble's stored base forecasts start
the trading day after the latest close (it raises only when they do not), a check that
the booster adapter's forecast starts at the origin, and tests. A check run at `60ecf03`
(naive and both boosters, AAPL and MSFT, 1,764 validation cells) equals run E6 exactly,
so T3 still describes the final code's forecasts.
