# QuantFlow — Real-Time Quantitative Stock Analytics Platform

An end-to-end quantitative stock analytics platform combining real-time data ingestion, technical analysis, anomaly detection, multi-model forecasting, and NLP-based news sentiment analysis — visualized through an interactive Streamlit dashboard.

**Stack:** Python · PostgreSQL (Supabase) · yFinance · Alpha Vantage · FinBERT · XGBoost · LightGBM · Prophet · ARIMA · NNLS stacking (SciPy) · MLflow · Streamlit · APScheduler · Backtesting

[![Live Demo](https://img.shields.io/badge/Live%20Demo-quantflow--analytics.streamlit.app-red)](https://quantflow-analytics.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Supabase-blue)](https://supabase.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Community%20Cloud-red)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Live Demo

**[quantflow-analytics.streamlit.app](https://quantflow-analytics.streamlit.app)**

---

## Pipeline Architecture

```
yFinance API ──────┐
                   ├──► PostgreSQL (Supabase)    NewsAPI + RSS Feeds
Alpha Vantage ─────┘      (raw_prices)       ──► FinBERT Sentiment Analysis
                              │                  (~290 headlines/run)
                              ▼                          │
                    Technical Indicators                 │
                    (RSI · MACD · Bollinger)             │
                              │                          │
                              ▼                          │
                    Anomaly Detection                    │
                    (Z-Score · IQR)                      │
                    │              │                     │
                    ▼              ▼                     ▼
              ┌──────────┐  ┌──────────┐  ┌─────────────────────────┐
              │  ARIMA   │  │ Prophet  │  │  XGBoost / LightGBM     │
              │(price    │  │(price    │  │  (33 features: price +   │
              │ only)    │  │ only)    │  │  indicators + anomaly +  │
              └──────────┘  └──────────┘  │  sentiment scores)       │
                    │              │       └─────────────────────────┘
                    └──────────────┴──────────────┐
                                                  ▼
                              30-day out-of-fold holdout predictions
                                                  │
                                                  ▼
                          ┌───────────────────────────────────────┐
                          │  Stacking Ensemble (ensemble.py)      │
                          │  NNLS meta-learner — weights >= 0     │
                          │  summing to 1 (convex combination).   │
                          │  Scored out-of-fold via an expanding  │
                          │  window, so no day sees its own actual│
                          └───────────────────────────────────────┘
                                                  │
                                                  ▼
                                        MLflow Experiment Tracking
                              (weights · per-model MAPE · oof_mape · eval_days)
                                                  │
                                                  ▼
                                       Streamlit Dashboard
                                  (6 tabs · 6 live metrics · Supabase)
```

---

## Model Performance

> **All figures below were measured on 2026-08-04** by running `python run_models.py`
> across all 8 tickers and reading the `model_metrics` table. They shift as new
> market data arrives — regenerate rather than trusting a stale copy.

Base models train on the daily close series assembled by `forecasting.load_prices`,
which collapses intraday bars to one close per date: **579 daily closes per ticker**,
spanning 2024-04-10 to 2026-08-04. The last 30 days are held out for evaluation.

### Base models — 30-day holdout

| Model | Features Used | AAPL MAPE | Avg MAPE (8 tickers) |
|---|---|---|---|
| ARIMA | Price history only | 5.68% | 5.48% |
| Prophet | Price history only | 6.92% | 12.86% |
| XGBoost | 33 engineered features | **4.23%** | **3.07%** |
| LightGBM | 33 engineered features | 4.83% | 3.20% |

**XGBoost and LightGBM beat ARIMA by ~42% on average MAPE** (3.07% and 3.20% vs
5.48%) by incorporating technical indicators, anomaly Z-scores, and FinBERT
sentiment compound scores. XGBoost wins outright on 6 of 8 tickers.

### Stacking ensemble — does it help?

Mostly no, and the honest number says so.

The ensemble is evaluated **out-of-fold**: for each day, an NNLS meta-learner is
fitted only on the days before it, so no prediction ever sees its own actual.
Fitting the first meta-learner consumes a 10-day warmup, leaving a **20-day**
evaluation window. Base models are re-scored on that same 20 days so the
comparison is like-for-like.

| Ticker | Eval window | Best base model (same window) | Ensemble (out-of-fold) | vs best base |
|---|---|---|---|---|
| AAPL | 20 days | Prophet 5.19% | **2.83%** | **+45.47%** |
| META | 20 days | XGBoost 2.87% | 3.10% | −8.01% |
| JPM | 20 days | Prophet 2.79% | 3.55% | −27.24% |
| MSFT | 20 days | LightGBM 3.12% | 6.30% | −101.92% |
| TSLA | 20 days | XGBoost 2.58% | 5.37% | −108.14% |
| GOOGL | 20 days | LightGBM 2.31% | 5.29% | −129.00% |
| NVDA | 20 days | XGBoost 2.10% | 8.59% | −309.05% |
| AMZN | 20 days | LightGBM 2.32% | 11.44% | −393.10% |

**The ensemble beats the best single base model on 1 of 8 tickers (AAPL).** Mean
out-of-fold ensemble MAPE is 5.81%, against best-base figures clustered around
2–3%.

Earlier versions of this README claimed the ensemble was "best" on every ticker.
That claim came from scoring the meta-learner on the same 30 days it was fitted
on, which cannot lose by construction. With the leak removed, a convex
combination that must keep all weights ≥ 0 gets dragged toward whichever base
models are noisy on a given ticker, and 20 evaluation days is too few for the
weights to settle. The ensemble is retained because it is genuinely useful on
AAPL and because its weights are a readable diagnostic of which model the data
favours — not because it is a general improvement.

### Per-ticker base-model MAPE (30-day holdout)

| Ticker | ARIMA | Prophet | XGBoost | LightGBM | Winner |
|---|---|---|---|---|---|
| AAPL | 5.68% | 6.92% | **4.23%** | 4.83% | XGBoost |
| MSFT | 4.52% | 12.81% | 2.89% | **2.64%** | LightGBM |
| GOOGL | 3.66% | 15.14% | **2.28%** | 2.38% | XGBoost |
| AMZN | 3.62% | 16.72% | **2.26%** | 2.30% | XGBoost |
| NVDA | 5.46% | 18.23% | **2.18%** | 2.58% | XGBoost |
| TSLA | 11.07% | 21.35% | **2.60%** | 2.79% | XGBoost |
| META | 6.27% | 9.32% | **2.98%** | 3.13% | XGBoost |
| JPM | 3.56% | **2.40%** | 5.14% | 4.94% | Prophet |

> Prophet is the weakest model on 7 of 8 tickers, worst on TSLA (21.35%), NVDA
> (18.23%) and AMZN (16.72%) — its additive trend-plus-seasonality assumption
> degrades when a stock re-rates sharply, which these did over the sample. The
> exception is JPM, where Prophet is the single best model (2.40%): a slower,
> more mean-reverting series is exactly what it is built for. The gradient
> boosters, which get indicators and sentiment as features, are more robust to
> those breaks but lose to Prophet when there is no break to exploit.

---

## Project Structure

```
QuantFlow/
├── config.py                       # Central config (reads .env)
├── seed_db.py                      # One-time historical data seeder
├── indicators.py                   # RSI, MACD, Bollinger Bands engine
├── anomaly_detection.py            # Z-score + IQR anomaly detection
├── forecasting.py                  # ARIMA + Prophet forecasting
├── xgboost_model.py                # XGBoost + LightGBM with feature engineering
├── ensemble.py                     # Stacking ensemble — NNLS meta-learner (Layer 3)
├── run_models.py                   # Combined forecasting pipeline (all 5 models)
├── backtest.py                     # Long/flat strategy backtest on out-of-fold predictions
├── sentiment.py                    # FinBERT news sentiment pipeline
├── dashboard.py                    # Streamlit dashboard (6 tabs)
├── conftest.py                     # Puts the repo root on sys.path for pytest
├── requirements.txt
├── runtime.txt                     # Python 3.11 for Streamlit Cloud
├── Dockerfile                      # GCP Cloud Run ready
├── Makefile                        # One-command pipeline runner
├── LICENSE                         # MIT
├── .env.example                    # Environment template — copy to .env
├── .gitignore
│
├── db/
│   ├── connection.py               # SQLAlchemy + psycopg2 helpers
│   ├── metrics.py                  # Reads/writes model_metrics for the dashboard
│   ├── schema.sql                  # Core tables
│   ├── schema_sentiment.sql        # Sentiment table
│   ├── schema_backtest.sql         # Backtest results table
│   ├── schema_metrics.sql          # Per-run model metrics table
│   └── migrations/
│       └── 001_dedupe_anomalies.sql  # One-time: dedupe anomalies + add UNIQUE
│
├── ingestion/
│   ├── yfinance_fetcher.py         # Yahoo Finance (free, no key)
│   └── alpha_vantage_fetcher.py    # Alpha Vantage REST API
│
├── scheduler/
│   └── job_runner.py               # APScheduler — full pipeline automation
│
├── tests/                          # Offline pytest suite (no DB, no network)
│   ├── test_backtest.py            # simulate_strategy + compute_metrics
│   ├── test_ensemble.py            # NNLS meta-learner + out-of-fold guarantees
│   ├── test_features.py            # Feature engineering + leakage checks
│   └── test_config.py              # Ticker parsing + config validation
│
├── utils/
│   └── logger.py                   # Shared rotating file logger
│
└── .github/
    └── workflows/
        └── daily_refresh.yml       # Daily CI data refresh (weekdays, 22:30 UTC)
```

---

## Pipeline Phases

Row counts measured 2026-08-04. Ingestion is cumulative, so the price, indicator
and sentiment counts grow every day; the forecast counts are rewritten each run.

| Phase | Description | Output |
|---|---|---|
| 1 — Ingestion | Pull OHLCV bars from yFinance + Alpha Vantage into PostgreSQL (Supabase), scheduled every 15 min | 21,032 rows across 8 tickers (2,529 per ticker from yFinance, ~26 intraday bars per trading day, collapsed to 579 daily closes for modelling) |
| 2 — Indicators | Compute RSI (14), MACD (12/26/9), Bollinger Bands (20-period) | 20,032 indicator rows |
| 3 — Anomalies | Z-score rolling window + IQR method flags unusual price events | 4,993 distinct (ticker, timestamp) flags — see the note below on duplicate rows |
| 4 — Forecasting | ARIMA + Prophet 7-day forecasts with MLflow experiment tracking | 112 forecast rows (7 days × 2 models × 8 tickers) |
| 5 — Dashboard | Interactive Streamlit app — 6 tabs, 6 live metrics, 5-model comparison, deployed on Streamlit Cloud | Live at quantflow-analytics.streamlit.app |
| 6 — Sentiment | FinBERT NLP on ~290 headlines per run from NewsAPI + Yahoo Finance RSS | 22,558 sentiment rows (16,754 distinct headlines) |
| Level 2 | XGBoost + LightGBM trained on 33 features including sentiment scores, best MAPE 2.18% (NVDA) | 112 ML forecast rows |
| Level 3 | Stacking ensemble — NNLS meta-learner (weights ≥ 0 summing to 1) over the 4 base models' 30-day holdout predictions, scored out-of-fold on a 20-day window. Logs learned weights, `oof_mape`, `eval_days` and `improvement_pct` to MLflow | 48 ensemble forecast rows (6 usable forecast dates × 8 tickers) |
| Phase 7 | Strategy backtesting — long/flat strategy simulated on the 20-day out-of-fold window. Metrics: total return, annualised return, Sharpe ratio, max drawdown, win rate, alpha vs buy-and-hold. 0.1% transaction cost. Results logged to MLflow + displayed in dashboard Backtest tab | 1 result row per ticker (8 rows) |

> **Duplicated anomaly rows — fixed in code, migration pending.** The live
> `anomalies` table holds 269,350 rows for only 4,993 distinct `(ticker, ts)`
> pairs — ~54 copies of each flag (max 82). Root cause: the table carries only
> `PRIMARY KEY (id)`. The `UNIQUE (ticker, ts)` in `db/schema.sql` was never
> applied, because the table already existed when that clause was added and
> `CREATE TABLE IF NOT EXISTS` cannot retrofit a constraint. With no unique
> index, the bare `ON CONFLICT DO NOTHING` in `save_anomalies` had nothing to
> detect, so every run appended a full copy.
>
> `save_anomalies` now replaces each ticker's flags inside one transaction
> instead of appending, which is correct regardless of constraint state and
> keeps the freshest estimates — the rolling Z-score and IQR fences are
> recomputed as history grows, so 2,300 keys legitimately differ between
> copies. Run `db/migrations/001_dedupe_anomalies.sql` once to collapse the
> existing duplicates (keeping the newest row per key) and add the missing
> constraint. Fresh installs are unaffected: `schema.sql` creates the
> constraint correctly. Until the migration runs, dashboard anomaly counts stay
> inflated; the flagged timestamps are correct either way.

---

## Dashboard Tabs

| Tab | Content |
|---|---|
| Price & Bollinger Bands | Candlestick chart with BB overlay, anomaly spike/crash markers |
| RSI & MACD | Momentum indicators with live overbought/oversold signal interpretation |
| Forecasts | All 4 base models + stacking ensemble on one chart. Ensemble shown as bold gold line. Side-by-side 7-day tables + MAPE comparison across all 5 models |
| Backtest | Cumulative return chart (strategy vs buy-and-hold), 6 live performance metrics, full summary table for all tickers with alpha colour-coded |
| Sentiment | FinBERT gauge, daily compound score chart, color-coded headlines |
| Market Overview | Sentiment heatmap for all tickers, anomaly counts, latest prices |

---

## Makefile Commands

```bash
make install      # Install all dependencies
make setup        # Create DB tables (run once)
make seed         # Seed 2 years of historical data
make indicators   # Compute RSI, MACD, Bollinger Bands
make anomalies    # Run anomaly detection
make models       # Run ALL 5 models incl. stacking ensemble (recommended)
make sentiment    # Fetch + analyze news headlines
make backtest     # Simulate the trading strategy (run models first)
make test         # Run the offline test suite (no DB needed)
make dashboard    # Launch Streamlit dashboard
make scheduler    # Start the live data scheduler
make all          # Run full pipeline end to end
make clean-dry    # Preview what clean would remove
make clean        # Remove logs and caches (keeps mlflow.db)
```

`make forecast`, `make train` and `make ensemble` run individual model families
if you need them; `make models` is the recommended path.

---

## Setup

### Prerequisites
- Python 3.11+
- PostgreSQL 17 (or Supabase for cloud)

### 1. Clone the repo
```bash
git clone https://github.com/logn1602/QuantFlow.git
cd QuantFlow
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Set up PostgreSQL
```bash
psql -U postgres -c "CREATE DATABASE stock_pipeline;"
psql -U postgres -d stock_pipeline -f db/schema.sql
psql -U postgres -d stock_pipeline -f db/schema_sentiment.sql
psql -U postgres -d stock_pipeline -f db/schema_backtest.sql
psql -U postgres -d stock_pipeline -f db/schema_metrics.sql
```
All four schema files are required. Skipping `schema_backtest.sql` breaks the
Backtest tab; skipping `schema_metrics.sql` leaves the Forecasts tab with no
MAPE figures. Once the database exists, `make setup` runs the same four files.

### 4. Configure environment
```bash
cp .env.example .env
```
Open `.env` and fill in:
```
DB_PASSWORD=your_postgres_password
ALPHA_VANTAGE_API_KEY=your_key    # free at alphavantage.co
NEWS_API_KEY=your_key             # free at newsapi.org
```

### 5. Seed 2 years of historical data (run once)
```bash
python seed_db.py
```

### 6. Run the full pipeline
```bash
python indicators.py              # compute RSI, MACD, Bollinger Bands
python anomaly_detection.py       # detect price anomalies
python sentiment.py               # fetch + analyze news headlines
python run_models.py              # run all 5 models (ARIMA, Prophet, XGBoost, LightGBM, Ensemble)
python backtest.py                # simulate trading strategy, compute Sharpe ratio + alpha
streamlit run dashboard.py        # launch the dashboard
```

### 7. Start the live scheduler
```bash
python scheduler/job_runner.py    # auto-updates everything on schedule
```

---

## Tickers Tracked

`AAPL` · `MSFT` · `GOOGL` · `AMZN` · `NVDA` · `TSLA` · `META` · `JPM`

---

## MLflow Experiment Tracking

```bash
mlflow ui
# Open http://localhost:5000
```
Tracks RMSE, MAE, MAPE, top features, and forecast artifacts for every model run across all 5 models.

For the stacking ensemble it also logs:

| Key | Meaning |
|---|---|
| `weight_arima`, `weight_prophet`, `weight_xgboost`, `weight_lightgbm` | The learned NNLS weights — non-negative, summing to 1 |
| `oof_mape` | Out-of-fold ensemble MAPE (the honest number) |
| `eval_days` | Length of the out-of-fold evaluation window (20) |
| `arima_mape`, `prophet_mape`, `xgboost_mape`, `lightgbm_mape` | Base-model MAPE over that *same* window, for a like-for-like comparison |
| `improvement_pct` | Ensemble vs best base model on that window. Negative when the ensemble loses, which is most tickers |
| `meta_learner` | `nnls_convex` |
| `warmup_days` | Days consumed fitting the first meta-learner (10) |

MLflow writes to a local `mlflow.db`, which the deployed Streamlit app cannot read — that is why the dashboard reads its MAPE figures from the `model_metrics` table in Postgres instead.

---

## Database TLS

The database is a public Supabase endpoint reached from three places — a
laptop, a GitHub Actions runner, and Streamlit Cloud — so the connection is
explicitly encrypted rather than left to libpq's defaults.

**What ships:** `DB_SSLMODE=require`. Verified active on both connection paths
(`get_engine` and `get_conn`): TLS 1.3, `TLS_AES_256_GCM_SHA384`, 256-bit.

`require` was chosen over libpq's default `prefer` because `prefer` falls back
to **cleartext silently** — no exception, no log line — if the peer ever
declines TLS. It was chosen over `verify-full` because Supabase signs its
pooler certificate with a private CA:

```
subject: CN=*.pooler.supabase.com, O=Supabase Inc
issuer:  CN=Supabase Intermediate 2021 CA, O=Supabase Inc
```

That CA is in neither the system trust store nor certifi's bundle, so
`verify-full` fails out of the box with `certificate verify failed`.

### Upgrading to verify-full (recommended)

`require` encrypts but does not authenticate the peer, so it does not stop an
active man-in-the-middle. To close that:

1. Supabase dashboard → **Project Settings → Database → SSL Configuration** →
   download the CA certificate.
2. Save it outside the repo, then set both variables:

```bash
DB_SSLMODE=verify-full
DB_SSLROOTCERT=/absolute/path/to/prod-ca-2021.crt
```

3. Confirm it works before deploying:

```bash
python -c "from db.connection import test_connection; print(test_connection())"
```

**If you set these locally, set them in GitHub Actions secrets and Streamlit
Cloud too.** The certificate file has to exist on each host, which is why
`verify-full` is not the shipped default — a path that exists on your laptop
does not exist on a CI runner.

---

## Deployment

The live app is deployed on **Streamlit Community Cloud** backed by **Supabase** (managed PostgreSQL).

- **Frontend:** Streamlit Community Cloud (free)
- **Database:** Supabase PostgreSQL (free tier)
- **Docker:** `Dockerfile` included for GCP Cloud Run deployment

---

## Useful SQL Queries

```sql
-- Row counts per ticker and source
SELECT ticker, source, COUNT(*) AS rows
FROM raw_prices
GROUP BY ticker, source
ORDER BY ticker;

-- Latest price per ticker
SELECT DISTINCT ON (ticker) ticker, ts, close
FROM raw_prices
ORDER BY ticker, ts DESC;

-- Latest sentiment per ticker
SELECT ticker, ROUND(AVG(compound)::numeric, 3) AS avg_compound
FROM news_sentiment
WHERE published_at >= NOW() - INTERVAL '7 days'
GROUP BY ticker
ORDER BY avg_compound DESC;
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `DB_HOST` | Postgres host |
| `DB_PORT` | Postgres port (default: 5432) |
| `DB_NAME` | Database name |
| `DB_USER` | Postgres user |
| `DB_PASSWORD` | Postgres password |
| `DB_SSLMODE` | TLS mode for the DB connection (default: `require`) |
| `DB_SSLROOTCERT` | CA bundle path, only needed for `verify-ca` / `verify-full` |
| `ALPHA_VANTAGE_API_KEY` | Free key from alphavantage.co |
| `NEWS_API_KEY` | Free key from newsapi.org |
| `TICKERS` | Comma-separated e.g. `AAPL,MSFT,NVDA` |
| `FETCH_INTERVAL_MINUTES` | Scheduler interval (default: 15) |
| `LOG_LEVEL` | INFO, DEBUG, WARNING, ERROR |

---

## Author

**Shubh Dave** — MS Data Analytics @ Northeastern University  
[LinkedIn](https://linkedin.com/in/shubh-dave) · [GitHub](https://github.com/logn1602) · [Live Demo](https://quantflow-analytics.streamlit.app)