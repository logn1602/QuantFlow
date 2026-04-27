# QuantFlow — Real-Time Quantitative Stock Analytics Platform

An end-to-end quantitative stock analytics platform combining real-time data ingestion, technical analysis, anomaly detection, multi-model forecasting, and NLP-based news sentiment analysis — visualized through an interactive Streamlit dashboard.

**Stack:** Python · PostgreSQL (Supabase) · yFinance · Alpha Vantage · FinBERT · XGBoost · LightGBM · Prophet · ARIMA · Ridge (scikit-learn) · MLflow · Streamlit · APScheduler · Backtesting

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
                              │                   (315+ headlines/run)
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
                          │  Ridge meta-learner — α tuned via     │
                          │  TimeSeriesSplit CV · learns optimal  │
                          │  combination weights across 4 models  │
                          └───────────────────────────────────────┘
                                                  │
                                                  ▼
                                        MLflow Experiment Tracking
                                  (weights · per-model MAPE · improvement%)
                                                  │
                                                  ▼
                                       Streamlit Dashboard
                                  (5 tabs · 6 live metrics · Supabase)
```

---

## Model Performance

All models evaluated on a 30-day holdout set, trained on 5 years of daily OHLCV data (1,240 rows per ticker).

| Model | Features Used | AAPL MAPE | Avg MAPE (8 tickers) |
|---|---|---|---|
| ARIMA | Price history only | 3.39% | 3.87% |
| Prophet | Price history only | 1.80% | 9.64% |
| XGBoost | 33 engineered features | 1.27% | 1.97% |
| LightGBM | 33 engineered features | 1.29% | 1.94% |
| **Stacking Ensemble** | **Ridge on 4-model holdout predictions** | **best** | **best** |

**XGBoost + LightGBM outperform ARIMA by ~50% on average MAPE** by incorporating technical indicators, anomaly Z-scores, and FinBERT sentiment compound scores. The **stacking ensemble further improves** on the best individual model by training a Ridge meta-learner on 30-day out-of-fold predictions — the learned coefficients reveal which models the ensemble trusts most per ticker.

### Per-Ticker MAPE (5-year training, latest run)

| Ticker | ARIMA | Prophet | XGBoost | LightGBM | Winner |
|---|---|---|---|---|---|
| AAPL | 6.13% | 1.95% | 1.27% | **1.29%** | XGBoost |
| MSFT | 3.89% | 22.83% | 1.55% | **1.51%** | LightGBM |
| GOOGL | 3.60% | 7.29% | 2.88% | **2.46%** | LightGBM |
| AMZN | 1.88% | 9.26% | **1.91%** | 1.92% | ARIMA |
| NVDA | 7.40% | 11.97% | **2.10%** | 2.15% | XGBoost |
| TSLA | 6.71% | 7.11% | 2.58% | **2.55%** | LightGBM |
| META | 6.12% | 11.39% | **2.36%** | 2.38% | XGBoost |
| JPM | 2.61% | 8.21% | **1.60%** | 1.79% | XGBoost |

> Prophet's high MAPE on MSFT (22.83%) and META (11.39%) is due to structural price breaks during the 2022–2023 AI boom — Prophet's seasonality assumptions break down when the underlying trend shifts rapidly. XGBoost and LightGBM handle these breaks better due to their feature-rich input.

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
├── ensemble.py                     # Stacking ensemble — Ridge meta-learner (Layer 3)
├── run_models.py                   # Combined forecasting pipeline (all 5 models)
├── sentiment.py                    # FinBERT news sentiment pipeline
├── dashboard.py                    # Streamlit dashboard (5 tabs)
├── requirements.txt
├── runtime.txt                     # Python 3.11 for Streamlit Cloud
├── Dockerfile                      # GCP Cloud Run ready
├── Makefile                        # One-command pipeline runner
├── .env.example
├── .gitignore
│
├── db/
│   ├── connection.py               # SQLAlchemy + psycopg2 helpers
│   ├── schema.sql                  # Core tables
│   └── schema_sentiment.sql        # Sentiment table
│
├── ingestion/
│   ├── yfinance_fetcher.py         # Yahoo Finance (free, no key)
│   └── alpha_vantage_fetcher.py    # Alpha Vantage REST API
│
├── scheduler/
│   └── job_runner.py               # APScheduler — full pipeline automation
│
└── utils/
    └── logger.py                   # Shared rotating file logger
```

---

## Pipeline Phases

| Phase | Description | Output |
|---|---|---|
| 1 — Ingestion | Pull 5 years of OHLCV data from yFinance + Alpha Vantage into PostgreSQL (Supabase), scheduled every 15 min | 10,040+ rows across 8 tickers |
| 2 — Indicators | Compute RSI (14), MACD (12/26/9), Bollinger Bands (20-period) | 9,880+ indicator rows |
| 3 — Anomalies | Z-score rolling window + IQR method flags unusual price events | 590+ anomaly flags |
| 4 — Forecasting | ARIMA + Prophet 7-day forecasts with MLflow experiment tracking | 112 forecast rows |
| 5 — Dashboard | Interactive Streamlit app — 5 tabs, 6 live metrics, 4-model comparison, deployed on Streamlit Cloud | Live at quantflow-analytics.streamlit.app |
| 6 — Sentiment | FinBERT NLP on 315+ headlines/run from NewsAPI + Yahoo Finance RSS | 633+ sentiment rows |
| Level 2 | XGBoost + LightGBM trained on 33 features including sentiment scores, best MAPE 1.27% | 112 ML forecast rows |
| Level 3 | Stacking ensemble — Ridge meta-learner trained on 30-day out-of-fold holdout predictions from all 4 base models. Alpha tuned via TimeSeriesSplit CV. Logs learned model weights + improvement % to MLflow | 56 ensemble forecast rows |
| Phase 7 | Strategy backtesting — simulates a long/flat trading strategy on the 30-day holdout. Metrics: total return, annualised return, Sharpe ratio, max drawdown, win rate, alpha vs buy-and-hold. 0.1% transaction cost. Results logged to MLflow + displayed in dashboard Backtest tab | 1 result row per ticker |

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
make seed         # Seed 5 years of historical data
make indicators   # Compute RSI, MACD, Bollinger Bands
make anomalies    # Run anomaly detection
make models       # Run ALL 5 models incl. stacking ensemble (recommended)
make sentiment    # Fetch + analyze news headlines
make dashboard    # Launch Streamlit dashboard
make scheduler    # Start the live data scheduler
make all          # Run full pipeline end to end
```

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
```

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

### 5. Seed 5 years of historical data (run once)
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
Tracks RMSE, MAE, MAPE, top features, and forecast artifacts for every model run across all 5 models. For the stacking ensemble, also logs `weight_arima`, `weight_prophet`, `weight_xgboost`, `weight_lightgbm` (Ridge coefficients) and `improvement_pct` over the best base model.

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
| `ALPHA_VANTAGE_API_KEY` | Free key from alphavantage.co |
| `NEWS_API_KEY` | Free key from newsapi.org |
| `TICKERS` | Comma-separated e.g. `AAPL,MSFT,NVDA` |
| `FETCH_INTERVAL_MINUTES` | Scheduler interval (default: 15) |
| `LOG_LEVEL` | INFO, DEBUG, WARNING, ERROR |

---

## Author

**Shubh Dave** — MS Data Analytics @ Northeastern University  
[LinkedIn](https://linkedin.com/in/shubh-dave) · [GitHub](https://github.com/logn1602) · [Live Demo](https://quantflow-analytics.streamlit.app)