# QuantFlow — Real-Time Quantitative Stock Analytics Platform

An end-to-end quantitative stock analytics platform combining real-time data ingestion, technical analysis, anomaly detection, multi-model forecasting, and NLP-based news sentiment analysis — visualized through an interactive Streamlit dashboard.

**Stack:** Python · PostgreSQL · yFinance · Alpha Vantage · FinBERT · XGBoost · LightGBM · Prophet · ARIMA · MLflow · Streamlit · APScheduler

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-blue)](https://www.postgresql.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Pipeline Architecture

```
yFinance API ──────┐
                   ├──► PostgreSQL          NewsAPI + RSS Feeds
Alpha Vantage ─────┘   (raw_prices)    ──► FinBERT Sentiment Analysis
                              │              (315+ headlines/run)
                              ▼                      │
                    Technical Indicators             │
                    (RSI · MACD · Bollinger)         │
                              │                      │
                              ▼                      │
                    Anomaly Detection                │
                    (Z-Score · IQR)                  │
                    │              │                 │
                    ▼              ▼                 ▼
              ┌──────────┐  ┌──────────┐  ┌─────────────────────┐
              │  ARIMA   │  │ Prophet  │  │ XGBoost / LightGBM  │
              │(price    │  │(price    │  │ (33 features: price  │
              │ only)    │  │ only)    │  │  indicators + anomaly│
              └──────────┘  └──────────┘  │  + sentiment scores) │
                    │              │       └─────────────────────┘
                    └──────────────┴──────────────┐
                                                  ▼
                                        MLflow Experiment Tracking
                                                  │
                                                  ▼
                                       Streamlit Dashboard
                                     (5 tabs · 6 live metrics)
```

---

## Model Performance

All models evaluated on a 30-day holdout set, trained on 5 years of daily OHLCV data (1,235 rows per ticker).

| Model | Features Used | AAPL MAPE | Avg MAPE (8 tickers) |
|---|---|---|---|
| ARIMA | Price history only | 3.39% | 3.87% |
| Prophet | Price history only | 1.80% | 9.64% |
| XGBoost | 33 engineered features | 1.13% | 1.85% |
| **LightGBM** | **33 engineered features** | **1.02%** | **1.82%** |

**LightGBM outperforms ARIMA by 58% on average MAPE** by incorporating technical indicators, anomaly Z-scores, and FinBERT sentiment compound scores as features alongside price history.

### Per-Ticker MAPE (5-year training data)

| Ticker | ARIMA | Prophet | XGBoost | LightGBM | Winner |
|---|---|---|---|---|---|
| AAPL | 3.39% | 1.80% | 1.13% | **1.02%** | LightGBM |
| MSFT | 3.07% | 21.80% | 1.53% | **1.45%** | LightGBM |
| GOOGL | 2.57% | 7.27% | **1.95%** | 2.20% | XGBoost |
| AMZN | 4.00% | 10.82% | 2.27% | **1.87%** | LightGBM |
| NVDA | 3.66% | 9.95% | 2.14% | **2.21%** | XGBoost |
| TSLA | 4.67% | 4.72% | 2.29% | **2.16%** | LightGBM |
| META | 4.84% | 11.11% | **2.01%** | 2.11% | XGBoost |
| JPM | 4.74% | 9.65% | 1.49% | **1.50%** | XGBoost |

> Prophet's high MAPE on MSFT (21.80%) and META (11.11%) is due to structural price breaks during the 2022–2023 AI boom period — Prophet's seasonality assumptions break down when the underlying trend shifts rapidly.

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
├── sentiment.py                    # FinBERT news sentiment pipeline
├── dashboard.py                    # Streamlit dashboard (5 tabs)
├── requirements.txt
├── .env.example
├── .gitignore
│
├── db/
│   ├── connection.py               # SQLAlchemy + psycopg2 helpers
│   ├── schema.sql                  # Core tables
│   └── schema_sentiment.sql       # Sentiment table
│
├── ingestion/
│   ├── yfinance_fetcher.py         # Yahoo Finance (free, no key)
│   └── alpha_vantage_fetcher.py   # Alpha Vantage REST API
│
├── scheduler/
│   └── job_runner.py               # APScheduler — runs every 15 min
│
└── utils/
    └── logger.py                   # Shared rotating file logger
```

---

## Pipeline Phases

| Phase | Description | Output |
|---|---|---|
| 1 — Ingestion | Pull 5 years of OHLCV data from yFinance + Alpha Vantage into PostgreSQL, scheduled every 15 min | 10,040+ rows across 8 tickers |
| 2 — Indicators | Compute RSI (14), MACD (12/26/9), Bollinger Bands (20-period) | 9,880 indicator rows |
| 3 — Anomalies | Z-score rolling window + IQR method flags unusual price events | 590 anomaly flags |
| 4 — Forecasting | ARIMA + Prophet 7-day forecasts with MLflow experiment tracking | 112 forecast rows |
| 5 — Dashboard | Interactive Streamlit app — 5 tabs, 6 live metrics, 4-model comparison | Live app |
| 6 — Sentiment | FinBERT NLP on 315+ headlines/run from NewsAPI + Yahoo Finance RSS | 633+ sentiment rows |
| Level 2 | XGBoost + LightGBM trained on 33 features, best MAPE 1.02% | 112 ML forecast rows |

---

## Dashboard Tabs

| Tab | Content |
|---|---|
| Price & Bollinger Bands | Candlestick chart with BB overlay, anomaly spike/crash markers |
| RSI & MACD | Momentum indicators with live overbought/oversold signal interpretation |
| Forecasts | All 4 models on one chart with confidence bands + side-by-side forecast tables |
| Sentiment | FinBERT gauge, daily compound score chart, color-coded headlines |
| Market Overview | Sentiment heatmap, anomaly counts, latest prices for all 8 tickers |

---

## Setup

### Prerequisites
- Python 3.10+
- PostgreSQL 17

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
python forecasting.py             # ARIMA + Prophet forecasts
python xgboost_model.py           # XGBoost + LightGBM forecasts
python sentiment.py               # fetch + analyze news headlines
streamlit run dashboard.py        # launch the dashboard
```

### 7. Start the live scheduler
```bash
python scheduler/job_runner.py    # auto-updates prices every 15 min
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
Tracks RMSE, MAE, MAPE, top features, and forecast artifacts for every model run.

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
| `DB_HOST` | Postgres host (default: localhost) |
| `DB_PORT` | Postgres port (default: 5432) |
| `DB_NAME` | Database name (default: stock_pipeline) |
| `DB_USER` | Postgres user (default: postgres) |
| `DB_PASSWORD` | Postgres password |
| `ALPHA_VANTAGE_API_KEY` | Free key from alphavantage.co |
| `NEWS_API_KEY` | Free key from newsapi.org |
| `TICKERS` | Comma-separated e.g. `AAPL,MSFT,NVDA` |
| `FETCH_INTERVAL_MINUTES` | Scheduler interval (default: 15) |
| `LOG_LEVEL` | INFO, DEBUG, WARNING, ERROR |

---

## Author

**Shubh Dave** — MS Data Analytics @ Northeastern University  
[LinkedIn](https://linkedin.com/in/shubh-dave) · [GitHub](https://github.com/logn1602)