# Real-Time Stock Analytics Pipeline

End-to-end pipeline: ingestion → storage → feature engineering → forecasting → live dashboard.

**Stack:** Python · PostgreSQL · yFinance · Alpha Vantage · APScheduler · Prophet · Streamlit

---

## Project Structure

```
stock_pipeline/
├── config.py                    # Central config (reads .env)
├── seed_db.py                   # One-time historical data seeder
├── requirements.txt
├── .env.example                 # Copy to .env and fill in values
├── .gitignore
│
├── db/
│   ├── connection.py            # SQLAlchemy + psycopg2 helpers
│   └── schema.sql               # Run once to create tables
│
├── ingestion/
│   ├── yfinance_fetcher.py      # Yahoo Finance (free, no key)
│   └── alpha_vantage_fetcher.py # Alpha Vantage REST API
│
├── scheduler/
│   └── job_runner.py            # APScheduler — runs every 15 min
│
├── utils/
│   └── logger.py                # Shared rotating file logger
│
└── logs/                        # Auto-created on first run
```

---

## Phase Roadmap

| Phase | What you build | Status |
|---|---|---|
| 1 | Environment, DB, ingestion | ✅ This folder |
| 2 | Technical indicators (RSI, MACD, Bollinger) | 🔜 |
| 3 | Anomaly detection (Z-score) | 🔜 |
| 4 | Forecasting (ARIMA + Prophet) + MLflow | 🔜 |
| 5 | Streamlit dashboard + Slack/email alerts | 🔜 |

---

## Setup (Windows)

### 1. Install PostgreSQL
Download from https://www.postgresql.org/download/windows/
During install, remember the password you set for the `postgres` user.

### 2. Create the database
Open **pgAdmin** or **psql** (search for it in Start menu):
```sql
CREATE DATABASE stock_pipeline;
```

### 3. Run the schema
In your terminal (Command Prompt or PowerShell), from the project root:
```bash
psql -U postgres -d stock_pipeline -f db/schema.sql
```

### 4. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 5. Set up your .env
```bash
copy .env.example .env
```
Open `.env` in VS Code and fill in:
- `DB_PASSWORD` — the postgres password you set during install
- `ALPHA_VANTAGE_API_KEY` — get a free key at https://www.alphavantage.co/support/#api-key

### 6. Seed historical data (run once)
```bash
python seed_db.py
```

### 7. Start the scheduler
```bash
python scheduler/job_runner.py
```
This will run every 15 minutes and keep your DB updated.

---

## Quick Test

Test each component individually:
```bash
# Test DB connection
python -c "from db.connection import test_connection; test_connection()"

# Test yFinance fetcher
python ingestion/yfinance_fetcher.py

# Test Alpha Vantage fetcher (needs API key in .env)
python ingestion/alpha_vantage_fetcher.py
```

---

## Useful SQL Queries

```sql
-- How many rows per ticker?
SELECT ticker, source, COUNT(*) as rows
FROM raw_prices
GROUP BY ticker, source
ORDER BY ticker;

-- Latest price for each ticker
SELECT DISTINCT ON (ticker) ticker, ts, close
FROM raw_prices
ORDER BY ticker, ts DESC;

-- Check for any gaps in data
SELECT ticker, DATE(ts), COUNT(*)
FROM raw_prices
WHERE source = 'yfinance'
GROUP BY ticker, DATE(ts)
ORDER BY ticker, DATE(ts);
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `DB_HOST` | Postgres host (default: localhost) |
| `DB_PORT` | Postgres port (default: 5432) |
| `DB_NAME` | Database name |
| `DB_USER` | Postgres user |
| `DB_PASSWORD` | Postgres password |
| `ALPHA_VANTAGE_API_KEY` | Free API key from alphavantage.co |
| `TICKERS` | Comma-separated list e.g. `AAPL,MSFT,NVDA` |
| `FETCH_INTERVAL_MINUTES` | How often to fetch (default: 15) |
| `LOG_LEVEL` | INFO, DEBUG, WARNING, ERROR |
