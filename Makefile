# QuantFlow Makefile
# Run any pipeline stage with a single command.
# Usage: make <target>

.PHONY: help install setup seed indicators anomalies models forecast train ensemble sentiment backtest dashboard scheduler test all clean clean-dry

# ── Default: show help ────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  QuantFlow — Pipeline Commands"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make install      Install all dependencies"
	@echo "  make setup        Create DB tables (run once)"
	@echo "  make seed         Seed 2 years of historical data"
	@echo "  make indicators   Compute RSI, MACD, Bollinger Bands"
	@echo "  make anomalies    Run anomaly detection (Z-score + IQR)"
	@echo "  make models       Run ALL 5 models (ARIMA + Prophet + XGBoost + LightGBM + Ensemble)"
	@echo "  make forecast     Run ARIMA + Prophet only"
	@echo "  make train        Run XGBoost + LightGBM only"
	@echo "  make ensemble     Run stacking ensemble only"
	@echo "  make sentiment    Fetch + analyze news sentiment"
	@echo "  make backtest     Run strategy backtest (run models first)"
	@echo "  make dashboard    Launch Streamlit dashboard"
	@echo "  make scheduler    Start the live data scheduler"
	@echo "  make test         Run the offline test suite (no DB needed)"
	@echo "  make all          Run full pipeline end to end"
	@echo "  make clean        Remove logs and caches (keeps mlflow.db)"
	@echo "  make clean-dry    Preview what clean would remove"
	@echo "  ─────────────────────────────────────────────"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

setup:
	psql -U postgres -d stock_pipeline -f sql/schema.sql
	psql -U postgres -d stock_pipeline -f sql/schema_sentiment.sql
	psql -U postgres -d stock_pipeline -f sql/schema_backtest.sql
	psql -U postgres -d stock_pipeline -f sql/schema_metrics.sql
	@echo "Database tables created."

# ── Pipeline stages ───────────────────────────────────────────────────────────
seed:
	python -m quantflow.pipelines.seed

indicators:
	python -m quantflow.features.indicators

anomalies:
	python -m quantflow.features.anomalies

# Run all 4 models in one command (recommended)
models:
	python -m quantflow.pipelines.run_models

# Run individual model families if needed
forecast:
	python -m quantflow.models.statistical

train:
	python -m quantflow.models.boosting

ensemble:
	python -m quantflow.models.ensemble

backtest:
	python -m quantflow.evaluation.backtest

sentiment:
	python -m quantflow.features.sentiment

# ── Tests ─────────────────────────────────────────────────────────────────────
# Offline only: no database, no network, no model training.
test:
	pytest -q

# ── Dashboard + scheduler ─────────────────────────────────────────────────────
dashboard:
	streamlit run src/quantflow/dashboard/app.py

scheduler:
	python -m quantflow.pipelines.scheduler

# ── Run everything end to end ─────────────────────────────────────────────────
all:
	@echo "Running full QuantFlow pipeline..."
	python -m quantflow.pipelines.seed
	python -m quantflow.features.indicators
	python -m quantflow.features.anomalies
	python -m quantflow.features.sentiment
	python -m quantflow.pipelines.run_models
	python -m quantflow.evaluation.backtest
	@echo "Pipeline complete. Launch dashboard with: make dashboard"

# ── Cleanup ───────────────────────────────────────────────────────────────────
# Delegates to a Python script so this works on Windows too. The old target
# shelled out to find/rm, which do not exist in PowerShell.
# Keeps mlflow.db by default — see scripts/clean.py --mlflow.
clean:
	python scripts/clean.py

clean-dry:
	python scripts/clean.py --dry-run
