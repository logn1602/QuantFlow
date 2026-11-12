# QuantFlow Makefile
# Run any pipeline stage with a single command.
# Usage: make <target>

.PHONY: help install requirements setup seed indicators anomalies models forecast train ensemble sentiment backtest evaluate trending dashboard scheduler test eval all clean clean-dry

# ── Default: show help ────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  QuantFlow: Pipeline Commands"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make install      Install all dependencies"
	@echo "  make requirements Write requirements.txt from uv.lock (after any lock change)"
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
	@echo "  make evaluate     Rolling-origin evaluation, validation dates (SNAPSHOT=dir)"
	@echo "  make trending     One trending run, on .env's database (CAP=n: at most n discovered)"
	@echo "  make dashboard    Launch Streamlit dashboard"
	@echo "  make scheduler    Start the live data scheduler"
	@echo "  make test         Run the offline test suite (no DB needed)"
	@echo "  make eval         Run the evaluation suite (slower)"
	@echo "  make all          Run full pipeline end to end"
	@echo "  make clean        Remove logs and caches (keeps mlflow.db)"
	@echo "  make clean-dry    Preview what clean would remove"
	@echo "  ─────────────────────────────────────────────"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
# requirements.txt is uv.lock's exact versions, the dev tools included, so
# pip installs what CI tests. Its "." line installs a copy of the quantflow
# package; -e . replaces it with an editable install, so the targets below
# (python -m quantflow.*) run the checkout's code and read its config/ files.
install:
	pip install -r requirements.txt
	pip install -e . --no-deps

# Rewrite requirements.txt from uv.lock, after any change to the lock (uv
# lock, uv add). tests/unit/test_requirements.py fails while the two differ.
requirements:
	uv export --format requirements.txt --frozen --no-hashes --no-editable --extra dev --output-file requirements.txt

setup:
	psql -U postgres -d stock_pipeline -f sql/schema.sql
	psql -U postgres -d stock_pipeline -f sql/schema_sentiment.sql
	psql -U postgres -d stock_pipeline -f sql/schema_backtest.sql
	psql -U postgres -d stock_pipeline -f sql/schema_metrics.sql
	psql -U postgres -d stock_pipeline -f sql/schema_trending.sql
	@echo "Database tables created."

# ── Pipeline stages ───────────────────────────────────────────────────────────
seed:
	python -m quantflow.pipelines.seed

indicators:
	python -m quantflow.features.indicators

anomalies:
	python -m quantflow.features.anomalies

# Every model, the ensemble included, in one command (recommended)
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
	python -m quantflow.pipelines.backtest

# Reads a snapshot made by "python -m quantflow.pipelines.evaluate export";
# see docs/evaluation.md. Test-date runs are deliberately not a target.
SNAPSHOT ?= eval_snapshots/2026-11-06
evaluate:
	python -m quantflow.pipelines.evaluate run --snapshot $(SNAPSHOT)

sentiment:
	python -m quantflow.features.sentiment

# The trending tier (docs/trending.md): discovery and analytics only, in its
# own tables, never forecast or backtested. Writes to the database .env names,
# as every target here does. CAP=3 tracks at most 3 discovered symbols, for a
# local check: a run like any other, so the active symbols past the cap are
# deactivated until a run without it.
CAP ?=
trending:
	python -m quantflow.pipelines.trending $(if $(CAP),--cap $(CAP),)

# ── Tests ─────────────────────────────────────────────────────────────────────
# Offline only: no database, no network, no model training.
test:
	pytest -m "not eval and not integration"

# Evaluation suite: exercises the full scoring path. Slower, and excluded from
# `make test` and CI for that reason.
eval:
	pytest -m eval

# ── Dashboard + scheduler ─────────────────────────────────────────────────────
dashboard:
	streamlit run src/quantflow/dashboard/app.py

scheduler:
	python -m quantflow.pipelines.scheduler

# ── Run everything end to end ─────────────────────────────────────────────────
all:
	@echo "Running full QuantFlow pipeline..."
	python -m quantflow.pipelines.seed
	python -m quantflow.ingestion.split_guard
	python -m quantflow.features.indicators
	python -m quantflow.features.anomalies
	python -m quantflow.features.sentiment
	python -m quantflow.pipelines.run_models
	python -m quantflow.pipelines.backtest
	@echo "Pipeline complete. Launch dashboard with: make dashboard"

# ── Cleanup ───────────────────────────────────────────────────────────────────
# Delegates to a Python script so this works on Windows too. The old target
# shelled out to find/rm, which do not exist in PowerShell.
# Keeps mlflow.db by default; see scripts/clean.py --mlflow.
clean:
	python scripts/clean.py

clean-dry:
	python scripts/clean.py --dry-run
