# QuantFlow Makefile
# Run any pipeline stage with a single command.
# Usage: make <target>

.PHONY: help install setup seed indicators anomalies forecast train sentiment dashboard scheduler all clean

# ── Default: show help ────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  QuantFlow — Pipeline Commands"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make install      Install all dependencies"
	@echo "  make setup        Create DB tables (run once)"
	@echo "  make seed         Seed 5 years of historical data"
	@echo "  make indicators   Compute RSI, MACD, Bollinger Bands"
	@echo "  make anomalies    Run anomaly detection (Z-score + IQR)"
	@echo "  make forecast     Run ARIMA + Prophet forecasts"
	@echo "  make train        Run XGBoost + LightGBM models"
	@echo "  make sentiment    Fetch + analyze news sentiment"
	@echo "  make dashboard    Launch Streamlit dashboard"
	@echo "  make scheduler    Start the live data scheduler"
	@echo "  make all          Run full pipeline end to end"
	@echo "  make clean        Remove logs and MLflow artifacts"
	@echo "  ─────────────────────────────────────────────"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

setup:
	psql -U postgres -d stock_pipeline -f db/schema.sql
	psql -U postgres -d stock_pipeline -f db/schema_sentiment.sql
	@echo "Database tables created."

# ── Pipeline stages ───────────────────────────────────────────────────────────
seed:
	python seed_db.py

indicators:
	python indicators.py

anomalies:
	python anomaly_detection.py

forecast:
	python forecasting.py

train:
	python xgboost_model.py

sentiment:
	python sentiment.py

# ── Dashboard + scheduler ─────────────────────────────────────────────────────
dashboard:
	streamlit run dashboard.py

scheduler:
	python scheduler/job_runner.py

# ── Run everything end to end ─────────────────────────────────────────────────
all:
	@echo "Running full QuantFlow pipeline..."
	python seed_db.py
	python indicators.py
	python anomaly_detection.py
	python sentiment.py
	python forecasting.py
	python xgboost_model.py
	@echo "Pipeline complete. Launch dashboard with: make dashboard"

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -name "*.log" -delete
	rm -rf mlruns_artifacts/
	@echo "Cleaned logs and artifacts."