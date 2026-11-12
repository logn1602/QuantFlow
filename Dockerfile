# QuantFlow Dockerfile
# Builds an image that serves the Streamlit dashboard, for running it
# anywhere Docker runs. The live app is on Streamlit Community Cloud, which
# does not use this file.
# Usage: docker build -t quantflow . && docker run -p 8501:8501 --env-file .env quantflow

# 3.11 everywhere: runtime.txt (Streamlit Cloud), the CI workflow, and the
# README prerequisites all target 3.11. No dependency requires 3.12: the
# highest floor across the requirements is >=3.10.
FROM python:3.11.15-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for psycopg2 and prophet
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching). Its last line is ".", the
# quantflow package itself, for Streamlit Cloud; "." cannot be installed
# before the source is here, and installing it from this layer stopped every
# build. So this layer installs every other line, and is reused until
# requirements.txt changes, and the package goes in below, after the code.
COPY requirements.txt .

# Install the third-party dependencies
RUN grep -vx '\.' requirements.txt > /tmp/third-party.txt && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/third-party.txt

# Copy application code
COPY . .

# Install the package itself. requirements.txt above brings in the third-party
# dependencies; this makes `quantflow` importable, which the src/ layout needs.
RUN pip install --no-cache-dir -e . --no-deps

# Create logs directory
RUN mkdir -p logs

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit dashboard
CMD ["streamlit", "run", "src/quantflow/dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
