# QuantFlow Dockerfile
# Builds a container for GCP Cloud Run deployment
# Usage: docker build -t quantflow . && docker run -p 8501:8501 quantflow

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for psycopg2 and prophet
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit dashboard
CMD ["streamlit", "run", "dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]


     