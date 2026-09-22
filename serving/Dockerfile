FROM python:3.12-slim

WORKDIR /app

# Install system deps needed by bitsandbytes / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install project dependencies first (layer caches when code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install serving-specific dependencies
COPY serving/requirements_serving.txt .
RUN pip install --no-cache-dir -r requirements_serving.txt

# Copy project source
COPY . .

EXPOSE 8000

CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
