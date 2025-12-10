FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY best2.pt .
COPY best_dlab30.pth .
COPY api_combined.py .

EXPOSE 8000
CMD ["uvicorn", "api_combined:app", "--host", "0.0.0.0", "--port", "8000"]
