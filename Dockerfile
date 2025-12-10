FROM python:3.10-slim

# 필수 OS 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# lightweight PyTorch 설치 (CPU 전용)
RUN pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# requirements.txt 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 모델 및 API 파일 복사
COPY best2.pt .
COPY best_dlab30.pth .
COPY api_combined.py .

EXPOSE 8000
CMD ["uvicorn", "api_combined:app", "--host", "0.0.0.0", "--port", "8000"]
