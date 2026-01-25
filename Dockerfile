# 베이스 이미지 설정 (안정적인 Bullseye 버전 사용)
FROM python:3.10-slim-bullseye

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRUST_REMOTE_CODE=1

# 시스템 종속성 설치 (OpenCV, Anomalib 등을 위한 라이브러리)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 종속성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt


# 프로젝트 파일 복사 ( .dockerignore 에 정의된 파일 제외 )
COPY . .

# 모델 파일이 존재하지 않을 경우를 대비해 디렉토리 구조 생성 (있다면 덮어씀)
RUN mkdir -p exported_models/weights/torch

# 포트 설정 (Streamlit 기본 포트)
EXPOSE 8501

# 실행 명령 설정
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
