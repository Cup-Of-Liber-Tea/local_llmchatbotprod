# Python 3.10 + CUDA 11.8 베이스 이미지 사용
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Python 3.10 설치 및 기본 설정
ENV PYTHON_VERSION=3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    pip \
    && rm -rf /var/lib/apt/lists/*

# pip 및 python 링크 업데이트
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    pip install --no-cache-dir --upgrade pip

# 작업 디렉토리 설정
WORKDIR /code

# 환경 변수 설정
ENV HF_HOME=/code/.cache/huggingface
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/code

# requirements.txt 복사 (torch 제외)
COPY requirements.txt requirements.txt
# requirements.txt에서 PyTorch 제외한 패키지 먼저 설치
RUN pip install --no-cache-dir -r requirements.txt

# PyTorch, torchvision, torchaudio를 CUDA 11.8 호환 특정 버전으로 설치
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# 애플리케이션 코드 복사
COPY ./app /code/app

# FastAPI 애플리케이션 실행 포트 노출
EXPOSE 8000

# Uvicorn으로 FastAPI 앱 실행 (Dockerfile의 CMD는 docker-compose에서 override 가능)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 