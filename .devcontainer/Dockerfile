FROM python:3.8.2-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    apt-utils \
    gcc \
    build-essential \
    libopencv-dev \
    && pip install --no-cache-dir \
    autopep8 \
    flake8 \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*