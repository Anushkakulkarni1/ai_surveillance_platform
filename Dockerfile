

ARG PYTHON_VERSION=3.14


FROM python:${PYTHON_VERSION}-slim-bookworm AS builder


RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1


RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /build

COPY requirements.txt ./requirements.txt
COPY backend/requirements.txt ./backend/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt -r backend/requirements.txt

COPY ml/ ./ml/

ARG MODEL_CHECKPOINT_PATH=ml/best_model.pt
ARG VAD_BASE_CHANNELS=16


RUN python ml/export_onnx.py \
        --checkpoint "${MODEL_CHECKPOINT_PATH}" \
        --output models/vad_autoencoder.onnx \
        --base_channels "${VAD_BASE_CHANNELS}" \
        --device cpu


FROM python:${PYTHON_VERSION}-slim-bookworm AS runner


RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false


RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --shell /usr/sbin/nologin --no-create-home appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

COPY backend/ ./backend/
COPY dashboard/ ./dashboard/
COPY ai/ ./ai/
COPY ml/ ./ml/
COPY detection/ ./detection/
COPY config.py ./config.py

COPY --from=builder /build/models/ ./models/


RUN mkdir -p /app/logs /app/evidence /app/knowledge /app/.streamlit \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501


HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)" || exit 1


CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
