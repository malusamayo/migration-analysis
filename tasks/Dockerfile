# syntax=docker/dockerfile:1.7

# Shared base for the closest-match task images.
# Exact or near-exact variants:
# - tasks/webtest/Dockerfile
# - tasks/webarena/Dockerfile
# - tasks/browsecomplongcontext/Dockerfile
# - tasks/refactorbench/Dockerfile (+ COPY src ./src)
# - tasks/docbench/Dockerfile (+ poppler-utils, Playwright)
# - tasks/webgen/Dockerfile (+ nodejs npm chromium, Playwright)

FROM python:3.14-slim

ARG USER_UID

ENV HOME=/workspace \
    VIRTUAL_ENV=/workspace/.venv \
    PATH="/workspace/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gcc \
    g++ \
    make \
    python3-dev \
    poppler-utils \
    jq \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /bin/uv

RUN useradd --create-home --home-dir /workspace --shell /bin/bash --uid $USER_UID appuser \
 && chown appuser:appuser /workspace

COPY --chown=appuser:appuser pyproject.toml uv.lock ./
COPY --chown=appuser:appuser software-agent-sdk ./software-agent-sdk

USER appuser
RUN uv venv /workspace/.venv \
 && uv sync --frozen --no-dev --active

USER root
RUN python -m playwright install --with-deps chromium
USER appuser

ENV PYTHONPATH=/workspace

EXPOSE 8000

ENTRYPOINT ["python", "-m", "openhands.agent_server"]
