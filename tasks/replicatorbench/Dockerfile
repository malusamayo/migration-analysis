# syntax=docker/dockerfile:1.7

FROM python:3.14-slim

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
    cmake \
    python3-dev \
    r-base r-base-dev \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
    libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev \
    libnlopt-dev \
    pandoc jq \
 && rm -rf /var/lib/apt/lists/*

# Make the R system library writable so the agent can install packages at runtime
RUN chmod -R 777 /usr/local/lib/R/

# Install CRAN packages used in social-science replication studies
RUN Rscript -e "install.packages(c( \
    'tidyverse', 'haven', 'estimatr', \
    'lme4', 'lmerTest', 'car', 'afex', 'pbkrtest', \
    'sandwich', 'lmtest', 'fixest', \
    'plyr', 'reshape2', \
    'stargazer', 'broom', 'modelsummary', \
    'nloptr' \
  ), repos='https://cloud.r-project.org', Ncpus=4)"

# Install Bioconductor packages needed by some replication scripts
RUN Rscript -e "install.packages('BiocManager', repos='https://cloud.r-project.org')" && \
    Rscript -e "BiocManager::install(c('multtest', 'metaRNASeq'), ask=FALSE)" && \
    Rscript -e "install.packages('metap', repos='https://cloud.r-project.org')"

COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /bin/uv

RUN useradd --create-home --home-dir /workspace --shell /bin/bash appuser \
 && chown appuser:appuser /workspace

COPY --chown=appuser:appuser pyproject.toml uv.lock ./
COPY --chown=appuser:appuser software-agent-sdk ./software-agent-sdk
COPY --chown=appuser:appuser src ./src

USER appuser
RUN uv venv /workspace/.venv \
 && uv sync --frozen --no-dev --active

ENV PYTHONPATH=/workspace

EXPOSE 8000

ENTRYPOINT ["python", "-m", "openhands.agent_server"]
