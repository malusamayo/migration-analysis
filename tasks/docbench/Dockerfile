FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gcc g++ make \
    python3-dev \
    poppler-utils \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /bin/

RUN uv venv /workspace/.venv

ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV HOME=/workspace/project

COPY pyproject.toml uv.lock ./
COPY software-agent-sdk /software-agent-sdk
RUN uv sync --frozen --no-dev --active

# Copy task code used by the remote agent workflow.
COPY src /workspace/src
ENV PYTHONPATH=/workspace

# Make workspace writable for non-root users.
RUN chmod -R 777 /workspace

WORKDIR /workspace

EXPOSE 8000

ENTRYPOINT ["python", "-m", "openhands.agent_server"]
