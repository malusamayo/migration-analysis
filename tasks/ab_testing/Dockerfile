FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    gcc g++ make \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /bin/

RUN uv venv /workspace/.venv

ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV HOME=/workspace/project

COPY pyproject.toml uv.lock ./
COPY software-agent-sdk /software-agent-sdk
RUN uv sync --frozen --no-dev --active

# Copy task code used by the remote agent workflow
COPY src /workspace/src
ENV PYTHONPATH=/workspace

# Copy LOCA-bench mcp_convert for the mock Google Cloud MCP server
COPY LOCA-bench/mcp_convert /loca-bench/mcp_convert
ENV LOCA_BENCH_PATH=/loca-bench
# Install mcp_convert (and its deps: mcp, pandas) into the main venv so server.py
# can be run directly with /workspace/.venv/bin/python without needing uv on PATH
RUN uv pip install /loca-bench/mcp_convert

# Make workspace writable for non-root users
RUN chmod -R 777 /workspace

# Set working directory
WORKDIR /workspace

# Expose the default agent server port
EXPOSE 8000

# Run the OpenHands agent server
ENTRYPOINT ["python", "-m", "openhands.agent_server"]
