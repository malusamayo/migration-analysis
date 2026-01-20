FROM python:3.14-slim

# Install additional Python dependencies needed for your project

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    gcc g++ make \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /bin/

RUN uv venv /workspace/.venv

ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY pyproject.toml uv.lock ./
COPY --chown=openhands:openhands software-agent-sdk /software-agent-sdk
RUN uv sync --frozen --no-dev --active

# Install playwright browsers (required for playwright)
RUN playwright install --with-deps chromium

# Optional: Copy your source code if needed by the agent
# COPY src /workspace/src

# Set working directory
WORKDIR /workspace

# Expose the default agent server port
EXPOSE 8000

# Run the OpenHands agent server
ENTRYPOINT ["python", "-m", "openhands.agent_server"]