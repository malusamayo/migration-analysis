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
ENV HOME=/workspace
ENV USER=worker

COPY pyproject.toml uv.lock ./
COPY software-agent-sdk /software-agent-sdk
RUN uv sync --frozen --no-dev --active

# Install playwright browsers (required for browser-use / BrowserToolSet)
RUN playwright install --with-deps chromium

# Make workspace writable for non-root users
RUN chmod -R 777 /workspace

# Set working directory
WORKDIR /workspace

# Expose the default agent server port
EXPOSE 8000

# Run the OpenHands agent server
ENTRYPOINT ["python", "-m", "openhands.agent_server"]
