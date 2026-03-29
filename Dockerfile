FROM python:3.12-slim AS base
WORKDIR /app
RUN adduser --system --no-create-home appuser

FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    "cowork-platform[sdk] @ git+https://github.com/suman724/cowork-platform.git@main" \
    "cowork-agent-sdk @ git+https://github.com/suman724/cowork-agent-sdk.git@main"
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

FROM base AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY src/ src/
RUN mkdir -p /workspace && chown appuser /workspace
USER appuser
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["curl", "-f", "http://localhost:8080/health"]
CMD ["python", "-m", "agent_host.main", "--transport", "http"]
