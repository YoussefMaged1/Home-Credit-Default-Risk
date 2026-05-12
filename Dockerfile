# syntax=docker/dockerfile:1

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen --no-dev --no-install-project

COPY conf ./conf
COPY src ./src
COPY main.py ./main.py
COPY dvc.yaml dvc.lock ./
COPY data/processed ./data/processed

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "src.serving.app"]
