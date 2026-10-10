FROM ghcr.io/astral-sh/uv:0.8.22 AS uv

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Stages are run by path (run_vibe.py, cron `docker exec`), so keep src/ importable
# even outside the venv's editable install.
ENV PYTHONPATH=/app/src
ENV TZ=America/New_York
# Use the image's interpreter; never let uv download another one.
ENV UV_PYTHON=/usr/local/bin/python3 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv
# Plain `python` (docker exec, scripts/*_in_container.sh) resolves to the venv.
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# libgomp1: OpenMP runtime for lightgbm/xgboost. No compiler needed: every
# locked dependency ships a wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

# Dependencies first (cached layer), exactly as locked. Default groups = ml + dev;
# dev (pytest) stays because scripts/test_in_container.sh runs tests in this image.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Sources, templates/, and docs/ (Handbook / Architecture / Labs) for the Flask UI
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

EXPOSE 5000

CMD ["python", "src/finance_vibe/app.py"]
