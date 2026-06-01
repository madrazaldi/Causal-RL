# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
COPY vendor/wheels/ /wheels/

RUN --mount=type=cache,target=/root/.cache/pip \
    if find /wheels -type f \( -name "*.whl" -o -name "*.tar.gz" \) | grep -q .; then \
        python -m pip install --no-index --find-links=/wheels -r requirements.txt; \
    else \
        python -m pip install -r requirements.txt; \
    fi

COPY . .

CMD ["python", "-m", "compileall", "-q", "causal_rl"]
