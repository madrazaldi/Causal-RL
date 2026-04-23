FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        make \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir pytest==9.0.3

COPY pyproject.toml Makefile README.md ./
COPY causal_rl/ ./causal_rl/
COPY tests/ ./tests/
COPY causalog_synthetic_urban_logistics.csv ./
COPY causalog_full_data_dictionary_v2.json ./

RUN pip install --no-deps -e .

CMD ["make", "test"]
