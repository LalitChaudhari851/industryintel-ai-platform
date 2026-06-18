FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV HOME=/home/user

WORKDIR $HOME/app

RUN useradd -m -u 1000 user \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=user pyproject.toml ./
COPY --chown=user app ./app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir . \
    && apt-get purge -y --auto-remove build-essential

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

USER user

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
