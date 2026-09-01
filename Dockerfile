FROM python:3.12-slim-bookworm

ARG GEOCHEMISTRYPI_SOURCE_REVISION=unknown

LABEL org.opencontainers.image.title="Geochemistry Pi Online API" \
      org.opencontainers.image.source="https://github.com/ZJUEarthData/Geochemistrypi" \
      org.opencontainers.image.revision="${GEOCHEMISTRYPI_SOURCE_REVISION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GEOCHEMISTRYPI_RUNTIME_DIR=/app/runtime \
    TMPDIR=/app/runtime/tmp

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 geochemistrypi \
    && useradd --system --uid 10001 --gid geochemistrypi --home-dir /app geochemistrypi

COPY requirements-online.txt ./
RUN python -m pip install --no-cache-dir --requirement requirements-online.txt

COPY --chown=geochemistrypi:geochemistrypi geochemistrypi ./geochemistrypi
RUN mkdir -p /app/runtime/tmp \
    && chown -R geochemistrypi:geochemistrypi /app/runtime

USER geochemistrypi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"]

CMD ["python", "-m", "uvicorn", "geochemistrypi.online.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips=*"]
