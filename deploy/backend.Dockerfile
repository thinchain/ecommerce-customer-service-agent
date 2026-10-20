FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/backend:/app
WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt
COPY backend/constraints-py312.txt /app/backend/constraints-py312.txt
RUN pip install --no-cache-dir -c backend/constraints-py312.txt -r backend/requirements.txt && useradd --uid 10001 --create-home commerce
COPY backend /app/backend
COPY scripts /app/scripts
COPY fixtures /app/fixtures
RUN mkdir -p /app/data && chown -R commerce:commerce /app/data
USER commerce
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
