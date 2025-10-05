FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ /app/src/
COPY ARCHITECTURE.md /app/
ENV PYTHONPATH=/app
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
