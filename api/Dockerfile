FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY static ./static

# Place your trained model here before building:
#   cp -r /path/to/output_stage2/model-best ./model
COPY model ./model

ENV MODEL_PATH=/app/model
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
