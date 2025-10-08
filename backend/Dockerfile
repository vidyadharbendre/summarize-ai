FROM python:3.11.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY app/ ./app/

ENV PYTHONPATH="/app"

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]