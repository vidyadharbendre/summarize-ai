FROM python:3.11.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY app/ ./app/

CMD ["python", "app/interface.py"]
