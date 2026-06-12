FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pandas \
    pandas_ta \
    yfinance \
    yahooquery \
    flask

COPY . .

EXPOSE 5000

CMD ["python", "src/finance_vibe/app.py"]
