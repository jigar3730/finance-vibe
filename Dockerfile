FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV TZ=America/New_York

WORKDIR /app

# System dependencies (Must remain here)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    tzdata \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "src/finance_vibe/app.py"]