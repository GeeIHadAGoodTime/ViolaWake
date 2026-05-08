FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi==0.115.6 \
    "uvicorn[standard]==0.34.0" \
    numpy==2.2.6 \
    scipy==1.15.3 \
    soundfile==0.13.1

RUN addgroup --system decoder && adduser --system --ingroup decoder decoder

COPY console/decoder/app.py /app/app.py

USER decoder
EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
