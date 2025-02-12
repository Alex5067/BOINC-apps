FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY new_sportsGoodsDescription.csv .
COPY main.py .

ENTRYPOINT ["python", "main.py"]
