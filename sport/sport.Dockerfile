FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY new_sportsGoodsDescription.csv .
COPY main.py .

ENTRYPOINT ["python", "main.py"]
