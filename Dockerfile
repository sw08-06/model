FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

COPY requirements.txt requirements.txt
COPY src/model_training /app

RUN pip install -r requirements.txt

ENV DATA_PATH=/app/data
ENV MODEL_PATH=/app/models

CMD ["python3", "training.py"]
