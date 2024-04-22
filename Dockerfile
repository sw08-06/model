FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

COPY requirements.txt requirements.txt
COPY src/model_training /app

RUN pip install -r requirements.txt

ENV TRAINING_DATA_PATH=/app/data/frames1/training.h5
ENV VALIDATION_DATA_PATH=/app/data/frames1/validation.h5
ENV MODEL_PATH=/app/models

CMD ["python3", "training.py"]
