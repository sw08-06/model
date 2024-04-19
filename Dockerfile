FROM tensorflow/tensorflow:latest-gpu

WORKDIR /model_training

COPY requirements.txt requirements.txt
COPY src/model_training /model_training
COPY data/frames1 model_training/data/frames1

RUN pip install -r requirements.txt
