FROM tensorflow/tensorflow:latest-gpu

WORKDIR /model_training

COPY requirements.txt requirements.txt
COPY src/model_training /model_training

RUN pip install -r requirements.txt
