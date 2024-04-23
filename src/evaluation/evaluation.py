import os
import h5py
import numpy as np
import keras


def load_data(testing_data_path):
    with h5py.File(testing_data_path, "r") as file:
        dataset_names = list(file.keys())
        data = []
        labels = []
        for dataset in dataset_names:
            data.append(file[dataset][:])
            labels.append(file[dataset].attrs["label"])
    return np.array(data), np.array(labels)[:, np.newaxis]


def evaluate_model(data, labels, model_path):
    model = keras.models.load_model(model_path)
    preds = model.predict(x=data, verbose=2)
    return preds


if __name__ == "__main__":
    data, labels = load_data(testing_data_path=os.path.join("data", "frames1", "testing.h5"))
    preds = evaluate_model(data, labels, model_path=os.path.join("models", "model_v1.keras"))
    print(preds)
