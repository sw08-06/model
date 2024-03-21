import os
import pickle
import numpy as np


def create_labeled_frames(path, data_types, fs, window_size):
    """
    Creates maximum amount of labeled frames from the WESAD dataset in the segments of the data
    corresponding to stressed (label 2) and not stressed (labels 1, 3, and 4).
    Args: path: the path of the WESAD dataset.
          data_types: which data types to create frames from.
          fs: the sampling frequencies of the data types.
          window_size: the window size of the frames for each data type.
    """
    for subject in os.listdir(path):
        if not subject.endswith(".pdf"):
            with open(f"data/WESAD/{subject}/{subject}.pkl", "rb") as file:
                data = pickle.load(file, encoding="latin1")
                print(f"Loaded pickle file of {subject}")

                label_indicies = find_label_indices(data["label"])

                for label_pair in label_indicies:
                    label = 1 if label_pair == label_indicies[0] else 0
                    for j, data_type in enumerate(data_types):
                        f = fs[j] / 700
                        start = np.floor(label_pair[0] * f)
                        end = np.floor(label_pair[1] * f)
                        w = window_size[j] * fs[j]
                        num_frames = 1 + np.floor(end - start - w)

                        for j in range(int(num_frames)):
                            frame = data["signal"]["wrist"][data_type][
                                int(start) : int(end)
                            ][j : w + j]

                            os.makedirs(
                                f"data/frames/{subject}/{data_type}/", exist_ok=True
                            )
                            np.save(
                                f"data/frames/{subject}/{data_type}/{label}_{data_type}_{j}.npy",
                                frame,
                            )
                            print(f"Created frame {label}_{data_type}_{j}.npy")


def find_label_indices(labels):
    """
    Finds the start and end indices of label 2 (stressed) as well as
    labels 1, 3, and 4 (not stressed) in a label signal.
    Returns start and end pairs in a list with first pair being label 2 (stressed).
    """
    label_indices = []
    for value in [2, 1, 3, 4]:
        indices = np.where(labels == value)[0]
        first_index = indices[0]
        last_index = indices[-1]
        label_indices.append([first_index, last_index])
    print(
        f"Label indices - stressed: {label_indices[0]} - not stressed: {label_indices[1]}, {label_indices[2]}, {label_indices[3]}"
    )
    return label_indices


if __name__ == "__main__":
    create_labeled_frames(
        path="data/WESAD",
        data_types=["BVP", "EDA", "TEMP"],
        fs=[64, 4, 4],
        window_size=[60, 60, 60],
    )
