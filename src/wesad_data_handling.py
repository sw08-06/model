import os
import pickle
import numpy as np


def create_labeled_frames(data_types, fs, window_size, overlap):
    """
    Creates maximum amount of labeled frames from the WESAD dataset in the segments of the data
    corresponding to label 2 (stressed) and labels 1, 3, and 4 (not stressed).
    Args: data_types: which data types to create frames from.
          fs: the sampling frequencies of the data types.
          window_size: the window size of the frames of each data type.
          overlap: the overlap of the windows in for each data type.
    """
    for subject in os.listdir("data/WESAD"):
        with open(f"data/WESAD/{subject}/{subject}.pkl", "rb") as file:
            data = pickle.load(file, encoding="latin1")
            print(f"Loaded pickle file of {subject}")

            label_indicies = find_label_indices(data["label"])

            for label_pair in label_indicies:
                label = 1 if label_pair == label_indicies[0] else 0
                for i, data_type in enumerate(data_types):
                    f = fs[i] / 700
                    start = np.floor(label_pair[1] * f)
                    end = np.floor(label_pair[0] * f)
                    w = window_size[i] * fs[i]
                    num_frames = 1 + np.floor((start - end - w) / (w - w * overlap[i]))

                    for i in range(int(num_frames)):
                        frame = data["signal"]["wrist"][data_type][
                            int(start) : int(end)
                        ][w * i : w * i + 1]

                        os.makedirs(
                            f"data/frames/{subject}/{data_type}/", exist_ok=True
                        )
                        np.save(
                            f"data/frames/{subject}/{data_type}/{label}_{data_type}_{i}.npy",
                            frame,
                        )
                        print(f"Created frame {label}_{data_type}_{i}.npy")


def find_label_indices(labels):
    """
    Find start and end indices of label 2 (stressed) as well as labels 1, 3, and 4 (not stressed)
    in label signal. Returns stand and end pairs in list with first pair being label 2 (stressed).
    """
    label_indices = []
    for value in [2, 1, 3, 4]:
        indices = np.where(labels == value)[0]
        first_index = indices[0]
        last_index = indices[-1]
        label_indices.append([first_index, last_index])
    print(f"Label indices: {label_indices}")
    return label_indices


if __name__ == "__main__":
    create_labeled_frames(
        data_types=["BVP", "EDA", "TEMP"],
        fs=[64, 4, 4],
        window_size=[60, 60, 60],
        overlap=[0.5, 0.5, 0.5],
    )
