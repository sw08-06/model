from concurrent.futures import ThreadPoolExecutor
import os
import pickle
import numpy as np


class DataHandler:
    def __init__(self, path, data_types, fs, window_seconds, overlap_seconds):
        """
        Initializes a DataHandler instance.
        Args: path: the path of the WESAD dataset.
              data_types: which data types to create frames from.
              fs: the sampling frequencies of the data types.
              window_seconds: the window size in seconds for each data type.
        """
        self.path = path
        self.data_types = data_types
        self.fs = fs
        self.window_seconds = window_seconds
        self.overlap_seconds = overlap_seconds

    def process_data(self):
        """
        Utilizes multithreading to process the WESAD data to create labeled frames.
        """
        subjects = [subject for subject in os.listdir(self.path) if not subject.endswith(".pdf")]

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for subject in subjects:
                executor.submit(self._create_labeled_frames, subject)

    def _create_labeled_frames(self, subject):
        """
        Creates labeled frames from the WESAD dataset for a specific subject in the segments of the data
        corresponding to stressed (label 2) and not stressed (labels 1, 3, and 4).
        Args:
            subject (str): Name of the subject.
        """
        if not subject.endswith(".pdf"):
            with open(os.path.join(self.path, subject, f"{subject}.pkl"), "rb") as file:
                data = pickle.load(file, encoding="latin1")
                print(f"Loaded pickle file of {subject}")

                label_indices = self._find_label_indices(data["label"])

                for label_pair in label_indices:
                    label = 1 if label_pair == label_indices[0] else 0
                    for j, data_type in enumerate(self.data_types):
                        freq_ratio = self.fs[j] / 700
                        start = np.floor(label_pair[0] * freq_ratio)
                        end = np.floor(label_pair[1] * freq_ratio)
                        window_samples = self.window_seconds[j] * self.fs[j]
                        overlap_samples = self.overlap_seconds * self.fs[j]
                        sample_skip = window_samples - overlap_samples
                        num_frames = 1 + np.floor((end - start - window_samples) / sample_skip)

                        for j in range(int(num_frames)):
                            frame = data["signal"]["wrist"][data_type][int(start) : int(end)][int(j * sample_skip) : int(window_samples + j * sample_skip)]

                            os.makedirs(os.path.join("data", "frames", subject, data_type), exist_ok=True)
                            np.save(
                                os.path.join("data", "frames", subject, data_type, f"{label}_{data_type}_{j}.npy"),
                                frame,
                            )
                            print(f"Created frame {label}_{data_type}_{j}.npy for {subject}")

    def _find_label_indices(self, labels):
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
        print(f"Label indices - stressed: {label_indices[0]} - not stressed: {label_indices[1]}, {label_indices[2]}, {label_indices[3]}")
        return label_indices


if __name__ == "__main__":
    dataHandler = DataHandler(
        path=os.path.join("data", "WESAD"),
        data_types=["BVP", "EDA", "TEMP"],
        fs=[64, 4, 4],
        window_seconds=[60, 60, 60],
        overlap_seconds=59.75,
    )

    dataHandler.process_data()
