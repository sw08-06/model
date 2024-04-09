from concurrent.futures import ThreadPoolExecutor
import os
import pickle
import numpy as np


class DataHandler:
    def __init__(self, path, data_types, fs, window_seconds, overlap_seconds, loso_subject, train_val_split):
        """
        Initializes a DataHandler instance.
        Args: path: the path of the WESAD dataset.
              data_types: which data types to create frames from.
              fs: the sampling frequencies of the data types.
              window_seconds: the window size in seconds for each data type.
              overlap_seconds: the overlap in seconds for all data types.
        """
        self.path = path
        self.data_types = data_types
        self.fs = fs
        self.window_seconds = window_seconds
        self.overlap_seconds = overlap_seconds
        self.loso_subject = loso_subject
        self.train_val_split = train_val_split

    def process_all_subjects(self):
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
        Args: subject: Name of the subject.
        """
        with open(os.path.join(self.path, subject, f"{subject}.pkl"), "rb") as file:
            data = pickle.load(file, encoding="latin1")
            print(f"Loaded pickle file for subject: {subject}")

            label_indices = self._find_label_indices(data["label"])

            if self.loso_subject == subject:
                os.makedirs(os.path.join("data", "frames", "testing", subject), exist_ok=True)
            else:
                os.makedirs(os.path.join("data", "frames", "training", subject), exist_ok=True)
                os.makedirs(os.path.join("data", "frames", "validation", subject), exist_ok=True)

            for label_pair in label_indices:
                label = 1 if label_pair == label_indices[0] else 0

                freq_ratio = self.fs[0] / 700
                start = np.floor(label_pair[0] * freq_ratio)
                end = np.floor(label_pair[1] * freq_ratio)
                window_samples = self.window_seconds[0] * self.fs[0]
                overlap_samples = self.overlap_seconds * self.fs[0]
                sample_skip = window_samples - overlap_samples
                num_frames = 1 + np.floor((end - start - window_samples) / sample_skip)

                num_ones = num_frames * self.train_val_split
                num_zeros = num_frames - num_ones
                split_vec = np.concatenate((np.ones(num_ones, dtype=int), np.zeros(num_zeros, dtype=int)))
                np.random.shuffle(split_vec)

                print(num_frames)
                print(np.sum(split_vec))

                for j in range(int(num_frames)):
                    frame_vecs = [[] for _ in range(len(self.data_types))]
                    print(frame_vecs)
                    for k, data_type in enumerate(self.data_types):
                        window_samples = self.window_seconds[k] * self.fs[k]
                        overlap_samples = self.overlap_seconds * self.fs[k]
                        sample_skip = window_samples - overlap_samples

                        frame_vecs[k].append(data["signal"]["wrist"][data_type][int(start) : int(end)][int(j * sample_skip) : int(window_samples + j * sample_skip)])
                        print(frame_vecs.shape)

                    frame_data = np.array(frame_vecs)

                    try:
                        if self.loso_subject == subject:
                            np.save(
                                os.path.join("data", "frames", "testing", subject, f"{label}_{j}.npy"),
                                frame_data,
                            )
                        else:
                            if split_vec[j]:
                                np.save(
                                    os.path.join("data", "frames", "training", subject, f"{label}_{j}.npy"),
                                    frame_data,
                                )
                            else:
                                np.save(
                                    os.path.join("data", "frames", "validation", subject, f"{label}_{j}.npy"),
                                    frame_data,
                                )
                    except Exception as e:
                        print(f"Error processing file {label}_{j}.npy: {e}")

                print(f"Finished creating frames for {subject}")

    def _find_label_indices(self, labels):
        """
        Finds the start and end indices of label 2 (stressed) as well as labels 1, 3, and 4 (not stressed) in a label signal.
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
        loso_subject="S7",
        train_val_split=0.7,
    )

    dataHandler.process_all_subjects()
