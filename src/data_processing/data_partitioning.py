from concurrent.futures import ThreadPoolExecutor
import os
import pickle
import numpy as np
import h5py


class DataPartitioner:
    def __init__(self, data_path, data_types, fs, window_seconds, overlap_seconds, loso_subject, train_val_split):
        """
        Initializes a DataHandler instance.

        Args:
            data_path (str): The path of the WESAD dataset.
            data_types (list): List of data types to create frames from.
            fs (list): List of sampling frequencies of the data types.
            window_seconds (float): List of window sizes in seconds for each data type.
            overlap_seconds (float): The overlap in seconds for all data types.
            loso_subject (str): Name of the subject for leave-one-subject-out cross-validation.
            train_val_split (float): The ratio to split the data into training and validation sets.
        """
        self.data_path = data_path
        self.data_types = data_types
        self.fs = fs
        self.window_seconds = window_seconds
        self.overlap_seconds = overlap_seconds
        self.loso_subject = loso_subject
        self.train_val_split = train_val_split

    def process_all_subjects(self):
        """
        Utilizes multithreading to process all subjects from the WESAD dataset to create labeled frames.
        """
        subjects = [subject for subject in os.listdir(self.data_path) if not subject.endswith(".pdf")]

        frames_dir = os.path.join("data", f"frames_{self.window_seconds}s_{self.loso_subject}")
        os.makedirs(frames_dir, exist_ok=True)
        h5_file_names = [
            os.path.join(frames_dir, "training.h5"),
            os.path.join(frames_dir, "validation.h5"),
            os.path.join(frames_dir, "testing.h5"),
        ]

        with ThreadPoolExecutor(max_workers=15) as executor:
            for subject in subjects:
                executor.submit(self._create_labeled_frames, subject, h5_file_names)

    def _create_labeled_frames(self, subject, h5_file_names):
        """
        Creates labeled frames from the WESAD dataset for a specific subject in the segments of the data
        corresponding to stressed (label 2) and not stressed (labels 1, 3, and 4).

        Args:
            subject (str): The name of the subject.
            h5_file_names (list): List of HDF5 file names.
        """
        try:
            with open(os.path.join(self.data_path, subject, f"{subject}.pkl"), "rb") as file:
                wesad_data = pickle.load(file, encoding="latin1")
                label_indices = self._find_label_indices(wesad_data["label"])
                print(f"Loaded subject: {subject} - label_indices: {label_indices}")
                index = 0

            with h5py.File(h5_file_names[0], "a") as h5_training, h5py.File(h5_file_names[1], "a") as h5_validation, h5py.File(h5_file_names[2], "a") as h5_testing:
                for label_pair in label_indices:
                    label = 1 if label_pair == label_indices[0] else 0

                    num_frames, _, _, _, _ = self._calculate_num_frames(self.fs[0], self.window_seconds, label_pair)
                    split_vec = self._create_split_vec(num_frames)

                    for i in range(num_frames):
                        data_arr = []
                        for j, data_type in enumerate(self.data_types):
                            _, start, end, sample_skip, window_samples = self._calculate_num_frames(self.fs[j], self.window_seconds, label_pair)
                            data_arr.append(wesad_data["signal"]["wrist"][data_type][int(start) : int(end)][int(i * sample_skip) : int(window_samples + i * sample_skip)])

                        dataset_name = f"{subject}_frame_{index + i}"
                        dataset = np.concatenate(data_arr)
                        if self.loso_subject == subject:
                            h5_testing.create_dataset(dataset_name, data=dataset)
                            h5_testing[dataset_name].attrs["label"] = label
                        elif split_vec[i]:
                            h5_training.create_dataset(dataset_name, data=dataset)
                            h5_training[dataset_name].attrs["label"] = label
                        else:
                            h5_validation.create_dataset(dataset_name, data=dataset)
                            h5_validation[dataset_name].attrs["label"] = label

                    index += num_frames

            h5_training.close()
            h5_validation.close()
            h5_testing.close()
            print(f"Finished creating frames for {subject}")

        except Exception as e:
            print(f"Error occurred while processing subject {subject}: {e}")

    def _find_label_indices(self, label_signal):
        """
        Finds the start and end indices of label 2 (stressed) as well as labels 1, 3, and 4 (not stressed) in a label signal.

        Args:
            label_signal (numpy.ndarray): An array containing the label signal.

        Returns:
            list: Start and end pairs in a list with the first pair being for label 2 (stressed).
        """
        label_indices = []
        for value in [2, 1, 3, 4]:
            indices = np.where(label_signal == value)[0]
            first_index = indices[0]
            last_index = indices[-1]
            label_indices.append([first_index, last_index])
        return label_indices

    def _calculate_num_frames(self, fs, window_seconds, label_pair):
        """
        Calculates the number of frames that can be created from a data signal.

        Args:
            fs (int): The sampling frequency of the data signal.
            window_seconds (float): The length of each frame in seconds.
            label_pair (tuple): A tuple containing the start and end indices of the label interval.

        Returns:
            tuple: A tuple containing the following elements:
                - num_frames (int): The number of frames that can be created.
                - start (float): The start index of the first frame.
                - end (float): The end index of the last frame.
                - sample_skip (float): The number of samples to skip between frames.
                - window_samples (float): The number of samples in each frame.
        """
        freq_ratio = fs / 700
        start = np.floor(label_pair[0] * freq_ratio)
        end = np.floor(label_pair[1] * freq_ratio)
        window_samples = window_seconds * fs
        overlap_samples = self.overlap_seconds * fs
        sample_skip = window_samples - overlap_samples
        num_frames = 1 + np.floor((end - start - window_samples) / sample_skip)
        return int(num_frames), start, end, sample_skip, window_samples

    def _create_split_vec(self, num_frames):
        """
        Creates a vector for splitting data into training and validation data.

        Args:
            num_frames (int): The number of frames in a label pair.

        Returns:
            numpy.ndarray: A binary vector with the same length as the number of frames,
            where 1's represent frames for training and validation, and 0's represent frames for testing.
        """
        num_ones = int(np.ceil(num_frames * self.train_val_split))
        num_zeros = int(np.floor(num_frames - num_ones))
        split_vec = np.concatenate((np.ones(num_ones, dtype=int), np.zeros(num_zeros, dtype=int)))
        np.random.shuffle(split_vec)
        return split_vec


if __name__ == "__main__":
    windows = [5, 15, 30, 60, 90, 120]
    for window in windows:
        dataPartitioner = DataPartitioner(
            data_path=os.path.join("data", "WESAD_hampel120s_nsigma3"),
            data_types=["BVP", "EDA", "TEMP"],
            fs=[64, 4, 4],
            window_seconds=window,
            overlap_seconds=window-0.25,
            loso_subject="S2",
            train_val_split=0.7,
        )
        dataPartitioner.process_all_subjects()
