from preprocessing_methods import butterworth_filter
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np


class FramePreprocessor:
    def __init__(self, data_path, data_types, functions_dict):
        """
        Initializes a FramePreprocessor instance.
        Args: data_path (str): The path to the directory containing the data.
              data_types (list): A list of strings representing the types of data to be processed.
              functions_dict (dict): A dictionary mapping data types to lists of preprocessing functions.
        """
        self.data_path = data_path
        self.data_types = data_types
        self.functions_dict = functions_dict

    def _preprocess_frames(self, subject):
        """
        Loads frames from the specified subject directory, preprocesses the frames, and saves them.
        Args: subject: Name of the subject.
        """
        for data_type in self.data_types:
            if data_type not in self.functions_dict:
                continue

            print(f"Loading {data_type} for {subject}")
            subject_signal_path = os.path.join(self.data_path, subject, data_type)
            output_dir = os.path.join("data", "preprocessed_frames", subject, data_type)
            os.makedirs(output_dir, exist_ok=True)

            for file_name in os.listdir(subject_signal_path):
                file_path = os.path.join(subject_signal_path, file_name)
                frame = np.load(file_path).flatten()
                for func in self.functions_dict[data_type]:
                    frame = func(frame)

                try:
                    np.save(os.path.join(output_dir, file_name.split(".")[0] + "_preprocessed.npy"), frame)
                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")

            print(f"Finished preprocessing {data_type} for {subject}")

    def process_all_subjects(self):
        """
        Utilizes multithreading to preprocess the frames for all subjects.
        """
        subjects = [subject for subject in os.listdir(self.data_path)]

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for subject in subjects:
                executor.submit(self._preprocess_frames, subject)


if __name__ == "__main__":
    preprocessor = FramePreprocessor(
        os.path.join("data", "frames"),
        ["BVP", "EDA", "TEMP"],
        {
            "BVP": [partial(butterworth_filter, cutoff_freq=4, sampling_freq=64, order=4)],
            "EDA": [],
            "TEMP": [],
        },
    )

    preprocessor.process_all_subjects()
