from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
import pickle
import numpy as np
from preprocessing_methods import butterworth_filter


class DataPreprocessor:
    def __init__(self, data_path, data_types, fs, functions_dict):
        """
        Initializes a DataPreprocessor instance.

        Args:
            data_path (str): The path to the directory containing the data.
            data_types (list): A list of strings representing the types of data to be processed.
            fs (list): A list of sampling frequencies corresponding to each data type.
            functions_dict (dict): A dictionary mapping data types to lists of preprocessing functions.
        """
        self.data_path = data_path
        self.data_types = data_types
        self.fs = fs
        self.functions_dict = functions_dict
        self.dir_number = 1

    def process_all_subjects(self):
        """
        Utilizes multithreading to process all subjects from the WESAD dataset to create a preprocessed dataset.
        """
        subjects = [subject for subject in os.listdir(self.data_path) if not subject.endswith(".pdf")]

        while True:
            dir_name = os.path.join("data", f"WESAD_preprocessed{self.dir_number}")
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                break
            else:
                self.dir_number += 1

        with ThreadPoolExecutor(max_workers=15) as executor:
            for subject in subjects:
                os.makedirs(os.path.join(dir_name, subject))
                executor.submit(self._preprocess_data, subject)

    def _preprocess_data(self, subject):
        """
        Preprocesses data for a specific subject and saves it.

        Args:
            subject (str): Name of the subject.
        """
        try:
            with open(os.path.join(self.data_path, subject, f"{subject}.pkl"), "rb") as file:
                wesad_data = pickle.load(file, encoding="latin1")
                print(f"Loaded data for subject: {subject}")

                for data_type in self.data_types:
                    preprocessing_funcs = self.functions_dict.get(data_type)
                    if preprocessing_funcs:
                        for preprocessing_func in preprocessing_funcs:
                            wesad_data["signal"]["wrist"][data_type] = preprocessing_func(np.array(wesad_data["signal"]["wrist"][data_type]))

                with open(os.path.join("data", f"WESAD_preprocessed{self.dir_number}", subject, f"{subject}.pkl"), "wb") as file:
                    pickle.dump(wesad_data, file)
                    print(f"Saved preprocessed data for subject: {subject}")

        except Exception as e:
            print(f"Error occurred while preprocessing subject {subject}: {e}")


if __name__ == "__main__":
    dataPreprocessor = DataPreprocessor(
        data_path=os.path.join("data", "WESAD"),
        data_types=["BVP", "EDA", "TEMP"],
        fs=[64, 4, 4],
        functions_dict={"BVP": [partial(butterworth_filter, cutoff_freq=4, sampling_freq=64, order=4)], "EDA": [], "TEMP": []},
    )

    dataPreprocessor.process_all_subjects()
