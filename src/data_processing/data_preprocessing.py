import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from scipy.signal import butter, lfilter
from scipy.stats import iqr
import numpy as np


def butterworth_filter(data, cutoff_freq, sampling_freq, order):
    """
    Apply a Butterworth filter to the data.
    """
    nyquist = 0.5 * sampling_freq
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data


def remove_outliers_iqr(data):
    """
    Detects outliers using IQR and removes them.
    Args: data: Numpy array
    """
    iqr_data = iqr(data)

    lower_bound = np.percentile(data, 25) - 1.5 * iqr_data
    upper_bound = np.percentile(data, 75) + 1.5 * iqr_data

    cleaned_data = data[(data >= lower_bound) & (data <= upper_bound)]

    return cleaned_data


def replace_outliers_iqr(data, range_lower=25, range_upper=75):
    """
    Detects outliers using IQR and replaces them with closest valid neighbors.
    Modifies the data object instead of creating copy - should be used as void function.
    Args: data: The numpy array to clean
    """
    data_iqr = iqr(data, rng=(range_lower, range_upper))
    print("iqr: ", data_iqr)
    print("median: ", np.median(data))

    lower_bound = np.percentile(data, range_lower) - 1.5 * data_iqr
    upper_bound = np.percentile(data, range_upper) + 1.5 * data_iqr

    print("Lower bound: ", lower_bound)
    print("Upper bound: ", upper_bound)

    for i, value in enumerate(data):
        if value < lower_bound or value > upper_bound:
            print("outlier detected: ", i)
            distance = 1
            while True:
                if i - distance >= 0 and data[i - distance] >= lower_bound and data[i - distance] <= upper_bound:
                    value = data[i - distance]
                    break
                elif i + distance < len(data) and data[i + distance] >= lower_bound and data[i + distance] <= upper_bound:
                    value = data[i + distance]
                    break
                else:
                    distance += 1
    return data


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
