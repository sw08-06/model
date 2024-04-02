import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class FramePreprocessor:
    def __init__(self, data_path, data_types, output_path, max_workers=None):
        """
        Initializes a FramePreprocessor instance.
        Args: data_path (str): The path to the directory containing the data.
              data_types (list): A list of strings representing the types of data to be processed.
              output_path (str): The path to the directory where preprocessed data will be saved.
              max_workers (int, optional): The maximum number of worker threads for concurrent processing.
        """
        self.data_path = data_path
        self.data_types = data_types
        self.output_path = output_path
        self.max_workers = max_workers

    def load_frames(self, subject):
        """
        Loads frames from the specified subject directory.
        Args: subject: Name of the subject.
        Returns: dict - A dictionary containing loaded frames, organized by label and index.
        """
        frames = {}
        subject_path = os.path.join(self.data_path, subject)

        for data_type in self.data_types:
            print("loading datatype: " + data_type)
            subject_signal_path = os.path.join(subject_path, data_type)
            for file_name in os.listdir(subject_signal_path):
                if file_name.endswith(".npy"):
                    label = int(file_name.split("_")[0])
                    index = int(file_name.split("_")[-1].split(".")[0])
                    file_path = os.path.join(subject_signal_path, file_name)
                    if (label, index) not in frames:
                        frames[(label, index)] = {}
                    frames[(label, index)][data_type] = np.load(file_path)
        
        return frames
    
    def detect_outliers(self, feature_vector):
        pass
        
    def normalize_feature_vector(self, feature_vector):
        pass
    
    def process_subject(self, subject):
        """
        Preprocesses frames for a specific subject and saves the feature vectors.
        Args: subject: Name of the subject.
        """
        frames = self.load_frames(subject)
        
        print("all frames loaded.")
        output_subject_dir = os.path.join("data", self.output_path, subject)
        os.makedirs(output_subject_dir, exist_ok=True)
        
        for (label, index), data in frames.items():
            feature_vector = np.array([(data[data_type]) for data_type in self.data_types])
            # TODO: uncomment detect_outliers and fill out
            #feature_vector = self.detect_outliers(feature_vector)
            # TODO: 
            #feature_vector = self.normalize_feature_vector(feature_vector)
            
            output_file_path = os.path.join(output_subject_dir, f"{label}_FV_{index}.npy")
            np.save(output_file_path, feature_vector)

    def process_all_subjects(self):
        """
        Utilizes multithreading to preprocess the frames for all subjects.
        """
        subjects = [subject for subject in os.listdir(self.data_path)]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for subject in subjects:
                executor.submit(self.process_subject, subject)


if __name__ == "__main__":
    preprocessor = FramePreprocessor(
        data_path=os.path.join("data", "frames"), 
        data_types=["BVP", "EDA", "TEMP"],
        output_path="preprocessed_frames", 
        max_workers=os.cpu_count())

    os.makedirs(os.path.join("data", "preprocessed_frames"), exist_ok=True)

    preprocessor.process_all_subjects()