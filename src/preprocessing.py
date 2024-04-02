import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class FramePreprocessor:
    def __init__(self, data_path, data_types):
        """
        Initializes a FramePreprocessor instance.
        Args: data_path (str): The path to the directory containing the data.
              data_types (list): A list of strings representing the types of data to be processed.
        """
        self.data_path = data_path
        self.data_types = data_types

    def preprocess_frames(self, subject):
        """
        Loads frames from the specified subject directory, preprocesses the frames, and saves them.
        Args: subject: Name of the subject.
        """
        for data_type in self.data_types:
            print("loading datatype: " + data_type)
            subject_signal_path = os.path.join(self.data_path, subject, data_type)
            output_dir = os.path.join("data", "preprocessed_frames", subject, data_type)
            os.makedirs(output_dir, exist_ok=True)
            
            for file_name in os.listdir(subject_signal_path):
                file_path = os.path.join(subject_signal_path, file_name)
                frame = np.load(file_path)
                frame = self._detect_outliers(frame)
                frame = self._normalize_frame(frame)
                np.save(os.path.join(output_dir, file_name.split(".")[0], "_preprocessed.npy"), frame)
    
    def _detect_outliers(self, frame):
        pass
        
    def _normalize_frame(self, frame):
        pass

    def process_all_subjects(self):
        """
        Utilizes multithreading to preprocess the frames for all subjects.
        """
        subjects = [subject for subject in os.listdir(self.data_path)]

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for subject in subjects:
                executor.submit(self.preprocess_frames, subject)


if __name__ == "__main__":
    preprocessor = FramePreprocessor(
        data_path=os.path.join("data", "frames"), 
        data_types=["BVP", "EDA", "TEMP"],
    )

    os.makedirs(os.path.join("data", "preprocessed_frames"), exist_ok=True)

    preprocessor.process_all_subjects()