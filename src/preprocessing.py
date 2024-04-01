import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor

class FramePreprocessor:
    def __init__(self, data_path, data_types, output_path, max_workers=None):
        self.data_path = data_path
        self.data_types = data_types
        self.output_path = output_path
        self.max_workers = max_workers

    def load_frames(self, subject):
        frames = {"BVP": [], "EDA": [], "TEMP": []}
        subject_path = os.path.join(self.data_path, subject) # data\frames\subject_num        
        
        for data_type, _ in frames.items():
            print("starting on data type: " + data_type)
            subject_signal_path = os.path.join(subject_path, data_type)
            print("subject datatype path: " + subject_signal_path)
            for file_name in os.listdir(subject_signal_path):                 
                if file_name.endswith(".npy"):
                    file_path = os.path.join(subject_signal_path, file_name)
                    print("File path: " + file_path)
                    with open(file_path, "rb") as file:
                        frames[data_type].append(np.load(file))
                
        return frames
    
    def detect_outliers():
        pass

    def normalize_frames():
        pass

    def preprocess_frames(self, subject):
        frames = self.load_frames(subject)
        print("All frames loaded into memory")
        
        # outlier detection
        # normalize
        
        return frames

    def preprocess_subject(self, subject):
        frames = self.preprocess_frames(subject)
        
        output_subject_dir = os.path.join("data", self.output_path, subject) # data\preprocessed_frames\subject_num
        os.makedirs(output_subject_dir, exist_ok=True)
        
        
        
        for data_type, frames_list in frames.items():
            output_subject_signal_dir = os.path.join(output_subject_dir, data_type)
            print("loading subject signal: " + output_subject_signal_dir)
            os.makedirs(output_subject_signal_dir, exist_ok=True)
            for i, frame in enumerate(frames_list):
                output_file_path = os.path.join(output_subject_signal_dir, f"{data_type}_{i}.npy")   # TODO: Add non-stress and stress labels to file name.
                np.save(output_file_path, frame)


    def preprocess_all_subjects(self):
        subjects = [subject for subject in os.listdir(self.data_path)]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            #for subject in subjects:
                executor.submit(self.preprocess_subject, subjects[1])
                print("Subject" + subjects[1] + " thread started.")



if __name__ == "__main__":
    preprocessor = FramePreprocessor(
        data_path=os.path.join("data", "frames"), 
        data_types=["BVP", "EDA", "TEMP"],
        output_path="preprocessed_frames", 
        max_workers=os.cpu_count())

    os.makedirs(os.path.join("data", "preprocessed_frames"), exist_ok=True)

    preprocessor.preprocess_all_subjects()