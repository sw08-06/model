import os
import numpy as np
import keras


class Generator(keras.utils.PyDataset):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.subjects = [subject for subject in os.listdir(self.path)]
        files = []
        for subject in self.subjects:
            files += [os.path.join(self.path, subject, file) for file in os.listdir(os.path.join(self.path, subject))]
        self.files = files
        self.num_files = len(self.files)

    def __len__(self):
        return int(np.floor(self.num_files / self.batch_size))

    def __getitem__(self, idx):
        batch_files = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = []
        batch_labels = []
        for file in batch_files:
            file_data = np.load(file)
            batch_data.append(file_data)
            label = int(os.path.basename(file).split("_")[0])
            batch_labels.append(label)
        batch_data = np.array(batch_data)
        batch_labels = np.array(batch_labels)
        return batch_data, batch_labels
