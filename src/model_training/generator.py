import random
import numpy as np
import keras
import h5py


class Generator(keras.utils.PyDataset):
    """
    Generates batches of data from an HDF5 file.

    Args:
        path (str): Path to the HDF5 file.
        batch_size (int): Size of each batch.

    Attributes:
        path (str): Path to the HDF5 file.
        batch_size (int): Size of each batch.
        data (h5py.File): The HDF5 file object.
        data_type_groups (list): List of data type groups in the HDF5 file.
        first_dataset_names (list): List of dataset names under the first data type group.
        num_datasets (int): Total number of datasets in the HDF5 file.
        random_indices (list): List of random indices used for shuffling the datasets.
    """

    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.data = h5py.File(self.path, "r")
        self.data_type_groups = list(self.data.keys())
        self.first_dataset_names = list(self.data[self.data_type_groups[0]].keys())
        self.num_datasets = len(self.first_dataset_names)
        self.random_indices = [i for i in range(self.num_datasets)]
        random.shuffle(self.random_indices)

    def __len__(self):
        """
        Returns the number of batches in the dataset.
        """
        return int(np.floor(self.num_datasets / self.batch_size))

    def __getitem__(self, idx):
        """
        Generates one batch of data.

        Args:
            idx (int): Index of the batch.

        Returns:
            tuple: A tuple containing batch data and batch labels.
        """
        batch_indices = self.random_indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_data = [[] for _ in self.data_type_groups]
        batch_labels = [[] for _ in self.data_type_groups]
        for i, data_type in enumerate(self.data_type_groups):
            dataset_names = list(self.data[data_type].keys())
            for j in batch_indices:
                batch_data[i].append(self.data[data_type][dataset_names[j]][:].flatten())
                batch_labels[i].append(self.data[data_type][dataset_names[j]].attrs["label"])
            batch_data[i] = np.array(batch_data[i])
            batch_labels[i] = np.array(batch_labels[i])
        return batch_data, batch_labels
