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
        self.dataset_names = list(self.data.keys())
        self.num_datasets = len(self.dataset_names)

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
        batch_dataset_name = self.dataset_names[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_data = []
        batch_labels = []
        for name in batch_dataset_name:
            batch_data.append(self.data[name][:])
            label = self.data[name].attrs["label"]
            batch_labels.append(label)
        return np.array(batch_data), np.array(batch_labels)[:, np.newaxis]


if __name__ == "__main__":
    gene = Generator("data/frames1/training.h5", 256)
    print(gene.__len__())
    print(gene.__getitem__(0)[0].shape)
    print(gene.__getitem__(0)[1].shape)
