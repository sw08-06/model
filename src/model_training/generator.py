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
        dataset_names (list): List of dataset names.
        num_datasets (int): Total number of datasets in the HDF5 file.
        random_indices (list): List of random indices used for shuffling the data.
    """

    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.data = h5py.File(self.path, "r")
        self.dataset_names = list(self.data.keys())
        self.num_datasets = len(self.dataset_names)
        self.random_indicies = [i for i in range(self.num_datasets)]
        random.shuffle(self.random_indicies)

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
        batch_random_indices = self.random_indicies[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_data = []
        batch_labels = []
        for i in batch_random_indices:
            batch_data.append(self.data[self.dataset_names[i]][:])
            batch_labels.append(self.data[self.dataset_names[i]].attrs["label"])
        return np.array(batch_data), np.array(batch_labels)[:, np.newaxis]


if __name__ == "__main__":
    gene = Generator("data/frames1/training.h5", 256)
    print(gene.__len__())
    print(gene.__getitem__(0)[0].shape)
    print(gene.__getitem__(0)[1].shape)
