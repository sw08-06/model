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
        batch_random_indices = self.random_indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_data = []
        batch_labels = []
        for i in batch_random_indices:
            data_arr = []
            for data_type in self.data_type_groups:
                dataset_names = list(self.data[data_type].keys())
                data = self.data[data_type][dataset_names[i]][:]
                data_arr.append(data)
            label = self.data[data_type][dataset_names[i]].attrs["label"]
            batch_labels.append(label)
            batch_data.append(np.concatenate(data_arr))
        return np.array(batch_data), np.array(batch_labels)[:, np.newaxis]


if __name__ == "__main__":
    gene = Generator("data/frames1/training.h5", 4)
    print(gene.__getitem__(0)[0].shape)
    print(gene.__getitem__(0)[1].shape)
    print(gene.__getitem__(0))
