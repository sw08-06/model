import os
import sys
import pickle
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import h5py
sys.path.append("src")
from data_processing.preprocessing_methods import * 


def load_HDF5(file_path, dataset_index):
    """
    Loads data from a specific dataset within a group in an HDF5 file.

    Args:
        file_path (str): The path to the HDF5 file.
        dataset_index (int): The index of the dataset within the group.

    Returns:
        numpy.ndarray: The loaded data.
    """
    with h5py.File(file_path, "r") as file:
        dataset_name = list(file.keys())[dataset_index]
        data = file[dataset_name][:].flatten()

    return data


def load_pkl(file_path, data_type):
    """
    Loads data of a specific data type from a specific subject.

    Args:
        file_path (str): The path to the HDF5 file.
        data_type (str): The type of data.

    Returns:
        numpy.ndarray: The loaded data.
    """
    with open(file_path, "rb") as file:
        dataset = pickle.load(file, encoding="latin1")
        data = dataset["signal"]["wrist"][data_type]

    return data


def fourier_analysis(data, sampling_freq):
    """
    Performs Fourier analysis on a frame.

    Args:
        data (numpy.ndarray): The input data.
        sampling_freq (float): The sampling frequency of the data.

    Returns:
        tuple: A tuple containing the frequency values and the Fourier transformed data.
    """
    n = len(data)
    freq = fftfreq(n, 1 / sampling_freq)[: n // 2]
    fft_data = np.abs(fft(data)[0 : n // 2]) * 2 / n
    return freq, fft_data


def plot_data(original_data, filtered_data, freq, fft_data, freq2, fft_data2):
    """
    Plots the original and filtered data.

    Args:
        original_data (numpy.ndarray): The original data.
        filtered_data (numpy.ndarray): The filtered data.
        freq (numpy.ndarray): The frequency values for the Fourier analysis of the original data.
        fft_data (numpy.ndarray): The Fourier transformed data of the original data.
        freq2 (numpy.ndarray): The frequency values for the Fourier analysis of the filtered data.
        fft_data2 (numpy.ndarray): The Fourier transformed data of the filtered data.
    """
    plt.figure(figsize=(16, 10))

    plt.subplot(4, 1, 1)
    plt.plot(original_data, label="Original Data")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Original Data")
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(filtered_data, label="Filtered Data")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Filtered Data")
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(freq, fft_data)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Fourier Analysis")
    plt.tight_layout()

    plt.subplot(4, 1, 4)
    plt.plot(freq2, fft_data2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Fourier Analysis of Filtered Data")
    plt.tight_layout()

    plt.show()


def visualize_frames(frames):
    """
    Visualizes a list of frames.

    Args:
        frames (list of numpy.ndarray): A list containing the frames to be visualized.

    Each frame is plotted as a separate subplot in a single figure. The x-axis represents
    the sample index, and the y-axis represents the value of each sample in the frame.

    """
    _, axes = plt.subplots(nrows=len(frames), ncols=1, figsize=(12, 8))

    for i, frame in enumerate(frames):
        ax = axes[i]
        ax.plot(frame)
        ax.set_title(f"Frame {i+1}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.set_xlim(0, len(frame))
        ax.set_ylim(np.min(frame), np.max(frame))


    plt.tight_layout()
    plt.show()

def visualize_frame_paper(frame):
    """
    Visualizes a list of frames.

    Args:
        frames (list of numpy.ndarray): A list containing the frames to be visualized.

    Each frame is plotted as a separate subplot in a single figure. The x-axis represents
    the sample index, and the y-axis represents the value of each sample in the frame.

    """
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(frame)
    ax.set_ylabel("Amplitude", fontsize = 18)
    ax.set_xlabel("Time [s]", fontsize = 18)
    ax.set_xlim(0, len(frame))
    ax.set_ylim(np.min(frame) - 5, np.max(frame) + 5)
    ax.set_xticks([0, 160])  # Set x-axis ticks at 0 and 70
    ax.set_xticklabels(["0", "2.5"], fontsize = 14)  # Set labels for the ticks
    ax.set_yticks([0])
    ax.set_yticklabels(["0"], fontsize = 14)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pre_data = load_pkl("data/WESAD_preprocessed1/S2/S2.pkl", "EDA")
    data = load_pkl("data/WESAD/S2/S2.pkl", "EDA")

    visualize_frames([pre_data, data])


