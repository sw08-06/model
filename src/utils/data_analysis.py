import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt


def load_file(file_path):
    """
    Load data from a file.
    """
    data = np.load(file_path).flatten()

    x = np.linspace(0.0, len(data) * 1 / 64, len(data), endpoint=False)
    y = 0.2 * np.sin(10.0 * 2.0 * np.pi * x) + 0.2 * np.sin(20.0 * 2.0 * np.pi * x)
    z = data + y

    return data


def butterworth_filter(data, cutoff_freq, sampling_freq, order):
    """
    Apply a Butterworth filter to the data.
    """
    nyquist = 0.5 * sampling_freq
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data


def fourier_analysis(data, sampling_freq):
    """
    Perform Fourier analysis on a frame.
    """
    n = len(data)
    freq = fftfreq(n, 1 / sampling_freq)[: n // 2]
    fft_data = np.abs(fft(data)[0 : n // 2]) * 2 / n
    return freq, fft_data


def plot_data(original_data, filtered_data, freq, fft_data, freq2, fft_data2):
    """
    Plot the original and filtered data.
    """
    plt.figure(figsize=(16, 10))

    # Plot original data
    plt.subplot(4, 1, 1)
    plt.plot(original_data, label="Original Data")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Original Data")
    plt.legend()

    # Plot filtered data
    plt.subplot(4, 1, 2)
    plt.plot(filtered_data, label="Filtered Data")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Filtered Data")
    plt.legend()

    # Plot Fourier analysis
    plt.subplot(4, 1, 3)
    plt.plot(freq, fft_data)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Fourier Analysis")
    plt.tight_layout()

    # Plot Fourier analysis
    plt.subplot(4, 1, 4)
    plt.plot(freq2, fft_data2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Fourier Analysis of Filtered Data")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    data = load_file("data/frames/S2/BVP/1_BVP_10.npy")
    freq, fft_data = fourier_analysis(data, sampling_freq=64)
    filtered_data = butterworth_filter(data, cutoff_freq=4, sampling_freq=64, order=4)
    freq2, fft_data2 = fourier_analysis(filtered_data, sampling_freq=64)
    plot_data(data, filtered_data, freq, fft_data, freq2, fft_data2)
