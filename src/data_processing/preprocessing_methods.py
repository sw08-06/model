from scipy.signal import butter, lfilter
from scipy.stats import iqr
import numpy as np


def butterworth_filter(data, cutoff_freq, sampling_freq, order):
    """
    Applies a Butterworth filter to the data.

    Args:
        data (numpy.ndarray): The input data.
        cutoff_freq (float): The cutoff frequency of the filter.
        sampling_freq (float): The sampling frequency of the data.
        order (int): The order of the Butterworth filter.

    Returns:
        numpy.ndarray: The filtered data.
    """
    nyquist = 0.5 * sampling_freq
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data


def remove_outliers_iqr(data):
    """
    Detects outliers using IQR and removes them.

    Args:
        data: Numpy array
    """
    iqr_data = iqr(data)

    lower_bound = np.percentile(data, 25) - 1.5 * iqr_data
    upper_bound = np.percentile(data, 75) + 1.5 * iqr_data

    cleaned_data = data[(data >= lower_bound) & (data <= upper_bound)]

    return cleaned_data


def replace_outliers_iqr(data, range_lower=25, range_upper=75):
    """
    Detects outliers using IQR and replaces them with closest valid neighbors.
    Modifies the data object instead of creating copy - should be used as void function.

    Args:
        data: The numpy array to clean
    """
    data_iqr = iqr(data, rng=(range_lower, range_upper))
    print("iqr: ", data_iqr)
    print("median: ", np.median(data))

    lower_bound = np.percentile(data, range_lower) - 1.5 * data_iqr
    upper_bound = np.percentile(data, range_upper) + 1.5 * data_iqr

    print("Lower bound: ", lower_bound)
    print("Upper bound: ", upper_bound)

    for i, value in enumerate(data):
        if value < lower_bound or value > upper_bound:
            print("outlier detected: ", i)
            distance = 1
            while True:
                if i - distance >= 0 and data[i - distance] >= lower_bound and data[i - distance] <= upper_bound:
                    value = data[i - distance]
                    break
                elif i + distance < len(data) and data[i + distance] >= lower_bound and data[i + distance] <= upper_bound:
                    value = data[i + distance]
                    break
                else:
                    distance += 1
    return data
