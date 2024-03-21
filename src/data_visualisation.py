import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
import pickle

def data_visualiser(path, data_type, hz):
    """
    Visualises the raw WESAD data.
    Args:
        path: The path to .pkl file
        data_type: The signal to visualise
        hz: The sampling frequence of the signal specified in data_type
    """

    with open(path, 'rb') as f:
        # Load the data from the file using 'latin1' encoding
        data = pickle.load(f, encoding='latin1')

    wrist_data = data['signal']['wrist'][data_type]
    label_data = data['label']

    # Resample label data to match the sampling rate of wrist data (64 Hz)
    resampled_label_data = resample(label_data, len(wrist_data))

    # Time arrays for both datasets
    time_wrist = np.arange(len(wrist_data)) / hz 
    time_label = np.arange(len(resampled_label_data)) / hz

    # Plotting
    plt.figure(figsize=(10, 8))

    # Plot wrist data
    plt.subplot(2, 1, 1)
    plt.plot(time_wrist, wrist_data, color='blue', label='Wrist Data')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Wrist Data')
    plt.title('Wrist Data')

    # Plot resampled label data
    plt.subplot(2, 1, 2)
    plt.plot(time_label, resampled_label_data, color='red', label='Resampled Label Data')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Label Data')
    plt.title('Resampled Label Data')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_visualiser("data/WESAD/S11/S11.pkl", "EDA", 4)
    
