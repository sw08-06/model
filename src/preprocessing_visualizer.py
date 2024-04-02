import numpy as np
import matplotlib.pyplot as plt

file_path = "data\\preprocessed_frames\\S10\\0_FV_10.npy"
feature_vector = np.load(file_path, allow_pickle=True)

data_to_visualize = feature_vector[0]

# Create time axis
time_axis = np.arange(len(data_to_visualize))

# Plot the data
plt.plot(time_axis, data_to_visualize)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Feature Vector 0')
plt.grid(True)
plt.show()