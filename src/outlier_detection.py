import numpy as np
from scipy.stats import iqr
import matplotlib.pyplot as plt


def remove_outliers_iqr(data):
    cleaned_data = []
    for dataset in data:
        dataset_iqr = iqr(dataset)
        
        lower_bound = np.percentile(dataset, 25) - 1.5 * dataset_iqr
        upper_bound = np.percentile(dataset, 75) + 1.5 * dataset_iqr
        
        cleaned_dataset = dataset[(dataset >= lower_bound) & (dataset <= upper_bound)]
        
        cleaned_data.append(cleaned_dataset)
    
    return cleaned_data

def remove_outliers_modified_zscore(data, threshold=3.5):
    cleaned_data = []
    for dataset in data:

        median = np.median(dataset)
        mad = np.median(np.abs(dataset - median))
        
        modified_z_scores = 0.6745 * (dataset - median) / mad
        
        outliers = np.abs(modified_z_scores) > threshold
        
        cleaned_dataset = dataset[~outliers]
        cleaned_data.append(cleaned_dataset)
    
    return cleaned_data

    
if __name__ == "__main__":
    bvp = np.load("data/frames/S2/BVP/1_BVP_0.npy")
    eda = np.load("data/frames/S2/EDA/1_EDA_0.npy")
    temp = np.load("data/frames/S17/TEMP/1_TEMP_0.npy")
    data = [bvp, eda, temp]
    cleaned_data_iqr = remove_outliers_iqr(data)
    cleaned_data_modified_zscore = remove_outliers_modified_zscore(data)

    print("Original data: ", len(data[0]))
    print("IQR cleaned data: ", len(cleaned_data_iqr[0]))
    print("Modified z-score cleaned data: ", len(cleaned_data_modified_zscore[0]))

     # Plot original data[0]
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(data[0], label='Original Data[1]')
    plt.legend()
    plt.title('Original Data[0]')
    
    # Plot cleaned data[0]
    plt.subplot(2, 1, 2)
    plt.plot(cleaned_data_iqr[0], label='Cleaned Data[1]', color='orange')
    plt.legend()
    plt.title('Cleaned Data[0]')
    
    plt.tight_layout()
    plt.show()