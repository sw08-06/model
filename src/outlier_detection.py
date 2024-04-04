import numpy as np
from scipy.stats import iqr
import matplotlib.pyplot as plt


def remove_outliers_iqr(data):
    iqr = iqr(data)
        
    lower_bound = np.percentile(data, 25) - 1.5 * iqr
    upper_bound = np.percentile(data, 75) + 1.5 * iqr
        
    cleaned_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    return cleaned_data

def remove_outliers_modified_zscore(data, threshold=3.5):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
        
    modified_z_scores = 0.6745 * (data - median) / mad
        
    outliers = np.abs(modified_z_scores) > threshold
        
    cleaned_data = data[~outliers]
    
    return cleaned_data

    
if __name__ == "__main__":
        
    data = np.load("data/frames/S2/BVP/1_BVP_0.npy")
    cleaned_data_iqr = remove_outliers_iqr(data)
    cleaned_data_modified_zscore = remove_outliers_modified_zscore(data)

    print("Original data: ", len(data))
    print("IQR cleaned data: ", len(cleaned_data_iqr))
    print("Modified z-score cleaned data: ", len(cleaned_data_modified_zscore))

     # Plot original data[0]
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(data, label='Original Data[0]')
    plt.legend()
    plt.title('Original Data[0]')
    
    # Plot cleaned data[0]
    plt.subplot(2, 1, 2)
    plt.plot(cleaned_data_iqr, label='Cleaned Data[0]', color='orange')
    plt.legend()
    plt.title('Cleaned Data[0]')
    
    plt.tight_layout()
    plt.show()