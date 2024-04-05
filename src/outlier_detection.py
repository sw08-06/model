import numpy as np
from scipy.stats import iqr
import matplotlib.pyplot as plt

#Only remove
def remove_outliers_iqr(data):
    iqr_data = iqr(data)
        
    lower_bound = np.percentile(data, 25) - 1.5 * iqr_data
    upper_bound = np.percentile(data, 75) + 1.5 * iqr_data

    cleaned_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    return cleaned_data

#Replaces with neighbors
def replace_outliers_iqr(data):
    """
    Detects outliers using IQR and replaces them with closest valid neighbors

    Modifies the data object instead of creating copy - should be used as void function

    Args:
        data - the numpy array to clean 
    """
    data_iqr = iqr(data, rng=(25,75)) #Adjust to appropiate sensitivity
    print("iqr: ", data_iqr)
    print("median: ", np.median(data))
    
    # Determine the lower and upper bounds for outlier detection
    lower_bound = np.percentile(data, 25) - 1.5 * data_iqr #Adjust to appropiate sensitivity
    upper_bound = np.percentile(data, 75) + 1.5 * data_iqr #Adjust to appropiate sensitivity

    print("Lower bound: ", lower_bound)
    print("Upper bound: ", upper_bound)
    
    # Replace outliers with neighboring values
    for i in range(len(data)):
        if data[i] < lower_bound or data[i] > upper_bound:
            print("outlier detected: ", i)
            distance = 1
            while True:
                if i - distance >= 0 and data[i - distance] >= lower_bound and data[i - distance] <= upper_bound:
                    data[i] = data[i - distance]
                    break
                elif i + distance < len(data) and data[i + distance] >= lower_bound and data[i + distance] <= upper_bound:
                    data[i] = data[i + distance]
                    break
                else:
                    distance += 1
    return data

    
if __name__ == "__main__":    
    data = np.load("data/frames/S2/EDA/0_EDA_0.npy")
    print("---Original data [min,max]: [", min(data), ", ", max(data), "]")
    print("Original data length: ", len(data))

    plt.figure(figsize=(10, 5))
    plt.subplot(3, 1, 1)
    plt.plot(data, label='Original')
    plt.legend()
    plt.title('Original')

    replace_outliers_iqr(data)

    print("Replaced data [min,max]: [", min(data), ", ", max(data), "]")
    print("IQR replaced data length: ", len(data))

    plt.subplot(3, 1, 3)
    plt.plot(data, label='Replaced', color='orange')
    plt.legend()
    plt.title('Replaced')
    
    plt.tight_layout()
    plt.show()