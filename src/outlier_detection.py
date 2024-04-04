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
    # Calculate IQR (Interquartile Range)
    data_iqr = iqr(data)
    
    # Determine the lower and upper bounds for outlier detection
    lower_bound = np.percentile(data, 25) - 1.5 * data_iqr
    upper_bound = np.percentile(data, 75) + 1.5 * data_iqr

    print("Lower bound: ", lower_bound)
    print("Upper bound: ", upper_bound)
    
    # Replace outliers with neighboring values
    for i in range(len(data)):
        if data[i] < lower_bound or data[i] > upper_bound:
            #print("Outlier index: ", i)
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
        
    data = np.load("data/frames/S2/BVP/0_BVP_4300.npy")
    print("---Original data [min,max]: [", min(data), ", ", max(data), "]")
    cleaned_data_iqr = remove_outliers_iqr(data)
    print("***Original data [min,max]: [", min(data), ", ", max(data), "]")
    replaced_data_iqr = replace_outliers_iqr(data)
    print(">>>Original data [min,max]: [", min(data), ", ", max(data), "]")

    print("Original data: ", len(data))
    print("IQR removed data: ", len(cleaned_data_iqr))
    print("IQR replaced data: ", len(replaced_data_iqr))

    print("Original data [min,max]: [", min(data), ", ", max(data), "]")
    print("IQR removed data [min,max]: [", min(cleaned_data_iqr), ", ", max(cleaned_data_iqr), "]")
    print("IQR replcaed data [min,max]: [", min(replaced_data_iqr), ", ", max(replaced_data_iqr), "]")


    plt.figure(figsize=(10, 5))
    plt.subplot(3, 1, 1)
    plt.plot(data, label='Original')
    plt.legend()
    plt.title('Original')
    

    plt.subplot(3, 1, 2)
    plt.plot(cleaned_data_iqr, label='Removed', color='orange')
    plt.legend()
    plt.title('Removed')

    plt.subplot(3, 1, 3)
    plt.plot(replaced_data_iqr, label='Replaced', color='blue')
    plt.legend()
    plt.title('Replaced')
    
    plt.tight_layout()
    plt.show()