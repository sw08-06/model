# ML Model
This repository contains code for creating frames from the WESAD dataset, preprocessing of the frames, training of ML models, and evaluation of the models.

The required packages to run the project can be installed with pip using the command:
```
pip install -r requirements.txt
```

## Data Handling
The `data_handling_wesad.py` script is used to divide the sensor data from the wrist signals. To use the script, provide the path to the WESAD dataset, the desired data types, their respective sampling frequencies, the window size in seconds for each data type, and the window overlap in seconds. If the script is rerun, previously created frames are not overwritten.

## Preprocessing
The `data_preprocessing.py` script applies preprocessing to the frames obtained from the data of the wrist signals. To use the script, provide the path to the data frames, the data types, and the preprocessing functions to be applied. If the script is rerun, preprocessed frames are not overwritten.
