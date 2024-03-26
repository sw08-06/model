# ML Model
This repository includes code for creating frames from the WESAD dataset, preprocessing of the frames, training of the ML model, and evaluation of the model.

### Data Handling
To divide the sensor data from the wrist signals, use the `data_handling_wesad.py` script. To use the script, the path to the WESAD dataset must be provided as well as the name of the desired data types, the sampling frequencies of these, the desired window size in seconds for each data type, and the window overlap in seconds. If the script is rerun, the previously created frames are not overwritten.