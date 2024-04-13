# ML Model
This repository contains code for creating frames from the WESAD dataset, preprocessing of the frames, training of ML models, and evaluating the models.

To install the required packages to run the project, use pip with the following command:
```
pip install -r requirements.txt
```

## Data Processing
The `data_partitioning.py` script is used to partion the sensor data from the wrist signals into frames. The data is split into training, validation, and testing data and saved in HDF5 files. To use the script, provide the following:
1. Path to the WESAD dataset
2. Data types
3. Sampling frequencies of the data types
4. Window size in seconds
5. Window overlap in seconds
6. Name of the LOSO subject
6. Ratio to split the data into training and validation datasets

If the script is rerun, a new folder is created with the frames, ensuring previously created frames are not overwritten.

The `data_preprocessing.py` script applies preprocessing to the frames obtained from the wrist signals' data. To use the script, provide:
1. Path to the data frames
2. Data types
3. Preprocessing functions to be applied

If the script is rerun, preprocessed frames are not overwritten.

## Model Training
The `training.py` is responsible for training machine learning models. This script automates the training process, making it easier to experiment with different model architectures and training configurations. It leverages pre-defined model architectures from `model_training\architectures`.

To execute training, provide the following parameters:
1. Path to the training data directory.
2. Path to the validation data directory.
3. Name of the model architecture to use.
4. Batch size for training.
5. Number of epochs for training.

### Implementation Details
- **Model Compilation:** The script compiles the model using Adam optimizer, binary cross-entropy loss, and binary accuracy metrics.
- **Learning Rate Scheduling:** A custom learning rate scheduler adjusts the learning rate during training epochs.
- **Callbacks:** Callbacks including model checkpointing, TensorBoard logging, and learning rate scheduling are utilized during training.
- **Model Saving:** Trained models are saved in the models directory with the specified model name.

## Model Evaluation

## Utils
The `data_analysis.py` script allows for analyzing the frames created by the `data_partioning.py` script. It provides the following:
1. **Loading Data:** The `load_file` function loads data from a specific dataset within a group in an HDF5 file.
2. **Fourier Analysis:** The `fourier_analysis` function performs Fourier analysis on a frame of data.
3. **Preprocessing Methods:** Any preprocessing methods from `data_processing\preprocessing_methods.py` can be applied to datasets than have been loaded.
3. **Plotting Data:** The `plot_data` function plots the original and filtered data, as well as their Fourier transforms.
4. **Visualizing Frames:** The `visualize_frames` function visualizes a list of frames, with each frame plotted as a separate subplot.

## License
This project is licensed under the MIT License.