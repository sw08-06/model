# ML Model
This repository contains code for creating frames from the WESAD dataset, preprocessing of the WESAD dataset, training of ML models, and evaluating the models.

To install the required packages locally to run the project, use pip with the following command:
```
pip install -r requirements.txt
```
Alternatively, build a Docker image with:
```
docker build -t model-training .
```
And then run the Docker container with:
```
docker run -it --gpus all -v ${PWD}/models:/app/models -v ${PWD}/data:/app/data model-training
```

## Data Processing
### Preprocessing
The `data_preprocessing.py` script applies preprocessing to the raw WESAD dataset and saves the preprocessed data in a new directory. To use the script, provide the following:
1. Path to the WESAD dataset
2. Data types
3. Sampling frequencies of the data types
3. Preprocessing functions to be applied

The script allows for flexible preprocessing method application. Each data type can have a list of preprocessing functions associated with it. Some preprocessing methods are specified in `data_processing\preprocessing_methods.py` file, however, other methods can also be applied. The preprocessed files are saved in the exact same structure as the original WESAD dataset, allowing for using the `data_partitioning.py` script on the preprocessed data. If the script is rerun, a new folder is created with the data, ensuring previously created data is not overwritten.

### Partitioning
The `data_partitioning.py` script is used to partion the sensor data from the wrist signals into frames. The data is split into training, validation, and testing data and saved in HDF5 files. To use the script, provide the following:
1. Path to the WESAD dataset
2. Data types
3. Data labels
4. Sampling frequencies of the data types
5. Window size in seconds
6. Window overlap in seconds
7. Name of the LOSO subject
8. Ratio to split the data into training and validation datasets
9. Stress multiplier
10. Preprocessing functions to be applied

If the script is rerun, a new folder is created with the frames, ensuring previously created frames are not overwritten.

## Model Training
The `training.py` is responsible for training machine learning models. This script automates the training process, making it easier to experiment with different model architectures and training configurations. It leverages pre-defined model architectures from `model_training\architectures.py`.

To execute training, provide the following parameters:
1. Path to the training data directory.
2. Path to the validation data directory.
3. Name of the model architecture to use.
4. Batch size for training.
5. Number of epochs for training.

The script compiles the model using Adam optimizer, binary cross-entropy loss, and binary accuracy metrics. A custom learning rate scheduler adjusts the learning rate during training epochs. Callbacks including model checkpointing and learning rate scheduling are utilized during training. Trained models are saved in the models directory with the specified model name.

## Model Evaluation
The `evaluation.py` script provides functionality for evaluating trained machine learning models using various metrics. To use the script, a list of the names of models that are desired to evaluate, must simply be provided. The scripts allows for evaluation using the following metrics:
- **F1 Score:** F1 score is calculated for each model and displayed in a tabular format.
- **Precision-Recall Curves:** Precision-recall curves are generated for each model and saved as an image.
- **ROC Curves:** ROC curves are generated for each model and saved as an image.
- **Confusion Matrix:** Confusion matrices are generated for each model and saved as images.

## Utils
The `data_analysis.py` script allows for analyzing the frames created by the `data_partioning.py` script. It provides the following:
- **Loading Data:** The `load_HDF5` function loads data from a specific dataset within a group in an HDF5 file. The `load_pkl` function loads data of a specific data type from a specific subject from the WESAD dataset or a preprocessed version of the WESAD dataset.
- **Fourier Analysis:** The `fourier_analysis` function performs Fourier analysis on a frame of data.
- **Preprocessing Methods:** Any preprocessing methods from `data_processing\preprocessing_methods.py` can be applied to datasets than have been loaded.
- **Plotting Data:** The `plot_data` function plots the original and filtered data, as well as their Fourier transforms.
- **Visualizing Frames:** The `visualize_frames` function visualizes a list of frames, with each frame plotted as a separate subplot.

## License
This project is licensed under the MIT License.
