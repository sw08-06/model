import os
import tensorflow as tf
import keras
from architectures import *
from generator import Generator


class ModelTrainer:
    def __init__(self, data_path, model_path, models, model_names, subjects, window_sizes):
        """
        Initializes a ModelTrainer instance.

        Args:
            data_path (str): Path to the directory containing the data.
            model_path (str): Path to the directory where trained models will be saved.
            models (list): List of model functions containing architectures to be trained.
            model_names (list): List of model names corresponding to the models.
            subjects (list): List of subject names.
            window_sizes (list): List of window sizes for data.
        """
        self.data_path = data_path
        self.model_path = model_path
        self.models = models
        self.model_names = model_names
        self.subjects = subjects
        self.window_sizes = window_sizes

    def _compile_model(self, model):
        """
        Compiles the Keras model with specified parameters.

        Args:
            model (keras.Model): Keras model to be compiled.
        """
        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[keras.metrics.BinaryAccuracy()],
        )

    def _create_checkpoint_callback(self, model_name):
        """
        Creates a callback for saving the best model checkpoints.

        Args:
            model_name (str): Name of the model.

        Returns:
            keras.callbacks.ModelCheckpoint: Model checkpoint callback.
        """
        return keras.callbacks.ModelCheckpoint(
            os.path.join(self.model_path, f"{model_name}.keras"),
            monitor="val_binary_accuracy",
            save_best_only=True,
        )

    def _create_scheduler_callback(self):
        """
        Creates a learning rate scheduler callback.

        Returns:
            keras.callbacks.LearningRateScheduler: Learning rate scheduler callback.
        """

        def _scheduler(epoch, learning_rate):
            if epoch == 18 or epoch == 22:
                return learning_rate * 0.1
            else:
                return learning_rate

        return keras.callbacks.LearningRateScheduler(_scheduler)

    def execute_training(self, training_data_path, validation_data_path, model, model_name, batch_size, num_epochs):
        """
        Executes the training process for a given model.

        Args:
            training_data_path (str): Path to the training data.
            validation_data_path (str): Path to the validation data.
            model (keras.Model): Keras model to be trained.
            model_name (str): Name of the model.
            batch_size (int): Size of the training batches.
            num_epochs (int): Number of training epochs.
        """
        model.summary()
        print("---- GPU available ----" if tf.config.list_physical_devices("GPU") else "---- No GPU available ----")
        print(f"Training data path: {training_data_path}")
        print(f"Validation data path: {validation_data_path}")

        self._compile_model(model)

        training_data_generator = Generator(training_data_path, batch_size)
        validation_data_generator = Generator(validation_data_path, batch_size)

        os.makedirs(self.model_path, exist_ok=True)
        model.fit(
            x=training_data_generator,
            validation_data=validation_data_generator,
            epochs=num_epochs,
            verbose=1,
            callbacks=[
                self._create_checkpoint_callback(model_name),
                self._create_scheduler_callback(),
            ],
        )


if __name__ == "__main__":
    models = [model_v2, model_v4]
    model_names = ["model_v2", "model_v4"]
    subjects = ["S2", "S3", "S4", "S5", "S6"]
    window_sizes = [5, 30, 60, 90, 120]

    trainer = ModelTrainer(
        data_path=os.environ.get("DATA_PATH"),
        model_path=os.environ.get("MODEL_PATH"),
        models=models,
        model_names=model_names,
        subjects=subjects,
        window_sizes=window_sizes,
    )

    for model, model_name in zip(models, model_names):
        for subject in subjects:
            for window in window_sizes:
                trainer.execute_training(
                    training_data_path=os.path.join(trainer.data_path, f"frames_{subject}_{window}", "training.h5"),
                    validation_data_path=os.path.join(trainer.data_path, f"frames_{subject}_{window}", "validation.h5"),
                    model=model(window),
                    model_name=f"{model_name}_{subject}_{window}",
                    batch_size=64,
                    num_epochs=25,
                )
