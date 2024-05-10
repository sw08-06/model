import os
import tensorflow as tf
import keras
from architectures import *
from generator import Generator


def execute_training(training_data_path, validation_data_path, model, model_name, batch_size, num_epochs):
    model.summary()
    print("---- GPU available ----" if tf.config.list_physical_devices("GPU") else "---- No GPU available ----")
    print(f"Training data path: {training_data_path}")
    print(f"Validation data path: {validation_data_path}")

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    training_data_generator = Generator(training_data_path, batch_size)
    validation_data_generator = Generator(validation_data_path, batch_size)

    def _scheduler(epoch, learning_rate):
        if epoch == 12:
            return learning_rate * 0.1
        elif epoch == 17:
            return learning_rate * 0.1
        else:
            return learning_rate

    os.makedirs(os.environ.get("MODEL_PATH"), exist_ok=True)
    model.fit(
        x=training_data_generator,
        validation_data=validation_data_generator,
        epochs=num_epochs,
        verbose=1,
        callbacks=[
            keras.callbacks.ModelCheckpoint(os.path.join(os.environ.get("MODEL_PATH"), f"{model_name}.keras"), monitor="val_binary_accuracy", save_best_only=True),
            keras.callbacks.LearningRateScheduler(_scheduler),
        ],
    )


if __name__ == "__main__":
    models = [lambda w: model_v2(w), lambda w: model_v4(w)]
    model_name = ["model_v2", "model_v4"]
    subjects = ["S2", "S3", "S4", "S5", "S6"]
    windows = [5, 30, 60, 90, 120]

    for i, model in enumerate(models):
        for subject in subjects:
            for window in windows:
                execute_training(
                    training_data_path=os.path.join(os.environ.get("DATA_PATH"), f"frames_{subject}_{window}", "training.h5"),
                    validation_data_path=os.path.join(os.environ.get("DATA_PATH"), f"frames_{subject}_{window}", "validation.h5"),
                    model=model(window),
                    model_name=f"{model_name[i]}_{subject}_{window}",
                    batch_size=64,
                    num_epochs=20,
                )
