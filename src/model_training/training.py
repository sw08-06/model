import os
import tensorflow as tf
import keras
from architectures import *
from generator import Generator


def execute_training(training_data_path, validation_data_path, model, model_name, batch_size, num_epochs):
    model.summary()
    print("---- GPU available ----" if tf.config.list_physical_devices("GPU") else "---- No GPU available ----")
    print(f"training data path: {training_data_path}")
    print(f"validation data path: {validation_data_path}")

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.BinaryFocalCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    training_data_generator = Generator(training_data_path, batch_size)
    validation_data_generator = Generator(validation_data_path, batch_size)

    def _scheduler(epoch, learning_rate):
        if epoch == 20:
            return learning_rate * 0.1
        elif epoch == 27:
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
    window_sizes = [5, 15, 30, 60, 90, 120]
    loso_subject = "S2"

    for window_size in window_sizes:
        execute_training(
            training_data_path=os.path.join(os.environ.get("DATA_PATH"), f"frames_{window_size}s_{loso_subject}_stress_mul8", "training.h5"),
            validation_data_path=os.path.join(os.environ.get("DATA_PATH"), f"frames_{window_size}s_{loso_subject}_stress_mul8", "validation.h5"),
            model=model_v4(window_size),
            model_name=f"model_v4_{loso_subject}_{window_size}s_focal_stress_mul8",
            batch_size=64,
            num_epochs=32,
        )
