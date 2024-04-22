import os
import tensorflow as tf
import keras
from architectures import model_v1
from generator import Generator


def execute_training(training_data_path, validation_data_path, model_name, batch_size, num_epochs):
    model = model_v1()
    model.summary()
    print("---- GPU available ----" if tf.config.list_physical_devices("GPU") else "---- No GPU available ----")

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    training_data_generator = Generator(training_data_path, batch_size)
    validation_data_generator = Generator(validation_data_path, batch_size)

    def _scheduler(epoch, lr):
        if epoch == 12:
            return lr * 0.1
        elif epoch == 20:
            return lr * 0.1
        else:
            return lr

    os.makedirs(os.environ.get("MODEL_PATH"), exist_ok=True)
    model.fit(
        x=training_data_generator,
        validation_data=validation_data_generator,
        epochs=num_epochs,
        verbose=1,
        callbacks=[
            keras.callbacks.ModelCheckpoint(os.path.join(os.environ.get("MODEL_PATH"), f"{model_name}.keras"), monitor="val_binary_accuracy", save_best_only=True),
            keras.callbacks.TensorBoard(),
            keras.callbacks.LearningRateScheduler(_scheduler),
        ],
    )


if __name__ == "__main__":
    execute_training(
        training_data_path=os.environ.get("TRAINING_DATA_PATH"),
        validation_data_path=os.environ.get("VALIDATION_DATA_PATH"),
        model_name="model_v1",
        batch_size=256,
        num_epochs=25,
    )
