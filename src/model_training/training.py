import os
import keras
from model_training.architectures import model_v1
from generator import Generator


def execute_training(training_data_path, validation_data_path, model_name, batch_size, num_epochs):
    model = model_v1()
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=keras.metrics.BinaryAccuracy(),
    )

    training_data = Generator(training_data_path, batch_size)
    validation_data = Generator(validation_data_path, batch_size)

    def _scheduler(epoch, lr):
        if epoch == 12:
            return lr * 0.1
        elif epoch == 20:
            return lr * 0.1
        else:
            return lr

    os.makedirs("models", exist_ok=True)
    model.fit(
        x=training_data,
        validation_data=validation_data,
        epochs=num_epochs,
        verbose=1,
        callbacks=[
            keras.callbacks.ModelCheckpoint(f"models/{model_name}.h5", monitor="val_binary_accuracy", save_best_only=True),
            keras.callbacks.TensorBoard(),
            keras.callbacks.LearningRateScheduler(_scheduler),
        ],
    )


if __name__ == "__main__":
    execute_training(
        training_data_path=os.path.join("data", "frames", "training"),
        validation_data_path=os.path.join("data", "frames", "validation"),
        model_name="model_v1",
        batch_size=256,
        num_epochs=25,
    )
