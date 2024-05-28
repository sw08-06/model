import os
import tensorflow as tf
import keras


def compress_model(model_name):
    """
    Converts a TensorFlow model into TensorFlow Lite model
    using the default optimization strategy of Tensorflow that enables post-training quantization.

    Args:
        model_name (str): Name of the model to compress
    """
    model = keras.models.load_model(f"models/{model_name}.keras")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tf_flite_model = converter.convert()
    os.makedirs("models_compressed", exist_ok=True)
    tf.io.write_file(f"models_compressed/{model_name}.tflite", tf_flite_model)


if __name__ == "__main__":
    compress_model("model_v4_S4_120")
