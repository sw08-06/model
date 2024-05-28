import os
import tensorflow as tf
from tensorflow import lite, io
from keras.models import load_model
from architectures import SliceLayer


def compress_model(model_name):
    """
    Converts a TensorFlow model into TensorFlow Lite model
    using the default optimization strategy of Tensorflow that enables post-training quantization.

    Args:
        model_name (str): Name of the model to compress
    """
    model = load_model(f"models/{model_name}.keras", custom_objects={"SliceLayer": SliceLayer})

    @tf.function
    def wrapped_model(input_tensor):
        return model(input_tensor)

    concrete_func = wrapped_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    converter = lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [lite.Optimize.DEFAULT]
    tf_lite_model = converter.convert()

    os.makedirs("models_compressed", exist_ok=True)
    io.write_file(f"models_compressed/{model_name}.tflite", tf_lite_model)


if __name__ == "__main__":
    compress_model("model_v4_S4_120")
