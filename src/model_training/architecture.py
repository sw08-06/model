import tensorflow as tf


def model_v1():
    bvp_input = tf.keras.Input(shape=(3840, 1))
    eda_temp_input = tf.keras.Input(shape=(240, 1))

    bvp_conv1 = tf.keras.layers.Conv1D(filters=40, kernel_size=16, strides=4, padding="same", activation="relu")(bvp_input)  # 40x960
    bvp_conv2 = tf.keras.layers.Conv1D(filters=20, kernel_size=8, strides=4, padding="same", activation="relu")(bvp_conv1)  # 20x240
    bvp_conv3 = tf.keras.layers.Conv1D(filters=10, kernel_size=4, strides=4, padding="same", activation="relu")(bvp_conv2)  # 10x60
    bvp_conv4 = tf.keras.layers.Conv1D(filters=10, kernel_size=2, strides=1, padding="same", activation="relu")(bvp_conv3)  # 10x30
    bvp_flatten = tf.keras.layers.Flatten()(bvp_conv4)

    eda_temp_conv1 = tf.keras.layers.Conv1D(filters=40, kernel_size=8, strides=2, padding="same", activation="relu")(eda_temp_input)  # 40x120
    eda_temp_conv2 = tf.keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="same", activation="relu")(eda_temp_conv1)  # 20X60
    eda_temp_conv3 = tf.keras.layers.Conv1D(filters=10, kernel_size=2, strides=2, padding="same", activation="relu")(eda_temp_conv2)  # 10X30
    eda_temp_flatten = tf.keras.layers.Flatten()(eda_temp_conv3)

    concatenated = tf.keras.layers.concatenate([bvp_flatten, eda_temp_flatten, eda_temp_flatten])

    dense1 = tf.keras.layers.Dense(units=32, activation="relu")(concatenated)
    dense2 = tf.keras.layers.Dense(units=16, activation="relu")(dense1)
    dense3 = tf.keras.layers.Dense(units=8, activation="relu")(dense2)
    output = tf.keras.layers.Dense(units=1, activation="sigmoid")(dense3)

    return tf.keras.Model(inputs=[bvp_input, eda_temp_input, eda_temp_input], outputs=output)
