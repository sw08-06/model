import keras


def model_v1():
    combined_input = keras.Input(shape=(4320, 1))

    bvp_input = keras.layers.Lambda(lambda x: x[:, :3840], output_shape=(3840, 1))(combined_input)
    eda_input = keras.layers.Lambda(lambda x: x[:, 3840:4080], output_shape=(240, 1))(combined_input)
    temp_input = keras.layers.Lambda(lambda x: x[:, 4080:4320], output_shape=(240, 1))(combined_input)

    bvp_conv1 = keras.layers.Conv1D(filters=40, kernel_size=16, strides=4, padding="same", activation="relu")(bvp_input)
    bvp_conv2 = keras.layers.Conv1D(filters=20, kernel_size=8, strides=4, padding="same", activation="relu")(bvp_conv1)
    bvp_conv3 = keras.layers.Conv1D(filters=10, kernel_size=4, strides=4, padding="same", activation="relu")(bvp_conv2)
    bvp_conv4 = keras.layers.Conv1D(filters=10, kernel_size=2, strides=2, padding="same", activation="relu")(bvp_conv3)
    bvp_flatten = keras.layers.Flatten()(bvp_conv4)

    eda_conv1 = keras.layers.Conv1D(filters=40, kernel_size=8, strides=2, padding="same", activation="relu")(eda_input)
    eda_conv2 = keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="same", activation="relu")(eda_conv1)
    eda_conv3 = keras.layers.Conv1D(filters=10, kernel_size=2, strides=2, padding="same", activation="relu")(eda_conv2)
    eda_flatten = keras.layers.Flatten()(eda_conv3)

    temp_conv1 = keras.layers.Conv1D(filters=40, kernel_size=8, strides=2, padding="same", activation="relu")(temp_input)
    temp_conv2 = keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="same", activation="relu")(temp_conv1)
    temp_conv3 = keras.layers.Conv1D(filters=10, kernel_size=2, strides=2, padding="same", activation="relu")(temp_conv2)
    temp_flatten = keras.layers.Flatten()(temp_conv3)

    concatenated = keras.layers.Concatenate()([bvp_flatten, eda_flatten, temp_flatten])

    dense1 = keras.layers.Dense(units=32, activation="relu")(concatenated)
    dense2 = keras.layers.Dense(units=16, activation="relu")(dense1)
    dense3 = keras.layers.Dense(units=8, activation="relu")(dense2)
    output = keras.layers.Dense(units=1, activation="sigmoid")(dense3)

    return keras.Model(inputs=combined_input, outputs=output)
