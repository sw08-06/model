import keras


class SliceLayer(keras.layers.Layer):
    def __init__(self, start_index, end_index, **kwargs):
        super(SliceLayer, self).__init__(**kwargs)
        self.start_index = start_index
        self.end_index = end_index

    def call(self, inputs):
        return inputs[:, self.start_index : self.end_index]

    def get_config(self):
        config = super(SliceLayer, self).get_config()
        config.update({"start_index": self.start_index, "end_index": self.end_index})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def model_v1(window_size):
    bvp_length = window_size * 64
    eda_temp_length = window_size * 4
    total_length = bvp_length + 2 * eda_temp_length

    combined_input = keras.Input(shape=(total_length, 1))

    bvp_input = SliceLayer(0, bvp_length)(combined_input)
    eda_input = SliceLayer(bvp_length, bvp_length + eda_temp_length)(combined_input)
    temp_input = SliceLayer(bvp_length + eda_temp_length, total_length)(combined_input)

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

    dense1 = keras.layers.Dense(units=100, activation="relu")(concatenated)
    dropout = keras.layers.Dropout(rate=0.25)(dense1)
    output = keras.layers.Dense(units=1, activation="sigmoid")(dropout)

    return keras.Model(inputs=combined_input, outputs=output)


def model_v2(window_size):
    bvp_length = window_size * 64
    eda_temp_length = window_size * 4
    total_length = bvp_length + 2 * eda_temp_length

    combined_input = keras.Input(shape=(total_length,))

    bvp_input = SliceLayer(0, bvp_length)(combined_input)
    eda_input = SliceLayer(bvp_length, bvp_length + eda_temp_length)(combined_input)
    temp_input = SliceLayer(bvp_length + eda_temp_length, total_length)(combined_input)

    bvp_dense1 = keras.layers.Dense(units=128, activation="relu")(bvp_input)
    bvp_dense2 = keras.layers.Dense(units=64, activation="relu")(bvp_dense1)
    bvp_dense3 = keras.layers.Dense(units=32, activation="relu")(bvp_dense2)

    eda_dense1 = keras.layers.Dense(units=32, activation="relu")(eda_input)

    temp_dense1 = keras.layers.Dense(units=32, activation="relu")(temp_input)

    concatenated = keras.layers.Concatenate(axis=1)([bvp_dense3, eda_dense1, temp_dense1])

    dense_combined_1 = keras.layers.Dense(units=32, activation="relu")(concatenated)
    dropout = keras.layers.Dropout(rate=0.25)(dense_combined_1)
    dense_combined_2 = keras.layers.Dense(units=16, activation="relu")(dropout)
    dense_combined_3 = keras.layers.Dense(units=8, activation="relu")(dense_combined_2)
    output = keras.layers.Dense(units=1, activation="sigmoid")(dense_combined_3)

    return keras.Model(inputs=combined_input, outputs=output)


if __name__ == "__main__":
    model_v1(60).summary()
    model_v2(60).summary()
