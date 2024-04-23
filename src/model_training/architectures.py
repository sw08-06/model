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


def model_v1():
    combined_input = keras.Input(shape=(4320, 1))

    bvp_input = SliceLayer(0, 3840)(combined_input)
    eda_input = SliceLayer(3840, 4080)(combined_input)
    temp_input = SliceLayer(4080, 4320)(combined_input)

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

    dense1 = keras.layers.Dense(units=80, activation="relu")(concatenated)
    dropout = keras.layers.Dropout(rate=0.75)(dense1)
    dense2 = keras.layers.Dense(units=40, activation="relu")(dropout)
    dense3 = keras.layers.Dense(units=20, activation="relu")(dense2)
    dense4 = keras.layers.Dense(units=10, activation="relu")(dense3)
    output = keras.layers.Dense(units=1, activation="sigmoid")(dense4)

    return keras.Model(inputs=combined_input, outputs=output)


if __name__ == "__main__":
    model_v1().summary()
