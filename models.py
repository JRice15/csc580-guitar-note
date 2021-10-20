import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

import custom_layers
from utils import MIDI_MAX, MIDI_MIN


def get_model(modelname, input_shape):
    if modelname == "tio":
        return tio_model(input_shape)
    elif modelname == "lstm1":
        return lstm1(input_shape)
    else:
        raise ValueError("Unknown model {}".format(modelname))

def tio_model(input_shape):
    """
    https://towardsdatascience.com/audio-to-guitar-tab-with-deep-learning-d76e12717f81
    """
    model_in = layers.Input(shape=input_shape)

    w_channels = layers.Reshape(input_shape + (1,))(model_in)
    conv1 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(w_channels)
    conv2 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(conv1)
    conv3 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(conv2)

    pool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)
    flat = layers.Flatten()(pool)

    ### Original string-wise model
    # outputs = []
    # for string in "EADGBe":
    #     x = layers.Dense(152, activation='relu')(flat)
    #     x = layers.Dropout(0.5)(x)
    #     x = layers.Dense(76)(x)
    #     x = layers.Dropout(0.2)(x)
    #     out = layers.Dense(19, activation='softmax', name=string+'-string')(x)
    #     outputs.append(out)

    ### New version
    x = layers.Dense(128, activation='relu')(flat)
    # x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    # x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(MIDI_MAX-MIDI_MIN, activation='sigmoid', name="output")(x)

    # Create model
    model = Model(inputs=model_in, outputs=outputs)

    loss = 'binary_crossentropy'
    # TODO we need to find a meaningful metric for this data
    metrics = None

    return model, loss, metrics


def lstm1(input_shape):
    model_in = layers.Input(shape=input_shape)
    x = model_in

    # change from shape (batch, freq, time) to (batch, time freq) to make it a timeseries
    x = custom_layers.Transpose([0, 2, 1])(x)

    # calculate timestep-wise features
    x = layers.LSTM(128, return_sequences=True)(x)
    # calculate accumulated feature
    x = layers.LSTM(128)(x)

    # dense output network
    x = layers.Dense(64, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Dense(64, activation='relu')(x)
    # x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(MIDI_MAX-MIDI_MIN, activation='sigmoid', name="output")(x)

    # Create model
    model = Model(inputs=model_in, outputs=outputs)

    loss = 'binary_crossentropy'
    # TODO we need to find a meaningful metric for this data
    metrics = None

    return model, loss, metrics
