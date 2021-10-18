import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, Input,
                                     MaxPooling2D, Reshape)
from tensorflow.keras import Model

from utils import MIDI_MAX, MIDI_MIN


def get_model(modelname, input_shape):
    if modelname == "tio":
        return tio_model(input_shape)
    else:
        raise ValueError("Unknown model {}".format(modelname))

def tio_model(input_shape):
    """
    https://towardsdatascience.com/audio-to-guitar-tab-with-deep-learning-d76e12717f81
    """
    model_in = Input(shape=input_shape)

    w_channels = Reshape(input_shape + (1,))(model_in)
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(w_channels)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(conv1)
    conv3 = Conv2D(64, kernel_size=(3, 3), activation='relu')(conv2)

    pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)
    flat = Flatten()(pool)

    ### Original string-wise model
    # outputs = []
    # for string in "EADGBe":
    #     x = Dense(152, activation='relu')(flat)
    #     x = Dropout(0.5)(x)
    #     x = Dense(76)(x)
    #     x = Dropout(0.2)(x)
    #     out = Dense(19, activation='softmax', name=string+'-string')(x)
    #     outputs.append(out)

    ### New version
    x = Dense(128, activation='relu')(flat)
    x = Dropout(0.5)(x)
    x = Dense(64)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(MIDI_MAX-MIDI_MIN, activation='sigmoid', name="output")(x)

    # Create model
    model = Model(inputs=model_in, outputs=outputs)

    loss = 'binary_crossentropy'
    metrics = ['accuracy']

    return model, loss, metrics
