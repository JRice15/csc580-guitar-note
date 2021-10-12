import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten, Dropout



def tio_model(input_shape):
    """
    https://towardsdatascience.com/audio-to-guitar-tab-with-deep-learning-d76e12717f81
    """
    model_in = Input(shape = input_shape)
    conv1 = Conv2D(32, kernel_size = (3, 3), activation = 'relu')(model_in)
    conv2 = Conv2D(64, kernel_size = (3, 3), activation = 'relu')(conv1)
    conv3 = Conv2D(64, kernel_size = (3, 3), activation = 'relu')(conv2)
    pool1 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(conv3)
    flat = Flatten()(pool1)

    y1 = Dense(152, activation = 'relu')(flat)
    y1 = Dropout(0.5)(y1)
    y1 = Dense(76)(y1)
    y1 = Dropout(0.2)(y1)

    y2 = Dense(152, activation = 'relu')(flat)
    y2 = Dropout(0.5)(y2)
    y2 = Dense(76)(y2)
    y2 = Dropout(0.2)(y2)

    y3 = Dense(152, activation = 'relu')(flat)
    y3 = Dropout(0.5)(y3)
    y3 = Dense(76)(y3)
    y3 = Dropout(0.2)(y3)

    y4 = Dense(152, activation = 'relu')(flat)
    y4 = Dropout(0.5)(y4)
    y4 = Dense(76)(y4)
    y4 = Dropout(0.2)(y4)

    y5 = Dense(152, activation = 'relu')(flat)
    y5 = Dropout(0.5)(y5)
    y5 = Dense(76)(y5)
    y5 = Dropout(0.2)(y5)

    y6 = Dense(152, activation = 'relu')(flat)
    y6 = Dropout(0.5)(y6)
    y6 = Dense(76)(y6)
    y6 = Dropout(0.2)(y6)

    # Connect heads to final output layer
    out1 = Dense(19, activation = 'softmax', name = 'estring')(y1)
    out2 = Dense(19, activation = 'softmax', name = 'Bstring')(y2)
    out3 = Dense(19, activation = 'softmax', name = 'Gstring')(y3)
    out4 = Dense(19, activation = 'softmax', name = 'Dstring')(y4)
    out5 = Dense(19, activation = 'softmax', name = 'Astring')(y5)
    out6 = Dense(19, activation = 'softmax', name = 'Estring')(y6)

    # Create model
    model = Model(inputs = model_in, outputs = [out1, out2, out3, out4, out5, out6])

    loss = ['categorical_crossentropy'] * 6
    metrics = ['accuracy']

    return model, loss, metrics
