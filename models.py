import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten, Dropout



def tio_model(input_shape):
    """
    https://towardsdatascience.com/audio-to-guitar-tab-with-deep-learning-d76e12717f81
    """
    model_in = Input(shape=input_shape)

    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(model_in)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(conv1)
    conv3 = Conv2D(64, kernel_size=(3, 3), activation='relu')(conv2)

    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)
    flat = Flatten()(pool1)

    outputs = []
    for string in "EADGBe":
        x = Dense(152, activation='relu')(flat)
        x = Dropout(0.5)(x)
        x = Dense(76)(x)
        x = Dropout(0.2)(x)
        out = Dense(19, activation='softmax', name=string+'-string')(x)
        outputs.append(out)

    # Create model
    model = Model(inputs=model_in, outputs=outputs)

    loss = ['categorical_crossentropy'] * 6
    metrics = ['accuracy']

    return model, loss, metrics
