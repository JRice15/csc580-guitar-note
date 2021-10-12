import tensorflow as tf
from tensorflow import keras

import numpy as np

class DataGenerator(keras.utils.Sequence):

    def __init__(self, ):
        ...

    def __len__(self):
        """
        return number of batches in this generator
        """
        ...

    def __getitem__(self, idx):
        """
        get batch number `idx`
        """
        ...

    def on_epoch_end(self):
        ...
