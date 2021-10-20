import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

from utils import MIDI_MAX, MIDI_MIN



class Transpose(layers.Layer):

    def __init__(self, perm, **kwargs):
        super().__init__(**kwargs)
        self.perm = perm
    
    def call(self, x):
        return tf.transpose(x, perm=self.perm)
    
    def get_config(self):
        config = super().get_config()
        config["perm"] = self.perm
        return config



