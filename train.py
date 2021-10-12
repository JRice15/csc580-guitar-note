import glob
import os
import re
import argparse
import sys

import jams
import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, Input,
                                     MaxPooling2D)
from tensorflow.keras import optimizer

import utils
import visualize
from models import tio_model

model_options = {
    "tio": tio_model
}

parser = argparse.ArgumentParser()
parser.add_argument("--model",required=True)
ARGS = parser.parse_args()


input_shape = ...

model_fun = model_options[ARGS.model]
model, loss, metrics = model_fun(input_shape)

model.compile(
    optimizer=optimizers.Adam(),
    loss=loss,
    metrics=metrics,
)


model.fit(...)
