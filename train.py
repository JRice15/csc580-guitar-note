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
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, Input,
                                     MaxPooling2D)
from tensorflow.keras import optimizers, callbacks

import utils
import visualize
from models import get_model
from data_generator import get_generators


parser = argparse.ArgumentParser()
parser.add_argument("--model",required=True)
parser.add_argument("--lr",default=0.001,type=float,help="learning rate")
parser.add_argument("--batchsize",default=16,type=int)
ARGS = parser.parse_args()

train_gen, val_gen, test_gen = get_generators(batchsize=ARGS.batchsize)
input_shape = train_gen.x_shape

model, loss, metrics = get_model(ARGS.model, input_shape)

model.compile(
    optimizer=optimizers.Adam(ARGS.lr),
    loss=loss,
    metrics=metrics,
)

model.summary()

callback_dict = {
    "history": callbacks.History(),
    "model_checkpoint": callbacks.ModelCheckpoint("model.h5", verbose=1, save_best_only=True),
    "earlystopping": callbacks.EarlyStopping(patience=20),
}

model.fit(
    train_gen,
    batchsize=ARGS.batchsize,
    callbacks=list(callback_dict.values()),
    epochs=1000,
)
