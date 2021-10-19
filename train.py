import glob
import os
import re
import argparse
from pathlib import PurePath
import sys
import datetime
import json
import pickle

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
from tf_utils import output_model


parser = argparse.ArgumentParser()
parser.add_argument("--name",default="default")
parser.add_argument("--model",required=True)
parser.add_argument("--lr",default=0.001,type=float,help="learning rate")
parser.add_argument("--batchsize",default=16,type=int)
ARGS = parser.parse_args()

now = datetime.datetime.now()
timestamp = now.strftime("-%y%m%d-%H%M%S")
MODEL_DIR = "models/"+ARGS.name+timestamp+"/"
os.makedirs(MODEL_DIR)

with open(MODEL_DIR+"args.json", "w") as f:
    json.dump(vars(ARGS), f, indent=2)

train_gen, val_gen, _ = get_generators(train_batchsize=ARGS.batchsize)
input_shape = train_gen.x_shape

model, loss, metrics = get_model(ARGS.model, input_shape)

model.compile(
    optimizer=optimizers.Adam(ARGS.lr),
    loss=loss,
    metrics=metrics,
)

model.summary()
output_model(model, MODEL_DIR)

callback_dict = {
    "history": callbacks.History(),
    "model_checkpoint": callbacks.ModelCheckpoint(MODEL_DIR+"model.h5", 
        verbose=1, save_best_only=True),
    "reducelr": callbacks.ReduceLROnPlateau(factor=0.2, patience=10,
        min_lr=1e-6, verbose=1),
    "earlystopping": callbacks.EarlyStopping(patience=20),
}

try:
    H = model.fit(
        train_gen,
        callbacks=list(callback_dict.values()),
        epochs=1000,
        validation_data=val_gen.load_all(),
    )
except KeyboardInterrupt:
    print("\nTraining ended manually")
    H = callback_dict["history"]

with open(MODEL_DIR+"history.pickle", "wb") as f:
    pickle.dump(H.history, f)

