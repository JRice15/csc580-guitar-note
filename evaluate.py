import glob
import os
import re
import argparse
from pathlib import PurePath
import sys
import datetime
import json
import pickle
from pprint import pprint

import jams
import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, Input,
                                     MaxPooling2D)
from tensorflow.keras import optimizers, callbacks

from utils import MIDI_MAX, MIDI_MIN
from data_generator import get_generators
from tf_utils import output_model
from custom_layers import CUSTOM_LAYER_DICT


parser = argparse.ArgumentParser()
parser.add_argument("--name",default="default")
ARGS = parser.parse_args()

matches = glob.glob("models/"+ARGS.name+"-??????-??????/")
# get most recent (by date)
MODEL_DIR = sorted(matches)[-1]

print("\nLoading model stored at", MODEL_DIR)

_, _, test_gen = get_generators()

model = keras.models.load_model(MODEL_DIR+"model.h5", custom_objects=CUSTOM_LAYER_DICT)

print("Generating visualizations")
os.makedirs(MODEL_DIR+"visualizations/", exist_ok=True)
# plot 10 random samples
for i in range(0, len(test_gen), len(test_gen)//10):
    x, y = test_gen[i]
    pred = model.predict(x)
    y = np.squeeze(y)
    pred = np.squeeze(pred)
    index = np.arange(len(pred)) + MIDI_MIN
    plt.plot(index, pred, color="blue", label="prediction")
    plt.bar(index, y, color="green", label="ground-truth")
    plt.ylim(0, 1)
    plt.xlabel("Midi Note")
    plt.legend()
    plt.savefig(MODEL_DIR+"visualizations/"+str(i)+"_pred.png")
    plt.clf()
    plt.pcolormesh(np.squeeze(x))
    plt.savefig(MODEL_DIR+"visualizations/"+str(i)+"_input.png")
    plt.clf()


print("\nEvaluating on test set")
results = model.evaluate(test_gen)

results = {model.metrics_names[i]:v for i,v in enumerate(results)}
print("Results:")
for k,v in results.items():
    print(" ", k+":", v)

with open(MODEL_DIR+"test_results.json", "w") as f:
    json.dump(results, f, indent=2)