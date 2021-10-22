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

from utils import MIDI_MAX, MIDI_MIN, DB_MAX, DB_MIN
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
test_gen.summary()

model = keras.models.load_model(MODEL_DIR+"model.h5", custom_objects=CUSTOM_LAYER_DICT)

print("Generating train metric plots")
os.makedirs(MODEL_DIR+"training/", exist_ok=True)
with open(MODEL_DIR+"history.pickle", "rb") as f:
    history = pickle.load(f)

for metric, values in history.items():
    if not metric.startswith("val_"):
        plt.plot(values)
        plt.plot(history["val_"+metric])
        plt.title(metric)
        plt.legend(["train", "val"])
        plt.savefig(MODEL_DIR+"training/"+metric+".png")
        plt.clf()

print("Generating visualizations")
os.makedirs(MODEL_DIR+"visualizations/", exist_ok=True)
# plot 10 random samples
for i in range(0, len(test_gen), len(test_gen)//10):
    x, y, elem_ids = test_gen.get_batch(i)
    elem_id = np.squeeze(elem_ids)
    elem_id_str = str(elem_id[0]) + " step " + str(elem_id[1])
    pred = model.predict(x)
    y = np.squeeze(y)
    x = np.squeeze(x)
    pred = np.squeeze(pred)
    index = np.arange(len(pred)) + MIDI_MIN
    # prediction probs vs gt hist
    plt.plot(index, pred, color="blue", label="prediction")
    plt.bar(index, y, color="green", label="ground-truth")
    plt.ylim(0, 1)
    plt.xlabel("midi mote")
    plt.legend()
    plt.title(elem_id_str)
    plt.savefig(MODEL_DIR+"visualizations/"+str(i)+"_pred.png")
    plt.clf()
    # input spectrogram
    time_dim = np.linspace(0, test_gen.dur_step, num=x.shape[-1])
    meshX, meshY = np.meshgrid(time_dim, index)
    spectro = (x * (DB_MAX - DB_MIN)) + DB_MIN
    m = plt.pcolormesh(meshX, meshY, spectro, vmin=-120, vmax=0, shading="nearest")
    plt.colorbar(m, label="dB")
    plt.xlabel("seconds")
    plt.ylabel("midi note")
    plt.title(elem_id_str)
    plt.savefig(MODEL_DIR+"visualizations/"+str(i)+"_input.png")
    plt.clf()


print("\nEvaluating on test set")
results = model.evaluate(test_gen)

if isinstance(results, dict):
    results = {model.metrics_names[i]:v for i,v in enumerate(results)}
    print("Results:")
    for k,v in results.items():
        print(" ", k+":", v)
else:
    print(results)

with open(MODEL_DIR+"test_results.json", "w") as f:
    json.dump(results, f, indent=2)