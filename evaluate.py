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

import utils
import visualize
from data_generator import get_generators
from tf_utils import output_model


parser = argparse.ArgumentParser()
parser.add_argument("--name",default="default")
ARGS = parser.parse_args()

matches = glob.glob("models/"+ARGS.name+"-??????-??????/")
# get most recent (by date)
MODEL_DIR = sorted(matches)[-1]

print("\nLoading model stored at", MODEL_DIR)

_, _, test_gen = get_generators()

model = keras.models.load_model(MODEL_DIR+"model.h5")

print("\nEvaluating on test set")
results = model.evaluate(test_gen)

results = {model.metrics_names[i]:v for i,v in enumerate(results)}
print("Results:")
for k,v in results.items():
    print(" ", k+":", v)

with open(MODEL_DIR+"test_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Generating visualizations")
os.makedirs(MODEL_DIR+"visualizations/", exist_ok=True)
# plot 5 random samples
for i in range(0, len(test_gen), len(test_gen)//5):
    x, y = test_gen[i]
    pred = model.predict(x)
    y = np.squeeze(y)
    pred = np.squeeze(pred)
    # print(f"Prediction, Ground-Truth {i}:")
    # print(np.stack((pred, y), axis=-1))
    index = np.arange(len(pred))
    plt.plot(index, pred, color="blue", label="prediction")
    plt.bar(index, y, color="green", label="ground-truth")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(MODEL_DIR+"visualizations/"+str(i)+".png")
    plt.clf()
