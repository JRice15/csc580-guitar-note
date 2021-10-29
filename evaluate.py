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
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, Input,
                                     MaxPooling2D)
from tensorflow.keras import optimizers, callbacks

from utils import MIDI_MAX, MIDI_MIN, DB_MAX, DB_MIN, plot_spectrogram, plot_pred_probs_v_groundtruth
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
# plot 10 random samples throughout the dataset
for i in tqdm(range(0, len(test_gen), len(test_gen)//10)):
    x, y, elem_ids = test_gen.get_batch(i)
    elem_id = np.squeeze(elem_ids)
    elem_id_str = str(elem_id[0]) + "_step_" + str(elem_id[1])
    pred = model.predict(x)
    y = np.squeeze(y)
    x = np.squeeze(x)
    pred = np.squeeze(pred)
    # prediction probs vs gt hist
    plot_pred_probs_v_groundtruth(y, pred, elem_id_str+"_pred", MODEL_DIR+"visualizations/")
    # input spectrogram
    plot_spectrogram(x, test_gen.dur_step, elem_id_str+"_input", MODEL_DIR+"visualizations/")


print("\nEvaluating on test set")
results = model.evaluate(test_gen)

if isinstance(results, list):
    results = {model.metrics_names[i]:v for i,v in enumerate(results)}
else:
    results = {"loss": results}
results.update({
    "avg_f1": avg_f1,
    "avg_precision": avg_precision,
    "avg_recall": avg_recall,
})
print("Results:")
for k,v in results.items():
    print(" ", k+":", v)


with open(MODEL_DIR+"test_results.json", "w") as f:
    json.dump(results, f, indent=2)


print("Calculating metrics...")
# precision and recall parameters
# true/false positives, true/false negatives
truepos = 0
trueneg = 0
falsepos = 0
falseneg = 0
for i,(x,y) in tqdm(enumerate(test_gen), total=len(test_gen)):
    pred = model.predict(x)
    y = np.squeeze(y)
    x = np.squeeze(x)
    pred = np.squeeze(pred)
    # calculate prec/recall parameters
    pos = (pred > 0.5)
    ypos = (y > 0.5)
    correct = (ypos == pos)
    truepos += (correct & pos)
    trueneg += (correct & (~pos))
    falsepos += ((~correct) & pos)
    falseneg += ((~correct) & (~pos))


# calculate metrics
old_settings = np.seterr(divide='ignore')
precision = truepos / (truepos + falsepos)
recall = truepos / (truepos + falseneg)
precision = np.nan_to_num(precision, nan=0.0)
recall = np.nan_to_num(recall, nan=0.0)
f1 = (2 * precision * recall) / (precision + recall)
f1 = np.nan_to_num(f1, nan=0.0)
avg_f1 = np.mean(f1)
avg_precision = np.mean(precision)
avg_recall = np.mean(recall)
np.seterr(**old_settings)

# make plot
index = np.arange(MIDI_MIN, MIDI_MAX)
plt.plot(index, precision)
plt.plot(index, recall)
plt.plot(index, f1)
plt.xlabel("midi note")
plt.ylim(0, 1)
plt.legend(["precision", "recall", "f1"])
plt.savefig(MODEL_DIR+"test_set_f1_curve.png")


results.update({
    "avg_f1": avg_f1,
    "avg_precision": avg_precision,
    "avg_recall": avg_recall,
})
print("Results:")
for k,v in results.items():
    print(" ", k+":", v)

with open(MODEL_DIR+"test_results.json", "w") as f:
    json.dump(results, f, indent=2)
