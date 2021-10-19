import json
import os
import re
import sys

import tensorflow as tf
from tensorflow import keras


def output_model(model, directory):
    """
    print and plot model structure to output dir
    """
    with open(os.path.join(directory, "model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    try:
        tf.keras.utils.plot_model(model, to_file=os.path.join(directory, "model.png"))
    except Exception as e:
        print("Failed to plot model: " + str(e))
