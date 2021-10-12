import os, sys, re
import glob

import librosa
import jams
import numpy as np
import matplotlib.pyplot as plt


def show_jam(jam, midi=False):
    """
    plot jam pitch over time
    args:
        jam: jams object
        midi: bool, shows raw pitch (frequency) otherwise
    """
    annots = jam.search(namespace="pitch_contour")

    plt.gcf().set_size_inches(12, 5)
    for string_annot in annots:
        df = string_annot.to_dataframe()
        df["freq"] = df["value"].apply(lambda x: x["frequency"])
        df["midi"] = df["value"].apply(
            lambda x: librosa.hz_to_midi(x['frequency']))
        if midi:
            plt.scatter(df.time, df.midi)
        else:
            plt.scatter(df.time, df.freq)

    plt.legend(["E", "A", "D", "G", "B", "e"])
    plt.show()



