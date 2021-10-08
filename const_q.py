import os, sys, re
import glob

import numpy as np
import matplotlib.pyplot as plt
import librosa
import jams
import soundfile
import sounddevice as sd

from utils import *

# plt.ion()

def audio_CQT(audio_path, start, dur):
    """
    Perform the Constant-Q Transform
    from https://towardsdatascience.com/audio-to-guitar-tab-with-deep-learning-d76e12717f81
    args:
        start, dur: in seconds
    """
    sr = librosa.get_samplerate(audio_path)
    data, sr = librosa.load(audio_path, sr=sr, mono=True, offset=start, duration=dur)
    CQT = librosa.cqt(data, sr=sr, hop_length=1024, fmin=None, n_bins=96, bins_per_octave=12)
    CQT_mag = librosa.magphase(CQT)[0]**4
    CQT = librosa.core.amplitude_to_db(CQT_mag, ref=np.amax)
    CQT[CQT < -60] = -120
    return CQT

file = "annotation/02_SS1-100-C#_solo.jams"
print(file)
jam = jams.load(file)

# play_audio(data, sr)

annots = jam.search(namespace="pitch_contour")

plt.gcf().set_size_inches(12, 5)
for string_annot in annots:
    df = string_annot.to_dataframe()
    df["freq"] = df["value"].apply(lambda x: x["frequency"])

    plt.plot(df.time, df.freq)

plt.legend(["E", "A", "D", "G", "B", "e"])
plt.show()
