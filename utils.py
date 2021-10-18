import numpy as np
import pandas as pd

import soundfile
import sounddevice as sd
import jams
import matplotlib.pyplot as plt
import librosa


MIDI_MIN = 44
MIDI_MAX = 92


def play_audio(data, samplerate):
    sd.play(data, samplerate)
    sd.wait()

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


def load_annot_df_from_pitchcontour(filename):
    """
    DEPRECATED: the other version from_midi is probably preferred
    load dataframe from a jams filename
    returns:
        dataframe with columns: time, freq, midi, string_num
    """
    jam = jams.load(filename)
    annots = jam.search(namespace="pitch_contour")

    full_df = None
    for i,string_annot in enumerate(annots):
        df = string_annot.to_dataframe()[["value","time"]]
        freq = df["value"].apply(lambda x: x["frequency"])
        midi = freq.apply(lambda x: librosa.hz_to_midi(x))
        df = pd.concat([df["time"], freq, midi], axis=1, keys=["time", "freq", "midi"])
        df["string_num"] = i
        if full_df is None:
            full_df = df
        else:
            full_df = pd.concat((full_df, df))

    full_df = full_df[full_df["freq"] > 0]
    full_df.sort_values(by="time", inplace=True)
    return full_df.reset_index(drop=True)

def load_annot_df_from_midi(filename):
    """
    load dataframe from a jams filename
    returns:
        dataframe with columns: time, start, end, midi
    """
    jam = jams.load(filename)
    annots = jam.search(namespace="note_midi")

    full_df = None
    for i,string_annot in enumerate(annots):
        df = string_annot.to_dataframe()
        df = df.drop(columns=["confidence"])
        df["end"] = df["time"] + df["duration"]
        df["string_num"] = i
        if full_df is None:
            full_df = df
        else:
            full_df = pd.concat((full_df, df))

    # full_df = full_df[full_df["freq"] > 0]
    full_df.sort_values(by="time", inplace=True)
    full_df.rename(columns={"value": "midi"}, inplace=True)
    return full_df.reset_index(drop=True)


