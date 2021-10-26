import numpy as np
import pandas as pd
import os

import soundfile
import sounddevice as sd
import jams
import matplotlib.pyplot as plt
import librosa


MIDI_MIN = 40
MIDI_MAX = 93
FREQ_MIN = librosa.core.midi_to_hz(MIDI_MIN)
FREQ_MAX = librosa.core.midi_to_hz(MIDI_MAX)
DB_MIN = -120.0
DB_MAX = 0.0

def play_audio(data, samplerate):
    sd.play(data, samplerate)
    sd.wait()

def load_audio(audio_path, start, dur):
    sr = librosa.get_samplerate(audio_path)
    data, sr = librosa.load(audio_path, sr=sr, mono=True, offset=start, duration=dur)
    return data, sr

def audio_CQT(audio_path, start, dur):
    """
    Perform the Constant-Q Transform
    from https://towardsdatascience.com/audio-to-guitar-tab-with-deep-learning-d76e12717f81
    args:
        start, dur: in seconds
    """
    sr = librosa.get_samplerate(audio_path)
    data, sr = librosa.load(audio_path, sr=sr, mono=True, offset=start, duration=dur)
    CQT = librosa.cqt(data, sr=sr, hop_length=1024, fmin=FREQ_MIN, n_bins=(MIDI_MAX-MIDI_MIN), bins_per_octave=12)
    CQT_mag = librosa.magphase(CQT)[0]
    # print("\n", CQT_mag.max(), CQT_mag.min(), "\n")
    # CQT_mag = CQT_mag ** 4
    ref = max(CQT_mag.max(), 1.0)
    CQT = librosa.core.amplitude_to_db(CQT_mag, ref=ref) # removed ref=np.amax arg
    # CQT[CQT < -60] = -120
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


def plot_spectrogram(spectro, dur_step, title, outdir, zero_one_scaled=True):
    """
    plot a spectrogram
    args:
        zero_one_scaled: bool
    """
    index = np.arange(MIDI_MIN, MIDI_MAX)
    time_dim = np.linspace(0, dur_step, num=spectro.shape[-1])
    meshX, meshY = np.meshgrid(time_dim, index)
    if zero_one_scaled:
        spectro = (spectro * (DB_MAX - DB_MIN)) + DB_MIN
    m = plt.pcolormesh(meshX, meshY, spectro, vmin=-120, vmax=0, shading="nearest")
    plt.colorbar(m, label="dB")
    plt.xlabel("seconds")
    plt.ylabel("midi note")
    plt.title(title)
    plt.savefig(outdir + title + ".png")
    plt.clf()

def plot_pred_probs_v_groundtruth(gt, pred, title, outdir):
    index = np.arange(MIDI_MIN, MIDI_MAX)
    plt.plot(index, pred, color="blue", label="prediction")
    plt.bar(index, gt, color="green", label="ground-truth")
    plt.ylim(0, 1)
    plt.xlabel("midi mote")
    plt.legend()
    plt.title(title)
    plt.savefig(outdir + title + ".png")
    plt.clf()



TEST_AUDIO = "guitarset/audio_mono-mic/05_BN3-154-E_comp_mic.wav"
aud, sr = load_audio(TEST_AUDIO, 19, 0.2)
play_audio(aud, sr)


# spect = audio_CQT(TEST_AUDIO, 19, 0.2)
# plot_spectrogram(spect, 0.2, "test", "", zero_one_scaled=False)

TEST_JAMS = "guitarset/annotation/05_BN3-154-E_comp.jams"
midi = load_annot_df_from_midi(TEST_JAMS)
print(midi)

midi = midi[midi["time"] <= 19.2]
midi = midi[midi["end"] >= 19.0]
midi["midi"] = np.round(midi["midi"])

print(midi)
