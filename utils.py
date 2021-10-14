import numpy as np
import pandas as pd

import soundfile
import sounddevice as sd
import librosa

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

