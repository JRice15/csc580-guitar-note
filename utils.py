import numpy as np
import pandas as pd

import soundfile
import sounddevice as sd

def play_audio(data, samplerate):
    sd.play(data, samplerate)
    sd.wait()


