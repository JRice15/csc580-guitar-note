import argparse
import datetime
import glob
import json
import multiprocessing
import os
import pickle
import re
import sys
import curses
from pathlib import PurePath

import jams
import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, optimizers
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, Input,
                                     MaxPooling2D)

from custom_layers import CUSTOM_LAYER_DICT
from models import get_model
from tab_printing import Tabulator
from utils import MIDI_MAX, MIDI_MIN, SAMPLERATE, audio_CQT, const_q_transform


def get_random_tab():
    n_notes = np.random.randint(0, 7)
    strings = np.random.choice(['E', 'A', 'D', 'G', 'B', 'e'], size=n_notes, replace=False)
    out = {}
    for st in strings:
        out[st] = np.random.randint(0, 19)
    return out


def load_model(model_dir):
    return keras.models.load_model(model_dir+"model.h5", custom_objects=CUSTOM_LAYER_DICT)


def live_audio_handler(q, model_dir, output_filename):
    model = load_model(model_dir)
    tabulator = Tabulator()
    midi = np.arange(MIDI_MIN, MIDI_MAX)

    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    try:
        while True:
            audio = q.get()
            spectro = const_q_transform(audio, SAMPLERATE)
            probs = model.predict(spectro[np.newaxis,...])
            predicted = midi[np.squeeze(probs) >= 0.5]
            # TODO get read tabs
            string_fret_map = get_random_tab()
            tabulator.add_timestep(string_fret_map)
            tabulator.output_to_curses(stdscr)
    except KeyboardInterrupt:
        pass
    finally:
        curses.echo()
        curses.nocbreak()
        curses.endwin()
        tabulator.output_to_file(output_filename)


def predict_live(model_dir, ARGS):
    print(sd.query_devices())
    indevice, outdevice = sd.default.device

    sd_params = {
        "device": indevice,
        "channels": 1,
        "samplerate": SAMPLERATE,
    }
    sd.check_input_settings(**sd_params)

    q = multiprocessing.Queue()
    blocksize = int(ARGS.dur_step * SAMPLERATE)

    def callback(outdata, frames, time, status):
        if status:
            print("status:", status)
        q.put_nowait(np.squeeze(outdata))

    # data handler process
    p = multiprocessing.Process(
        target=live_audio_handler, 
        args=(q, model_dir, ARGS.saveas)
    )
    p.start()

    # input reading process
    try:
        with sd.InputStream(callback=callback, blocksize=blocksize, **sd_params):
            p.join()
    except KeyboardInterrupt:
        print("\nProgram ended. Tab saved to '{}'\n".format(ARGS.saveas))



def predict_file(model_dir, ARGS):
    model = load_model(model_dir)
    tabulator = Tabulator()
    print(ARGS.file)
    duration = librosa.get_duration(filename=ARGS.file)
    dur_step = ARGS.dur_step
    midi = np.arange(MIDI_MIN, MIDI_MAX)
    for start in np.arange(0, duration-dur_step, dur_step):
        spectro = audio_CQT(ARGS.file, start, dur_step)
        probs = model.predict(spectro[np.newaxis,...])
        predicted = midi[np.squeeze(probs) >= 0.5]
        # TODO get read tabs
        string_fret_map = get_random_tab()
        tabulator.add_timestep(string_fret_map)
    tabulator.output_to_file(ARGS.saveas)
    print("\nTab saved to '{}'\n".format(ARGS.saveas))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",default="default",help="model name to use")
    parser.add_argument("--saveas",default="tab.txt")
    inpt_grp = parser.add_mutually_exclusive_group(required=True)
    inpt_grp.add_argument("--file",help="input filename")
    inpt_grp.add_argument("--live",action="store_true",help="flag to record live")

    ARGS = parser.parse_args()

    matches = glob.glob("models/"+ARGS.name+"-??????-??????/")
    # get most recent (by date)
    MODEL_DIR = sorted(matches)[-1]

    with open(MODEL_DIR+"args.json", "r") as f:
        saved_args = json.load(f)
    for k,v in saved_args.items():
        setattr(ARGS, k, v)

    if ARGS.live:
        predict_live(MODEL_DIR, ARGS)
    else:
        predict_file(MODEL_DIR, ARGS)


if __name__ == "__main__":
    main()