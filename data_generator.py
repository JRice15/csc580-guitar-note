import glob
import os
import random as rd
import re
import sys
from pathlib import PurePath

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from utils import (MIDI_MAX, MIDI_MIN, audio_CQT, load_annot_df_from_midi,
                   play_audio, DB_MAX, DB_MIN, format_sec_strs)

AUDIO_DIR = "guitarset/audio_mono-mic/"
ANNOT_DIR = "guitarset/annotation/"



class SpectogramGenerator(keras.utils.Sequence):

    def __init__(self, ids, name, dur_step=0.2, batchsize=16, random_seed=10):
        self.name = name
        print("Loading", self.name, "dataset")
        self.ids = ids
        self.dur_step = dur_step
        self.batchsize = batchsize
        # random number generator
        self.rng = np.random.default_rng(seed=random_seed)

        # training element ids
        self.elems = []
        for id in self.ids:
            dur = librosa.get_duration(filename=AUDIO_DIR+id+"_mic.wav")
            steps = format_sec_strs(np.arange(0, dur-dur_step, dur_step))
            self.elems += [(id, s) for s in steps]
        self.elems = self.elems
        self.rng.shuffle(self.elems)

        # get ground-truth files
        self.load_jams_to_df()

        # get data shape
        id0 = self.elems[0][0]
        x = audio_CQT(AUDIO_DIR+id0+"_mic.wav", 0, self.dur_step)
        self.x_shape = x.shape
        self._X_batch = np.empty((self.batchsize,) + x.shape)
        self._Y_batch = np.empty((self.batchsize, MIDI_MAX - MIDI_MIN))

    def load_jams_to_df(self):
        os.makedirs("data", exist_ok=True)
        save_file = "data/"+self.name+"_annotations_step"+str(self.dur_step)+"_df.hdf5"
        if os.path.exists(save_file):
            self.annot_df = pd.read_hdf(save_file, key="df")
        else:
            print("  Generating ground-truth dataframe")
            dfs = []
            for i,file_id in enumerate(self.ids):
                if i % 20 == 0:
                    print(" ", i, "of", len(self.ids))
                df = load_annot_df_from_midi(ANNOT_DIR+file_id+".jams")
                df["round_start"] = np.floor(df["time"] / self.dur_step) * self.dur_step
                df["round_end"] = np.ceil(df["end"] / self.dur_step) * self.dur_step
                df["round_midi"] = df["midi"].round().astype(int)
                new_rows = []
                for row in df.itertuples():
                    steps = np.arange(row.round_start, row.round_end, self.dur_step)
                    steps = format_sec_strs(steps)
                    for step in steps:
                        if MIDI_MIN <= row.round_midi <= MIDI_MAX:
                            new_rows.append([
                                step, row.string_num, row.round_midi
                            ])
                df = pd.DataFrame(new_rows, columns=["sec", "string_num", "midi"])
                df = df.sort_values("sec")
                df = df.set_index("sec")
                dfs.append(df)

            full = pd.concat(dfs, axis=0, keys=self.ids)
            full.to_hdf(save_file, key="df")
            print("  done")
            self.annot_df = full

    def __len__(self):
        """
        return number of batches in this generator
        """
        return len(self.elems) // self.batchsize

    def __getitem__(self, idx):
        """
        get batch of number `idx`
        """
        elems = self.elems[idx*self.batchsize:(idx+1)*self.batchsize]
        self._Y_batch.fill(0)
        for i,elem in enumerate(elems):
            file_id, sec = elem
            x = audio_CQT(AUDIO_DIR+file_id+"_mic.wav", float(sec), self.dur_step)
            self._X_batch[i] = x
            if elem in self.annot_df.index:
                notes = self.annot_df.loc[elem]
                self._Y_batch[i][notes["midi"] - MIDI_MIN] = 1
        self._X_batch = (self._X_batch - DB_MIN) / (DB_MAX - DB_MIN)
        return self._X_batch, self._Y_batch

    def get_batch(self, idx):
        """
        return X, Y, and file/timestamp IDs
        """
        x, y = self.__getitem__(idx)
        elems = self.elems[idx*self.batchsize:(idx+1)*self.batchsize]
        return x, y, elems

    def on_epoch_end(self):
        self.rng.shuffle(self.elems)

    def load_all(self):
        X = None
        Y = None
        for i in range(len(self)):
            x, y = self[i]
            if X is None:
                X = x
                Y = y
            else:
                X = np.concatenate((X, x))
                Y = np.concatenate((Y, y))
        return X, Y

    def summary(self):
        print("DataGenerator:", self.name)
        print(" ", len(self.ids), "files,", len(self.elems), "individual examples")
        print("  batchsize", self.batchsize, "results in", len(self.elems)//self.batchsize, "batches")


def get_generators(val_split=0.1, train_batchsize=16, **kwargs):
    """
    create train, validation, and test generators
    test set is all files created by player #05
    validation set is a random split of the rest
    args:
        val_split: fraction out of 1
        **kwargs: passed to the generator's init
    """
    files = glob.glob(ANNOT_DIR+"*.jams")
    ids = [PurePath(i).stem.strip("_mic") for i in files]
    val_size = int(len(ids) * val_split)
    test_ids = np.array([i for i in ids if i.startswith("05")])
    train_ids = np.array([i for i in ids if not i.startswith("05")])
    # seeded rng so we get the same validation set every time
    rng = np.random.default_rng(seed=94958)
    rng.shuffle(train_ids)
    val_ids, train_ids = train_ids[:val_size], train_ids[val_size:]
    
    train_gen = SpectogramGenerator(train_ids, name="train", random_seed=324, 
                    batchsize=train_batchsize, **kwargs)
    val_gen = SpectogramGenerator(val_ids, name="val", random_seed=943, 
                    batchsize=1, **kwargs)
    test_gen = SpectogramGenerator(test_ids, name="test", random_seed=860, 
                    batchsize=1, **kwargs)

    return train_gen, val_gen, test_gen

if __name__ == "__main__":
    tr, v, te = get_generators()
    tr.summary()
    v.summary()
    te.summary()

    print("Train set annotation dataframe:")
    print(tr.annot_df)

    TEST_JAMS = "guitarset/annotation/05_BN2-131-B_comp.jams"
    midi = load_annot_df_from_midi(TEST_JAMS)

    midi = midi[midi["time"] <= 11.8]
    midi = midi[midi["end"] >= 11.6]
    midi["midi"] = np.round(midi["midi"])

    print("Jams midi:")
    print(midi)

    print("TestGen midi:")
    elem = ("05_BN2-131-B_comp", "11.60")
    print(te.annot_df.loc[elem])
