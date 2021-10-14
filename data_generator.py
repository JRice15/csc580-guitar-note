import glob
import random as rd
import re
import sys
from pathlib import PurePath

import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from utils import audio_CQT

AUDIO_DIR = "guitarset/audio_mono-mic/"
ANNOT_DIR = "guitarset/annotation/"

class SpectogramGenerator(keras.utils.Sequence):

    def __init__(self, ids, dur_step=1.0, batchsize=16):
        self.ids = ids
        self.dur_step = dur_step
        self.batchsize = batchsize

        # training element ids
        self.elems = []
        for id in self.ids:
            dur = librosa.get_duration(filename=AUDIO_DIR+id+"_mic.wav")
            steps = np.arange(0, dur-0.2, 0.2)
            self.elems += [(id, s) for s in steps]
        self.elems = self.elems
        id0 = self.elems[0][0]
        # get data shape
        x = audio_CQT(AUDIO_DIR+id0+"_mic.wav", 0, self.dur_step)
        self.x_shape = x.shape
        self._X_batch = np.empty((self.batchsize,) + x.shape)

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
        for i,(file_id,start_sec) in enumerate(elems):
            x = audio_CQT(AUDIO_DIR+file_id+"_mic.wav", start_sec, self.dur_step)
            self._X_batch[i] = x
            y = ...
            y.append(Y)
        return X, Y

    def on_epoch_end(self):
        rd.shuffle(self.elems)



def get_generators(val_split=0.1, **kwargs):
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
    np.random.shuffle(train_ids)
    val_ids, train_ids = train_ids[:val_size], train_ids[val_size:]
    
    train_gen = SpectogramGenerator(train_ids, **kwargs)
    val_gen = SpectogramGenerator(val_ids, **kwargs)
    test_gen = SpectogramGenerator(test_ids, **kwargs)

    return train_gen, val_gen, test_gen

tr, v, te = get_generators()

print(tr[0])