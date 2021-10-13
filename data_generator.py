import glob, re, sys
from pathlib import PurePath

import tensorflow as tf
from tensorflow import keras

import numpy as np
import librosa

class SpectogramGenerator(keras.utils.Sequence):

    def __init__(self, ids, dur_step=0.2):
        self.ids = ids
        self.dur_step = dur_step
        # TODO generate set of all possible 0.2-second non-overlapping windows
        # of all files listed in 'ids'
        # ex: [("01_id1...", 0.0), ("01_id1...", 0.2), ("01_id1...", 0.4), ...]

    def __len__(self):
        """
        return number of batches in this generator
        """
        ...

    def __getitem__(self, idx):
        """
        get batch number `idx`
        """
        ...

    def on_epoch_end(self):
        ...



def get_generators(val_split=0.1, **kwargs):
    """
    create train, validation, and test generators
    test set is all files created by player #05
    validation set is a random split of the rest
    args:
        val_split: fraction out of 1
        **kwargs: passed to the generator's init
    """
    files = glob.glob("guitarset/audio_mono-mic/*_mic.wav")
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

get_generators()
