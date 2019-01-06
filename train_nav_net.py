""" It provides a function for training the neural network. """

from typing import List, Set, Tuple  # noqa: F401
import os
import json
import conv_net as f
import game_functions as gf
from keras.optimizers import Adam  # type: ignore
from keras.models import load_model  # type: ignore
import numpy as np


DATA_DIR: str = 'game_data'


def read_one_numpy_file(filename: str) -> gf.DataSet:
    """ It reads the training data from a file. """
    dat = np.load(filename)  # type: ignore
    return gf.DataSet(
        images=dat['arr_0'],
        target_velocity=dat['arr_1'],
        velocity_in=dat['arr_2'],
        velocity_out=dat['arr_3'])


def read_numpy_data():
    """
    It reads the training data from file and updates the list of
    used files.
    """
    filenames = os.listdir(DATA_DIR)
    gathered_data = [read_one_numpy_file(DATA_DIR + '/' + filename)
                     for filename in filenames]
    images = [g.images for g in gathered_data]
    target_vs = [g.target_velocity for g in gathered_data]
    vs_in = [g.velocity_in for g in gathered_data]
    vs_out = [g.velocity_out for g in gathered_data]
    return gf.DataSet(
        images=np.concatenate(images, axis=0),
        target_velocity=np.concatenate(target_vs, axis=0),
        velocity_in=np.concatenate(vs_in, axis=0),
        velocity_out=np.concatenate(vs_out, axis=0))



def main():
    """ It trains the neural network using the game data. """
    saved_net_file: str = 'nav_net.h5'
    model = f.main()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=0.0004),
        metrics=['accuracy'])
    print('Reading data from files...')
    d = read_numpy_data()
    np.set_printoptions(threshold=np.nan)
    print(d.velocity_in)
    # model.fit(
    #     {'image_in': d.images,
    #      'velocity_in': d.velocity_in,
    #      'target_in': d.target_velocity},
    #     d.velocity_out,
    #     batch_size=5000,
    #     epochs=10)
    # model.save(saved_net_file)
