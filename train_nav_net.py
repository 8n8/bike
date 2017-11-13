""" It provides a function for training the neural network. """

from typing import List, Set, Tuple  # noqa: F401
import os
import json
import conv_net as f
import evaluate_net
import game_gui as g
from keras.optimizers import Adam  # type: ignore
from keras.models import load_model  # type: ignore
import numpy as np


DATA_DIR: str = 'game_data'


def read_one_numpy_file(filename: str) -> g.DataSet:
    """ It reads the training data from a file. """
    dat = np.load(filename)  # type: ignore
    return g.DataSet(
        images=dat['arr_0'],
        target_velocity=dat['arr_1'],
        velocity_in=dat['arr_2'],
        velocity_out=dat['arr_3'])


def read_numpy_data(
        used_data_files: List[str],
        data_file_names: List[str]) -> Tuple[str, List[str], g.DataSet]:
    """
    It reads the training data from file and updates the list of
    used files.
    """
    setunused: Set[str] = set(data_file_names)
    setused: Set[str] = set(used_data_files)
    if setunused == setused:
        return "Data used up.", used_data_files, None
    filename = list(setunused - setused)[0]
    gathered_data = read_one_numpy_file(DATA_DIR + '/' + filename)
    used_data_files.append(filename)
    return None, used_data_files, gathered_data


def main():
    """ It trains the neural network using the game data. """
    data_file_names: List[str] = os.listdir(DATA_DIR)
    saved_net_file: str = 'nav_net.h5'
    used_file_list: str = 'used_files'
    if os.path.isfile(used_file_list):
        with open(used_file_list, 'r') as fff:
            used_data_files: List[str] = json.load(fff)
    else:
        used_data_files: List[str] = []
    if os.path.isfile(saved_net_file):
        model = load_model(saved_net_file)
    else:
        model = f.main()
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=0.001, decay=3e-4),
            metrics=['accuracy'])
    training_cycle_num: int = 0
    while True:
        print('Reading data from files...')
        err, used_data_files, d = read_numpy_data(used_data_files,
                                                  data_file_names)
        if err is not None:
            print(err)
            model.save(saved_net_file)
            with open(used_file_list, 'w') as ff:
                json.dump(used_data_files, ff)
            return
        print("Training cycle {}".format(training_cycle_num))
        training_cycle_num += 1
        model.fit(
            {'image_in': d.images,
             'velocity_in': d.velocity_in,
             'target_in': d.target_velocity},
            d.velocity_out,
            batch_size=1000,
            epochs=1)
        with open(used_file_list, 'w') as ff:
            json.dump(used_data_files, ff)
        print('score: {}'.format(evaluate_net.main(model)))
        model.save(saved_net_file)
