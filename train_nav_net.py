import data_gen_game as g
import func_net as f
import json
from keras.models import load_model  # type: ignore
import math
from mypy_extensions import TypedDict
import numpy as np
import os
from typing import List, Tuple
import world2sensor as w


def read_data_file(filepath: str) -> List[g.DataPoint]:
    with open(filepath, 'r') as f:
        return json.load(f)


def read_data_batch(
        used_data_files: List[str],
        data_file_names: List[str]
        ) -> Tuple[str, List[str], List[g.DataPoint]]:
    if set(data_file_names) == set(used_data_files):
        return "Data used up.", used_data_files, None
    gathered_data: List[g.DataPoint] = []
    for filename in list(set(data_file_names) - set(used_data_files)):
        if len(gathered_data) > 100:
            break
        used_data_files.append(filename)
        gathered_data += read_data_file('game_data/' + filename)
    return None, used_data_files, gathered_data


def _image_dict2np(i: w.ImageSet) -> 'np.ndarray[np.uint8]':
    top_plus_right = np.concatenate(  # type: ignore
        [i['front'], i['right']], axis=0)
    left_plus_bottom = np.concatenate(  # type: ignore
        [i['left'], i['back']], axis=0)
    return np.concatenate([top_plus_right,  # type: ignore
                           left_plus_bottom], axis=1)


def i_for_n_seconds_ago(timestamps, n, now):
    available: List[bool] = [isclose(t, now - n) for t in timestamps]
    for i, val in enumerate(available):
        if val:
            return i
    return None


def isclose(a: float, b: float) -> bool:
    return abs(a - b) < 0.1


def make_batch(
        ims: List['np.ndarray[np.uint8]'],
        timestamps: List[float],
        i: int
        ) -> Tuple[str, 'np.ndarray[np.uint8]']:
    now = timestamps[i]
    times = [8.1, 2.7, 0.9, 0.3, 0.1]
    i_s = [i_for_n_seconds_ago(timestamps, t, now) for t in times]
    nones = [i is None for i in i_s]
    if any(nones):
        return "Not enough history yet to make a batch.", None
    batch = [ims[i] for i in i_s]
    return None, np.stack(batch, axis=2)  # type: ignore


class TrainingData(TypedDict):
    image_in: 'np.ndarray[np.uint8]'
    v_in: 'np.ndarray[np.float64]'
    v_out: 'np.ndarray[np.float64]'


def velocity2array(v: g.Velocity) -> 'np.ndarray[np.float64]':
    return np.array([(v['speed']+10)/20, v['angle']/(2*math.pi)])


def convert_data(
        data_batch: List[g.DataPoint]
        ) -> Tuple[str, TrainingData]:
    def worldstate2images(s: g.WorldState) -> 'np.ndarray[np.uint8]':
        return _image_dict2np(w._calculate_images(
            s['obstacles'],
            s['position']['x'],
            s['position']['y'],
            s['velocity']['angle']))
    ims: List['np.ndarray[np.uint8]'] = [
        worldstate2images(i['world']) for i in data_batch]
    vs: List['np.ndarray[np.float64]'] = [
        velocity2array(i['world']['velocity']) for i in data_batch]
    target_vs: List['np.ndarray[np.float64]'] = [
        velocity2array(i['target_velocity']) for i in data_batch]
    timestamps: List[float] = [i['timestamp'] for i in data_batch]
    numpoints: int = len(timestamps)
    imbatches_with_errors_and_indices = [
        (i, make_batch(ims, timestamps, i)) for i in range(numpoints)]
    i_s = {i for i, (err, _) in imbatches_with_errors_and_indices
           if err is None}
    vs_used = [v for i, v in enumerate(vs) if i in i_s]
    target_vs_used = [t for i, t in enumerate(target_vs) if i in i_s]
    imbatches: List['np.ndarray[np.uint8]'] = [
        i for _, (err, i) in imbatches_with_errors_and_indices
        if err is None]
    if imbatches == []:
        return "No useful data in batch.", None
    image_in: 'np.ndarray[np.uint8]' = (
        np.stack(imbatches, axis=0))  # type: ignore
    v_in: 'np.ndarray[np.float64]' = (
        np.stack(target_vs_used, axis=0))  # type: ignore
    v_out: 'np.ndarray[np.float64]' = np.stack(vs_used, axis=0)  # type: ignore
    return (None, {
        'image_in': image_in,
        'v_in': v_in,
        'v_out': v_out})


def main():
    data_file_names: List[str] = os.listdir('game_data')
    savenetfile = 'nav_net.h5'
    usedfilelistfile = 'used_files'
    if os.path.isfile(usedfilelistfile):
        with open(usedfilelistfile, 'r') as fff:
            used_data_files = json.load(fff)
    else:
        used_data_files = []
    if os.path.isfile(savenetfile):
        model = load_model(savenetfile)
    else:
        model = f.velnet()
        model.compile(
            loss='mean_squared_error',
            optimizer='adam',
            metrics=['accuracy'])
    training_cycle_num = 0
    while True:
        print('Reading data from files...')
        err, used_data_files, data_batch = read_data_batch(
            used_data_files, data_file_names)
        if err is not None:
            print(err)
            model.save(savenetfile)
            if used_data_files == []:
                return
            with open(usedfilelistfile, 'w') as ff:
                json.dump(used_data_files, ff)
            return
        print('Converting data...')
        err, d = convert_data(data_batch)
        if err is not None:
            continue
        print("Training cycle {}".format(training_cycle_num))
        training_cycle_num += 1
        model.fit(
            {'image_in': d['image_in'], 'velocity_in': d['v_in']},
            d['v_out'],
            batch_size=10,
            epochs=1)
        with open(usedfilelistfile, 'w') as ff:
            json.dump(used_data_files, ff)
        model.save(savenetfile)


main()
