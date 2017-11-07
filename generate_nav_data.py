from concurrent.futures import ProcessPoolExecutor as Executor
from mypy_extensions import TypedDict
from nav_net_error import _put_obstacles_in_array
import numpy as np
from typing import List
import update_obstacle_pop as u
import world2sensor as w
from world2sensor import _calculate_rgb_images


class NNData(TypedDict):
    nn_input: 'np.ndarray[np.uint8]'
    nn_output: 'np.ndarray[np.float64]'


def main() -> NNData:
    with Executor() as executor:
        p1 = executor.submit(gen_many)
        p2 = executor.submit(gen_many)
        p3 = executor.submit(gen_many)
        p4 = executor.submit(gen_many)
    dat = p1.result() + p2.result() + p3.result() + p4.result()
    ins = (i['nn_input'] for i in dat)
    outs = (i['nn_output'].flatten() for i in dat)
    return {
        'nn_input': np.stack(ins, axis=0),  # type: ignore
        'nn_output': np.stack(outs, axis=0)}  # type: ignore


def gen_many():
    return [single_data_point() for _ in range(20)]


def _image_dict2np(i: w.ImageSet) -> 'np.ndarray[np.uint8]':
    top_plus_right = np.concatenate(  # type: ignore
        [i['front'], i['right']], axis=0)
    left_plus_bottom = np.concatenate(  # type: ignore
        [i['left'], i['back']], axis=0)
    return np.concatenate([top_plus_right,  # type: ignore
                           left_plus_bottom], axis=1)


def obstacle_is_too_near(o: w.Obstacle) -> bool:
    return (o['position']['x']**2 + o['position']['y']**2) < 2


def delete_near_obstacles(os: List[w.Obstacle]) -> List[w.Obstacle]:
    return [o for o in os if not obstacle_is_too_near(o)]


def single_data_point() -> NNData:
    pos: w.Vector = {'x': 0, 'y': 0}

    def g(os, t):
        return delete_near_obstacles(u.main(os, t, pos))

    obstacles8100 = g([], 1)
    obstacles2700 = g(obstacles8100, 5.4)
    obstacles900 = g(obstacles2700, 1.8)
    obstacles300 = g(obstacles900, 0.6)
    obstacles100 = g(obstacles300, 0.2)

    def f(x):
        return _image_dict2np(_calculate_rgb_images(x, 0, 0, 0))

    images = (
        f(obstacles8100),
        f(obstacles2700),
        f(obstacles900),
        f(obstacles300),
        f(obstacles100))
    nn_input = np.stack(images, axis=2)  # type: ignore
    nn_output = _put_obstacles_in_array(obstacles100)
    return {
        'nn_input': nn_input,
        'nn_output': nn_output}
