"""
It calculates a score for the neural net.  It runs the game 10 times
till the robot crashes into something and returns the average time.
"""

import random
import game_functions as g
import update_obstacle_pop
import numpy as np
import world2sensor as w


blank_image = np.ones((100))
initial_image_set: w.ImageSet = {
    'front': blank_image,
    'left': blank_image,
    'back': blank_image,
    'right': blank_image}


init = g.WorldState(
    crashed=False,
    velocity=g.Velocity(speed=0, angle=0),
    position={'x': 0, 'y': 0},
    target_velocity=g.Velocity(
        speed=random.uniform(5, 10),
        angle=0),
    obstacles=[],
    keyboard=g.KeyPress.NONE,
    timestamp=0,
    thin_view=initial_image_set)


def run_one_test(model):
    """
    It steers the robot using the neural network until it crashes,
    and returns how long it took.
    """
    history = [init]
    while not history[-1].crashed:
        _, new_state = g.auto_update_world(
            history,
            update_obstacle_pop.generate_params(),
            0.03,
            model)
        history.append(new_state)
    return history[-1].timestamp


def main(model):
    """
    It calculates the mean time till the robot crashes into something.
    """
    times = [run_one_test(model) for _ in range(30)]
    # 23.6s is my average time playing the game manually.
    return round(sum(times) / (23.6 * len(times)), 2)
