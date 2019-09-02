""" It runs the navigation game. """

import random
import numpy as np  # type: ignore
import game_functions as f
import game_gui as g
import world2sensor as w


blank_image = np.ones((100))
initial_image_set: w.ImageSet = {
    'front': blank_image,
    'left': blank_image,
    'back': blank_image,
    'right': blank_image}


target_velocity = f.Velocity(speed=random.uniform(3, 7), angle=0)


init = f.WorldState(
    crashed=False,
    velocity=target_velocity,
    position={'x': 0, 'y': 0},
    target_velocity=target_velocity,
    obstacles=[],
    keyboard=f.KeyPress.NONE,
    timestamp=0,
    thin_view=blank_image)


g.main(init, 0.03, f.auto_update_world, f.world2view)
