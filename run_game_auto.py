""" It runs the navigation game. """

import random
import numpy as np
import game_functions as f
import game_gui as g
import world2sensor as w


blank_image = np.ones((100))
initial_image_set: w.ImageSet = {
    'front': blank_image,
    'left': blank_image,
    'back': blank_image,
    'right': blank_image}


init = f.WorldState(
    crashed=False,
    velocity=f.Velocity(speed=0, angle=0),
    position={'x': 0, 'y': 0},
    target_velocity=f.Velocity(
        speed=random.uniform(5, 10),
        angle=0),
    obstacles=[],
    keyboard=f.KeyPress.NONE,
    timestamp=0,
    thin_view=initial_image_set)


g.main(init, 0.03, f.auto_update_world, f.world2view)
