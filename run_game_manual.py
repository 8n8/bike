""" It runs the navigation game. """

import random
import numpy as np
import game_gui as g
import game_functions as f
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


g.main(init, 0.03, f.manual_update_world, f.world2view)
