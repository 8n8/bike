import random
import game_gui as g
import game_functions as f


init = g.WorldState(
    crashed=False,
    velocity=g.Velocity(speed=0, angle=0),
    position={'x': 0, 'y': 0},
    target_velocity=g.Velocity(
        speed=random.uniform(5, 10),
        angle=0),
    obstacles=[],
    keyboard=g.KeyPress.NONE)


g.main(
    init,
    0.01,
    f.update_world,
    f.world2view)
