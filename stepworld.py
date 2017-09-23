import world2sensor as w


def run(
        worldstate: w.WorldState,
        timestep: float,
        randseed: int
        ) -> w.WorldState:
    return {
        'obstacles': [],
        'x': 1,
        'y': 1,
        'orientation': 1,
        'velocity': 1,
        'lean': 1,
        'lean_acceleration': 1,
        'steer': 1}
