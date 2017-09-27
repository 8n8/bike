import world2sensor as w


def run(
        worldstate: w.WorldState,
        timestep: float,
        randseed: int
        ) -> w.WorldState:
    return {
        'obstacles': [],
        'bike': {
            'psi': 1,
            'v': 1,
            'phi': 1,
            'phidot': 1,
            'delta': 1,
            'deltadot': 1,
            'Tdelta': 1,
            'Tm': 1,
            'x': 1,
            'y': 1}}
