import stepworld as s
import world2sensor as w


def worldbefore() -> w.WorldState:
    return {
        'obstacles': [],
        'bike': {
            'x': 0,
            'y': 0,
            'v': 0,
            'psi': 0,
            'phi': 0,
            'phidot': 0,
            'delta': 0,
            'deltadot': 0,
            'Tdelta': 0,
            'Tm': 0}}


def test_stepworld():
    assert s.run(worldbefore(), 1, 42) != worldbefore()
