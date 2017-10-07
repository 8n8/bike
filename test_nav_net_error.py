import nav_net_error as v
import world2sensor as w


def worldstate1() -> w.WorldState:
    return {
        'bike': {
            'v': 0,
            'psi': 0,
            'phi': 0,
            'phidot': 0,
            'delta': 0,
            'deltadot': 0,
            'Tdelta': 0,
            'Tm': 0,
            'x': 0,
            'y': 0},
        'obstacles': []}


def test_main():
    assert v.main(worldstate1(), []) >= 0
