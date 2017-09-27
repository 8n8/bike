import bikestep as b
import world2sensor as w


def bikestate1() -> w.BikeState:
    return {
        'v': 0,
        'psi': 0,
        'phi': 0,
        'phidot': 0,
        'delta': 0,
        'deltadot': 0,
        'x': 0,
        'y': 0,
        'Tdelta': 0,
        'Tm': 0}


def test_bikestep():
    assert b.main(bikestate1(), 0.03) is not None
