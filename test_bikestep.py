import bikestep as b
import world2sensor as w


def bikestate1() -> w.BikeState:
    return {
        'v': 0.0,
        'psi': 0.1,
        'phi': 0.0,
        'phidot': 0.0,
        'delta': 0.0,
        'deltadot': 0.0,
        'position': {
            'x': 0.0,
            'y': 0.0},
        'Tdelta': 0.0,
        'Tm': 10.0}


def bikestate2() -> w.BikeState:
    return {
        'v': 0.0,
        'psi': 3.14,
        'phi': 0.0,
        'phidot': 0.0,
        'delta': 0.0,
        'deltadot': 0.0,
        'position': {
            'x': 0.0,
            'y': 0.0},
        'Tdelta': 0.0,
        'Tm': 10.0}


def bikestate3() -> w.BikeState:
    return {
        'v': 0.0,
        'psi': 0.0,
        'phi': 0.0,
        'phidot': 0.0,
        'delta': 0.5,
        'deltadot': 0.0,
        'position': {
            'x': 0.0,
            'y': 0.0},
        'Tdelta': 0.0,
        'Tm': 10.0}


def test_bikestep1():
    i = bikestate1()
    o = b.main(i, 3.0)
    assert i['position']['x'] != o['position']['x']
    assert i['position']['y'] != o['position']['y']


def test_bikestep2():
    i = bikestate2()
    o = b.main(i, 3.0)
    assert o['position']['x'] < 0.0


def test_bikestep3():
    i = bikestate3()
    o = b.main(i, 1)
    assert o['position']['y'] > 0
