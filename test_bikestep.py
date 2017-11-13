""" It tests the main function from the 'bikestep' module. """


import bikestep as b
import world2sensor as w


bikestate1: w.BikeState = {
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


def test_bikestep1():
    """ It checks that the position changes. """
    i = bikestate1
    o = b.main(i, 3.0)
    assert i['position']['x'] != o['position']['x']
    assert i['position']['y'] != o['position']['y']


bikestate2: w.BikeState = {
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


def test_bikestep2():
    """
    It checks that the bike moves in response to a drive torque.
    """
    o = b.main(bikestate2, 3.0)
    assert o['position']['x'] < 0.0


bikestate3: w.BikeState = {
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


def test_bikestep3():
    """ Similar to test_bikestep2. """
    o = b.main(bikestate3, 1)
    assert o['position']['y'] > 0
