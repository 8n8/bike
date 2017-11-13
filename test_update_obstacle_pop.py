"""
It tests the update_obstacle_pop module.  It's not easy to
test so all it does is run the code.  At least it will find
any exceptions.
"""

import update_obstacle_pop as u


def test_simple():
    """ It just exercises the code. """
    params = u.generate_params()
    result = u.main([], 1, {'x': 0, 'y': 0}, params)
    assert result != ["just test by exercising"]
