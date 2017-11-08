import update_obstacle_pop as u


def test_simple():
    params = u.generate_params()
    result = u.main([], 1, {'x': 0, 'y': 0}, params)
    assert result != ["just test by exercising"]
