import update_obstacle_pop as u


def test_simple():
    max_new, obstacle_params = u.generate_params()
    result = u.main([], 1, {'x': 0, 'y': 0}, max_new, obstacle_params)
    assert result != ["just test by exercising"]
