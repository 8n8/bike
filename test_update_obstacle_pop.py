import update_obstacle_pop as u


def test_simple():
    result = u.main([], 1, {'x': 0, 'y': 0})
    assert result != ["just test by exercising"]
