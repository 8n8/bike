import stepworld as s
import world2sensor as w


def worldbefore() -> w.WorldState:
    return {
        'obstacles': [],
        'x': 0,
        'y': 0,
        'velocity': 0,
        'orientation': 0,
        'lean': 0,
        'lean_acceleration': 0,
        'steer': 0}


def test_stepworld():
    assert s.run(worldbefore(), 1, 42) != worldbefore()
