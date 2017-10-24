import math as m
# import numpy as np
import world2sensor as s


def test_rounded_image_parameters():
    cam: s.CamSpec = {
        'x': 0,
        'y': 0,
        'k': 0.1,
        'theta': m.pi/2,
        'alpha': m.pi/2}
    obs: s.Obstacle = {
        'position': {
            'x': 0.105,
            'y': 0.182},
        'velocity': {
            'x': 0,
            'y': 0},
        'radius': 0.064}
    _, result = s._rounded_image_parameters(cam, obs)
    assert isclose(result['x'], int(100 * 0.122 / 0.2))
    assert isclose(result['y'], int(100 * 0.078 / 0.2))


def test_width_of_camera_lens():
    cam: s.CamSpec = {
        'x': 0,
        'y': 0,
        'k': 0.1,
        'theta': m.pi/2,
        'alpha': m.pi/2}
    assert m.isclose(s._width_of_camera_lens(cam), 0.2)


def test_obstacle_image_parameters2():
    cam: s.CamSpec = {
        'x': 0,
        'y': 0,
        'k': 0.1,
        'theta': m.pi/2,
        'alpha': m.pi/2}
    obs: s.Obstacle = {
        'position': {
            'x': 0.103,
            'y': -0.142},
        'velocity': {
            'x': 0,
            'y': 0},
        'radius': 0.05}
    err, result = s._obstacle_image_parameters(cam, obs)
    assert err is not None
    assert result is None


def test_calculate_ABCD_coords():
    cam: s.CamSpec = {
        'x': 0,
        'y': 0,
        'k': 0.1,
        'theta': m.pi/2,
        'alpha': m.pi/2}
    obs: s.Obstacle = {
        'position': {
            'x': 0.103,
            'y': -0.142},
        'velocity': {
            'x': 0,
            'y': 0},
        'radius': 0.05}
    err, result = s._calculate_ABCD_coords(cam, obs)
    assert err is not None
    assert result is None


def test_flatten_points():
    points_in: s.FourPoints = {
        'A': {'x': -0.1, 'y': 0.1},
        'B': {'x': 0.021, 'y': 0.1},
        'C': {'x': 0.11, 'y': 0.1},
        'D': {'x': 0.1, 'y': 0.1}}
    result = s._flatten_points(points_in)
    assert isclose(result['A'], 0)
    assert isclose(result['B'], 0.121)
    assert isclose(result['C'], 0.21)
    assert isclose(result['D'], 0.2)


def test_compare_to_AD():
    A = {'x': -0.1, 'y': 0.1}
    B = {'x': 0.021, 'y': 0.1}
    C = {'x': 0.11, 'y': 0.1}
    D = {'x': 0.1, 'y': 0.1}
    assert isclose(s._compare_to_AD(A, D, A), 0)
    assert isclose(s._compare_to_AD(A, D, B), 0.121)
    assert isclose(s._compare_to_AD(A, D, C), 0.21)
    assert isclose(s._compare_to_AD(A, D, D), 0.2)


def test_calculate_D():
    cam: s.CamSpec = {
        'x': 0,
        'y': 0,
        'k': 0.1,
        'theta': m.pi/2,
        'alpha': m.pi/2}
    D = s._calculate_D(cam)
    assert m.isclose(D['x'], 0.1)
    assert m.isclose(D['y'], 0.1)


def test_calculate_C():
    cam: s.CamSpec = {
        'x': 0,
        'y': 0,
        'k': 0.1,
        'theta': m.pi/2,
        'alpha': m.pi/2}
    obs: s.Obstacle = {
        'position': {
            'x': 0.105,
            'y': 0.182},
        'velocity': {
            'x': 0,
            'y': 0},
        'radius': 0.064}
    _, C = s._calculate_C(cam, obs)
    assert isclose(C['x'], 0.11)
    assert isclose(C['y'], 0.1)


def test_calculate_C2():
    cam: s.CamSpec = {
        'x': 0,
        'y': 0,
        'k': 0.1,
        'theta': m.pi/2,
        'alpha': 0}
    obs: s.Obstacle = {
        'position': {
            'x': 5,
            'y': 0.5},
        'velocity': {
            'x': 0,
            'y': 0},
        'radius': 0.5}
    err, _ = s._calculate_C(cam, obs)
    assert err is not None


def test_calculate_B2():
    cam: s.CamSpec = {
        'x': 0,
        'y': 0,
        'k': 0.1,
        'theta': m.pi/2,
        'alpha': m.pi/2}
    obs: s.Obstacle = {
        'position': {
            'x': 0.103,
            'y': -0.142},
        'velocity': {
            'x': 0,
            'y': 0},
        'radius': 0.05}
    err, B = s._calculate_B(cam, obs)
    assert err is not None


def test_calculate_B():
    cam: s.CamSpec = {
        'x': 0,
        'y': 0,
        'k': 0.1,
        'theta': m.pi/2,
        'alpha': m.pi/2}
    obs: s.Obstacle = {
        'position': {
            'x': 0.105,
            'y': 0.182},
        'velocity': {
            'x': 0,
            'y': 0},
        'radius': 0.064}
    _, B = s._calculate_B(cam, obs)
    assert isclose(B['x'], 0.021)
    assert isclose(B['y'], 0.1)


def test_calculate_A():
    cam: s.CamSpec = {
        'x': 0,
        'y': 0,
        'k': 0.1,
        'theta': m.pi/2,
        'alpha': m.pi/2}
    A = s._calculate_A(cam)
    assert m.isclose(A['x'], -0.1)
    assert m.isclose(A['y'], 0.1)


def isclose(a, b):
    return abs(a - b) < 0.001


# def world1() -> s.WorldState:
#     return {
#         'obstacles': [],
#         'bike': {
#             'v': 0,
#             'psi': 0,
#             'phi': 0,
#             'phidot': 0,
#             'delta': 0,
#             'deltadot': 0,
#             'Tdelta': 0,
#             'Tm': 0,
#             'x': 0,
#             'y': 0}}
#
#
# def world2() -> s.WorldState:
#     return {
#         'obstacles': [
#             {'position': {
#                 'x': 0,
#                 'y': 0.2},
#              'velocity': {'x': 0, 'y': 0},
#              'radius': 0.1}],
#         'bike': {
#             'v': 0,
#             'x': 0,
#             'y': 0,
#             'Tdelta': 0,
#             'Tm': 0,
#             'psi': m.pi / 2,
#             'phi': 0,
#             'phidot': 0,
#             'delta': 0,
#             'deltadot': 0}}
#
#
# def world3() -> s.WorldState:
#     return {
#         'obstacles': [
#             {'position': {
#                 'x': 0.2,
#                 'y': 0},
#              'velocity': {'x': 0, 'y': 0},
#              'radius': 0.1}],
#         'bike': {
#             'v': 0,
#             'psi': m.pi / 2,
#             'phi': 0,
#             'phidot': 0,
#             'delta': 0,
#             'deltadot': 0,
#             'Tdelta': 0,
#             'Tm': 0,
#             'x': 0,
#             'y': 0}}
#
#
# def readings3() -> s.SensorReadings:
#     white_image: 'np.ndarray[np.uint8]' = (  # type: ignore
#         255 * np.ones((100, 100, 3), dtype=np.uint8)
#     right_image = np.concatenate((  # type: ignore
#         255 * np.ones((100, 21, 3), dtype=np.uint8),
#         np.zeros((100, 57, 3), dtype=np.uint8),
#         255 * np.ones((100, 22, 3), dtype=np.uint8)), axis=1)
#     return {
#         'cameras': {
#             'front': white_image,
#             'left': white_image,
#             'right': right_image,
#             'back': white_image},
#         'lean_acceleration': 0,
#         'steer': 0,
#         'velocity': 0,
#         'gps': {'x': 0, 'y': 0}}
#
#
# def readings2() -> s.SensorReadings:
#     white_image: 'np.ndarray[np.uint8]' = (  # type: ignore
#         255 * np.ones((100, 100, 3), dtype=np.uint8))
#     front_image: 'np.ndarray[np.uint8]' = np.concatenate((  # type: ignore
#         255 * np.ones((100, 21, 3), dtype=np.uint8),
#         np.zeros((100, 57, 3), dtype=np.uint8),
#         255 * np.ones((100, 22, 3), dtype=np.uint8)), axis=1)
#     return {
#         'cameras': {
#             'front': front_image,
#             'left': white_image,
#             'right': white_image,
#             'back': white_image},
#         'lean_acceleration': 0,
#         'steer': 0,
#         'velocity': 0,
#         'gps': {'x': 0, 'y': 0}}
#
#
# def readings1() -> s.SensorReadings:
#     white_image: 'np.ndarray[np.uint8]' = (  # type: ignore
#         255 * np.ones((100, 100, 3), dtype=np.uint8))
#     return {
#         'cameras': {
#             'front': white_image,
#             'left': white_image,
#             'right': white_image,
#             'back': white_image},
#         'lean_acceleration': 0,
#         'steer': 0,
#         'velocity': 0,
#         'gps': {'x': 0, 'y': 0}}
#
#
# def equal_nparray(
#         one: 'np.ndarray[np.uint8]',
#         two: 'np.ndarray[np.uint8]'
#         ) -> bool:
#     equalness: 'np.ndarray[bool]' = one == two  # type: ignore
#     answer: bool = equalness.all()  # type: ignore
#     return answer
#
#
# def equal_sensor_state(
#         one: s.SensorReadings,
#         two: s.SensorReadings
#         ) -> bool:
#     onecams: s.ImageSet = one['cameras']
#     twocams: s.ImageSet = two['cameras']
#     if not equal_nparray(onecams['front'], twocams['front']):
#         print('Front image was not correct.')
#         return False
#     if not equal_nparray(onecams['left'], twocams['left']):
#         print('Left-hand image was not correct.')
#         return False
#     if not equal_nparray(onecams['right'], twocams['right']):
#         print('Right-hand image was not correct.')
#         return False
#     if not equal_nparray(onecams['back'], twocams['back']):
#         print('Rear image was not correct.')
#         return False
#     if one['velocity'] != two['velocity']:
#         print('Velocity was not correct.')
#         return False
#     if one['lean_acceleration'] != two['lean_acceleration']:
#         print('Lean acceleration was not correct.')
#         return False
#     if one['steer'] != two['steer']:
#         print('Steer angle was not correct.')
#         return False
#     if one['gps']['x'] != two['gps']['x']:
#         print('GPS x-coordinate was not correct.')
#         return False
#     if one['gps']['y'] != two['gps']['y']:
#         print('GPS y-coordinate was not correct.')
#         return False
#     return True
#
#
# def test_no_obstacles():
#     assert (equal_sensor_state(
#         s.main(world1()),
#         readings1()))
#
#
# def test_one_obstacle_in_front():
#     assert (equal_sensor_state(
#         s.main(world2()),
#         readings2()))
#
#
# def test_one_obstacle_on_left():
#     assert (equal_sensor_state(
#         s.main(world3()),
#         readings3()))
