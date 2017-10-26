import math as m
# import numpy as np
import world2sensor as s


def test_rounded_image_parameters():
    cam: s.CamSpec = {
        'position': {
            'x': 0,
            'y': 0},
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
    err, result = s._rounded_image_parameters(cam, obs)
    assert err is None
    assert isclose(result['x'], int(100 * 0.122 / 0.2))
    assert isclose(result['y'], int(100 * 0.078 / 0.2))


def test_width_of_camera_lens():
    cam: s.CamSpec = {
        'position': {
            'x': 0,
            'y': 0},
        'k': 0.1,
        'theta': m.pi/2,
        'alpha': m.pi/2}
    assert m.isclose(s._width_of_camera_lens(cam), 0.2)


def test_obstacle_image_parameters2():
    cam: s.CamSpec = {
        'position': {
            'x': 0,
            'y': 0},
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
        'position': {
            'x': 0,
            'y': 0},
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
        'position': {
            'x': 0,
            'y': 0},
        'k': 0.1,
        'theta': m.pi/2,
        'alpha': m.pi/2}
    D = s._calculate_D(cam)
    assert m.isclose(D['x'], 0.1)
    assert m.isclose(D['y'], 0.1)


def test_calculate_C():
    cam: s.CamSpec = {
        'position': {
            'x': 0,
            'y': 0},
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
        'position': {
            'x': 0,
            'y': 0},
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
    err, C = s._calculate_C(cam, obs)
    assert err is None
    assert m.isclose(C['x'], 0.1)
    assert m.isclose(C['y'], 0)


def test_calculate_B2():
    cam: s.CamSpec = {
        'position': {
            'x': 0,
            'y': 0},
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
    assert err is None
    assert isclose(B['x'], -0.136)
    assert isclose(B['y'], 0.1)


def test_calculate_B():
    cam: s.CamSpec = {
        'position': {
            'x': 0,
            'y': 0},
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
        'position': {
            'x': 0,
            'y': 0},
        'k': 0.1,
        'theta': m.pi/2,
        'alpha': m.pi/2}
    A = s._calculate_A(cam)
    assert m.isclose(A['x'], -0.1)
    assert m.isclose(A['y'], 0.1)


def isclose(a, b):
    return abs(a - b) < 0.01
