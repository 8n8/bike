"""
Given the state of the simulated world, it works out what the
sensor readings should be.
"""


import math as m
from mypy_extensions import TypedDict
import numpy as np
from typing import List, Tuple


class ImageParameters(TypedDict):
    x: float
    y: float


class RoundedImageParameters(TypedDict):
    x: int
    y: int


class Velocity(TypedDict):
    x: float
    y: float


class Point(TypedDict):
    x: float
    y: float


class Obstacle(TypedDict):
    position: Point
    velocity: Velocity
    radius: float


class CamSpec(TypedDict):
    x: float
    y: float
    k: float
    theta: float
    alpha: float


class FourPoints(TypedDict):
    A: Point
    B: Point
    C: Point
    D: Point


class BikeState(TypedDict):
    v: float
    psi: float
    phi: float
    phidot: float
    delta: float
    deltadot: float
    Tdelta: float
    Tm: float
    x: float
    y: float


class WorldState(TypedDict):
    bike: BikeState
    obstacles: List[Obstacle]


class ImageSet(TypedDict):
    front: 'np.ndarray[np.uint8]'
    left: 'np.ndarray[np.uint8]'
    back: 'np.ndarray[np.uint8]'
    right: 'np.ndarray[np.uint8]'


class SensorReadings(TypedDict):
    cameras: ImageSet
    velocity: float
    lean_acceleration: float
    steer: float
    gps: Point


class AllCamSpecs(TypedDict):
    front: CamSpec
    left: CamSpec
    back: CamSpec
    right: CamSpec


def main(world_state: WorldState) -> SensorReadings:
    """
    Given the state of the simulated world, it works out what the
    sensor readings should be.

    It is assumed that the world
    + is perfectly flat and smooth
    + has obstacles moving around in it at constant velocity

    All the obstacles are vertical cylinders to make it easier: they
    look the same from all angles.  It is assumed that the cameras are
    high enough off the ground and the obstacles are tall enough that
    the obstacles always go from top to bottom of the images.  The only
    thing that can change about an image of an object is its width and
    horizontal position.

    It is assumed that a camera is the size of geometric point and that
    all the cameras are at the same place.  Each of the four cameras
    have a viewing angle of 90 degrees and are arranged back-to-back so
    that they collectively view 360 degrees.
    """
    return {
        'cameras': _calculate_images(
            world_state['obstacles'],
            world_state['bike']['x'],
            world_state['bike']['y'],
            world_state['bike']['psi']),
        'lean_acceleration': world_state['bike']['phidot'],
        'steer': world_state['bike']['delta'],
        'velocity': world_state['bike']['v'],
        'gps': {
            'x': world_state['bike']['x'],
            'y': world_state['bike']['y']}}


def _camera_properties(
        x: float,
        y: float,
        orientation: float
        ) -> AllCamSpecs:
    """
    It is assumed that the cameras are all attached at the same point
    on the frame of the bike.
    """
    return {
        'front': _generic_cam(orientation, x, y),
        'left': _generic_cam(orientation + m.pi/2, x, y),
        'back': _generic_cam(orientation + m.pi, x, y),
        'right': _generic_cam(orientation + 3*m.pi/2, x, y)}


def _generic_cam(alpha: float, x: float, y: float) -> CamSpec:
    return {
        'x': x,
        'y': y,
        'k': 0.1,
        'theta': m.pi/2,
        'alpha': alpha}


def _calculate_images(
        obstacles: List[Obstacle],
        x: float,
        y: float,
        orientation: float
        ) -> ImageSet:
    """
    It calculates the image seen by each camera given the list of
    obstacles in the simulated world and the position and orientation
    of the cameras.  Each camera has a viewing angle of 90 degrees
    and there are four of them back-to-back to view 360 degrees.
    """
    cams: AllCamSpecs = _camera_properties(x, y, orientation)
    return {
        'front': _image_of_all_visible_obstacles(
            cams['front'], obstacles),
        'left': _image_of_all_visible_obstacles(
            cams['left'], obstacles),
        'back': _image_of_all_visible_obstacles(
            cams['back'], obstacles),
        'right': _image_of_all_visible_obstacles(
            cams['right'], obstacles)}


def _image_of_all_visible_obstacles(
        cam_spec: CamSpec,
        obstacle_list: List[Obstacle]
        ) -> 'np.ndarray[np.uint8]':
    """
    It makes an image of all the obstacles that are in the view of
    the camera.  The parameters describing the camera are contained
    in variable 'cam_spec'.
    """
    parameter_list_with_errs: List[Tuple[str, RoundedImageParameters]] = [
        _rounded_image_parameters(cam_spec, obstacle)
        for obstacle in obstacle_list]
    image_parameter_list_no_errs: List[RoundedImageParameters] = [
        i[1] for i in parameter_list_with_errs
        if i[0] is None]
    return _convert_thin_image_to_thick(_make_thin_composite_image(
        image_parameter_list_no_errs))


def _convert_thin_image_to_thick(
        thin_im: 'np.ndarray[bool]'
        ) -> 'np.ndarray[np.uint8]':
    """
    It converts a 1D array of bools into a 3D array representing a 2D
    RGB image.  Ones are white, zeros are black.  The original array
    is stretched vertically.
    """
    # The transpose operation is necessary because when later this
    # Numpy array is viewed as an image it comes out rotated by 90
    # degrees without it.
    with_height = np.stack(  # type: ignore
        (thin_im for _ in range(100)), axis=1).T
    with_rgb = np.stack(  # type: ignore
        (with_height for _ in range(3)), axis=2)
    as_uint8 = with_rgb.astype(np.uint8)
    return as_uint8 * 255


def _make_thin_composite_image(
        image_parameter_list: List[RoundedImageParameters]
        ) -> 'np.ndarray[bool]':
    """
    It makes an array of bools, representing the image of all the
    obstacles seen by the camera.  The input is a list of dictionaries,
    each containing 2 parameters that represent the image.  Key 'x'
    is the white gap on the left of the obstacle and key 'y' is the
    width of the obstacle.  The resulting array contains ones for clear
    space and zeros for obstacles.
    """
    im: 'np.ndarray[bool]' = np.ones(100, dtype=bool)
    for i in image_parameter_list:
        im = im * _make_thin_image(i['x'], i['y'])  # type: ignore
    return im


def _make_thin_image(x: float, y: float) -> 'np.ndarray[bool]':
    """
    It makes an array of bools, representing the image of the obstacle
    in the camera.  Since the simulated world is black and white and
    has no vertical variation, a single row of bools contains all the
    information in the image, and can be easily expanded to a proper
    RGB image later.  Clear space is represented by ones and obstacles
    by zeros.
    """
    return np.concatenate((  # type: ignore
        np.ones(x, dtype=bool),
        np.zeros(y, dtype=bool),
        np.ones(100 - x - y, dtype=bool)))


def _rounded_image_parameters(
        cam: CamSpec,
        obs: Obstacle
        ) -> Tuple[str, RoundedImageParameters]:
    z: float = _width_of_camera_lens(cam)

    def n(a):
        return int(a * 100 / z)

    err, parameters = _obstacle_image_parameters(cam, obs)
    if err is not None:
        return err, None
    result =  (None, {
        'x': n(parameters['x']),
        'y': n(parameters['y'])})
    return result


def _obstacle_image_parameters(
        cam: CamSpec,
        obs: Obstacle
        ) -> Tuple[str, ImageParameters]:
    """
    It takes in the position and orientation of a camera, and the
    position of an obstacle and calculates the parameters needed to
    construct the image in the camera.

    A diagram is shown in ./simulateTrig.pdf.  The required values are
    x and y (shown on the diagram).
    """
    print('cam is {}'.format(cam))
    print('obs is {}'.format(obs))
    err, points = _calculate_ABCD_coords(cam, obs)
    if err is not None:
        return err, None
    X = _flatten_points(points)
    print(X)
    A: float = X['A']
    B: float = X['B']
    C: float = X['C']
    D: float = X['D']
    if A <= B and B <= C and C <= D:
        print('a')
        return (None, {
            'x': B - A,
            'y': D - C})
    if B <= A and A <= D and D <= C:
        print('b')
        return (None, {
            'x': 0,
            'y': 100})
    if B <= A and A <= C and C <= D:
        print('c')
        return (None, {
            'x': 0,
            'y': C - A})
    if A <= B and B <= D and D <= C:
        print('d')
        return (None, {
            'x': B - A,
            'y': D - B})
    return (None, {
        'x': 0,
        'y': 0})


def _width_of_camera_lens(cam: CamSpec) -> float:
    """
    It calculates the width of the camera lens.  See page 6 of
    ./simulateTrig.pdf for details.
    """
    return 2 * cam['k'] * m.tan(cam['theta'] / 2)


def _calculate_ABCD_coords(
        cam: CamSpec,
        obs: Obstacle,
        ) -> Tuple['str', FourPoints]:
    """
    It calculates the coordinates of the points A, B, C and D, which
    are points along the lens-line of the camera.  They are shown on
    the diagram in ./simulateTrig.pdf.
    """
    err, B = _calculate_B(cam, obs)
    if err is not None:
        return err, None
    err, C = _calculate_C(cam, obs)
    if err is not None:
        return err, None
    return (None, {
        'A': _calculate_A(cam),
        'B': B,
        'C': C,
        'D': _calculate_D(cam)})


class FlatPoints(TypedDict):
    A: float
    B: float
    C: float
    D: float


def _flatten_points(points: FourPoints) -> FlatPoints:
    """
    A, B, C and D are dictionaries, each containing 'x' and 'y' fields.
    These describe 2D Cartesian coordinates.  A, B, C and D are all on
    one straight line.  This function reduces them to one dimension
    by treating the common line as a real number line with A at 0 and
    D on the positive side of A.
    """
    def flatten(point: Point) -> float:
        return _compare_to_AD(points['A'], points['D'], point)
    Bflat: float = flatten(points['B'])
    Cflat: float = flatten(points['C'])
    Dflat: float = flatten(points['D'])
    if Dflat < 0:
        Bflat = -Bflat
        Cflat = -Cflat
        Dflat = -Dflat
    return {
        'A': 0.0,
        'B': Bflat,
        'C': Cflat,
        'D': Dflat}


def _compare_to_AD(A: Point, D: Point, X: Point) -> float:
    """
    Each of the inputs contains two coordinates describing a point in
    a 2D Cartesian coordinate system.  All three points are on the same
    line.

    The function calculates where point X is on the real number line
    defined by points A and D where A is at zero and D is on the
    positive side of A.

    Let the straight line be y = mx + c where m is the gradient and
    c is the y-intercept.  The method used here is to first translate
    the line downwards by c, then rotate clockwise about the origin by
    arctan(m).  This leaves the length of the line unchanged, but
    positions it on the x-axis.  Then the y-coordinates of A, B, C and
    D can be ignored and the x-coordinates are used as the number line.
    """
    if m.isclose(D['x'], A['x']):
        # The line is vertical.
        return X['y'] - A['y']
    gradient: float = (D['y'] - A['y']) / (D['x'] - A['x'])
    angle: float = m.atan(gradient)
    rotation: 'np.ndarray[float]' = np.array([
        [m.cos(angle), m.sin(angle)],
        [-m.sin(angle), m.cos(angle)]])

    def flatten(p: Point) -> float:
        return np.matmul(  # type: ignore
            rotation,
            np.array([[p['x']], [p['y']]]))[0][0]

    Anew, Xnew = flatten(A), flatten(X)
    return Xnew - Anew


def _calculate_A(cam: CamSpec) -> Point:
    """
    It calculates the position of the left-hand side of the camera
    lens.  See diagram and workings in ./simulatorTrig.pdf.
    """
    p: float = cam['k'] / (m.cos(cam['theta'] / 2))
    beta: float = m.pi - (cam['theta'] / 2) - cam['alpha']
    x1: float = p * m.cos(beta)
    x2: float = p * m.sin(beta)
    return {
        'x': cam['x'] - x1,
        'y': cam['y'] + x2}


def _calculate_B(cam: CamSpec, obs: Obstacle,) -> Tuple[str, Point]:
    """
    It calculates the position of the intercept of the left-hand
    view-line of the obstacle and the lens line.  See diagram and
    workings in ./simulateTrig.pdf.
    """
    z: float = (
        (obs['position']['x'] - cam['x'])**2
        + (obs['position']['y'] - cam['y'])**2)**0.5
    delta_x: float = obs['position']['x'] - cam['x']
    phi1: float
    if delta_x == 0:
        phi1 = m.pi / 2
    else:
        phi1 = (m.atan((obs['position']['y'] - cam['y'])
                / (obs['position']['x'] - cam['x'])))
    phi2: float = m.asin(obs['radius'] / z)
    phi: float = phi1 + phi2
    u: float = cam['alpha'] - phi
    if u > m.pi/2:
        return 'No intersection of lens-line.', None
    d1: float = cam['k'] * m.tan(u)
    s: float = (cam['k']**2 + d1**2)**0.5
    x2: float = s * m.cos(phi)
    x3: float = s * m.sin(phi)
    return (None, {
        'x': cam['x'] + x2,
        'y': cam['y'] + x3})


def _calculate_C(cam: CamSpec, obs: Obstacle) -> Tuple[str, Point]:
    """
    It calculates the position of the intercept of the right-hand
    view-line of the obstacle and the lens line.  See diagram and
    workings in ./simulateTrig.pdf.
    """
    x1: float = ((obs['position']['x'] - cam['x'])**2
                 + (obs['position']['y'] - cam['y'])**2)**0.5
    if m.isclose(x1, 0):
        return 'The obstacle is on the camera.', None
    phi2: float = m.asin(obs['radius'] / x1)
    x5: float = obs['position']['y'] - cam['y']
    phi5: float = m.asin(x5 / x1)
    phi1: float = phi5 - phi2
    phi3: float = cam['alpha'] - phi5
    phi4: float = phi2 + phi3
    print('phi1 is {}'.format(phi1))
    print('phi2 is {}'.format(phi2))
    print('phi3 is {}'.format(phi3))
    print('phi4 is {}'.format(phi4))
    print('phi5 is {}'.format(phi5))
    if phi4 < - m.pi/2:
        return 'No intersection of lens-line.', None
    s: float = cam['k'] / m.cos(phi4)
    x3: float = s * m.cos(phi1)
    x4: float = s * m.sin(phi1)
    return (None, {
        'x': cam['x'] + x3,
        'y': cam['y'] + x4})


def _calculate_D(cam: CamSpec) -> Point:
    """
    It calculates the position of the right-hand end of the camera
    lens.
    """
    p: float = cam['k'] / m.cos(cam['theta'] / 2)
    beta: float = cam['alpha'] - (cam['theta'] / 2)
    a: float = p * m.cos(beta)
    b: float = p * m.sin(beta)
    return {
        'x': cam['x'] + a,
        'y': cam['y'] + b}
