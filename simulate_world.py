import math as m
from mypy_extensions import TypedDict
import numpy as np
from typing import (List, Tuple)


class ImageParameters(TypedDict):
    x: int
    y: int


class Obstacle(TypedDict):
    x: float
    y: float
    radius: float


class CamSpec(TypedDict):
    x: float
    y: float
    k: float
    theta: float
    alpha: float


class Point(TypedDict):
    x: float
    y: float


class WorldState(TypedDict):
    obstacles: List[Obstacle]
    x: float
    y: float
    velocity: float
    orientation: float
    lean: float
    lean_acceleration: float
    steer: float


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


def calculate_sensor_readings(
        world_state: WorldState
        ) -> SensorReadings:
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
            world_state['x'],
            world_state['y'],
            world_state['orientation']),
        'lean_acceleration': world_state['lean_acceleration'],
        'steer': world_state['steer'],
        'velocity': world_state['velocity'],
        'gps': {
            'x': world_state['x'],
            'y': world_state['y']}}


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
        'right': _generic_cam(orientation - m.pi/2, x, y)}


def _generic_cam(
        alpha: float,
        x: float,
        y: float
        ) -> CamSpec:
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
    image_parameter_list: List[ImageParameters] = [
        _obstacle_image_parameters(cam_spec, obstacle)
        for obstacle in obstacle_list]
    return _convert_thin_image_to_thick(_make_thin_composite_image(
        image_parameter_list))


def _convert_thin_image_to_thick(
        thin_im: 'np.ndarray[bool]'
        ) -> 'np.ndarray[np.uint8]':
    """
    It converts a 1D array of bools into a 3D array representing a 2D
    RGB image.  Ones are white, zeros are black.  The original array
    is stretched vertically.
    """
    with_height = np.stack(  # type: ignore
        (thin_im for _ in range(100)), axis=1)
    with_rgb = np.stack(  # type: ignore
        (with_height for _ in range(3)), axis=2)
    as_uint8 = with_rgb.astype(np.uint8)
    return as_uint8 * 255


def _make_thin_composite_image(
        image_parameter_list: List[ImageParameters]
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
        np.ones(int(x), dtype=bool),
        np.zeros(int(y), dtype=bool),
        np.ones(int(100 - x - y), dtype=bool)))


def _obstacle_image_parameters(
        cam: CamSpec,
        obs: Obstacle
        ) -> ImageParameters:
    """
    It takes in the position and orientation of a camera, and the
    position of an obstacle and calculates the parameters needed to
    construct the image in the camera.

    A diagram is shown in ./simulateTrig.pdf.  The required values are
    x and y (shown on the diagram).
    """
    points = _calculate_ABCD_coords(cam, obs)
    A, B, C, D = _flatten_points(points)
    z: float = _width_of_camera_lens(cam)
    x: float = 0
    if C <= B:
        return {
            'x': 0,
            'y': 0}
    if B > 0:
        x = min(z, B)
    y: float = 0
    if D - C < 0:
        y = min(z, D - B)
    else:
        y = min(z, C - B)
    return {
        'x': int(x * 100 / z),
        'y': int(y * 100 / z)}


def _width_of_camera_lens(
        cam: CamSpec
        ) -> float:
    """
    It calculates the width of the camera lens.  See page 6 of
    ./simulateTrig.pdf for details.
    """
    return 2 * cam['k'] * m.tan(cam['theta'] / 2)


FourPoint = Tuple[Point, Point, Point, Point]


def _calculate_ABCD_coords(
        cam: CamSpec,
        obs: Obstacle,
        ) -> FourPoint:
    """
    It calculates the coordinates of the points A, B, C and D, which
    are points along the lens-line of the camera.  They are shown on
    the diagram in ./simulateTrig.pdf.
    """
    return (
        _calculate_A(cam),
        _calculate_B(cam, obs),
        _calculate_C(cam, obs),
        _calculate_D(cam))


def _flatten_points(
        points: FourPoint
        ) -> Tuple[float, float, float, float]:
    """
    A, B, C and D are dictionaries, each containing 'x' and 'y' fields.
    These describe 2D Cartesian coordinates.  A, B, C and D are all on
    one straight line.  This function reduces them to one dimension
    by treating the common line as a real number line with A at 0 and
    D on the positive side of A.
    """
    A, B, C, D = points

    def flatten(point: Point) -> float:
        return _compare_to_AD(A, D, point)
    Bflat: float = flatten(B)
    Cflat: float = flatten(C)
    Dflat: float = flatten(D)
    if Dflat < 0:
        Bflat = -Bflat
        Cflat = -Cflat
        Dflat = -Dflat
    return 0.0, Bflat, Cflat, Dflat


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
    gradient: float = (D['y'] - A['y']) / (D['x'] - A['x'])
    y_intercept: float = A['y'] - gradient * A['x']
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


def _calculate_A(
        cam: CamSpec
        ) -> Point:
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


def _calculate_B(
        cam: CamSpec,
        obs: Obstacle,
        ) -> Point:
    """
    It calculates the position of the intercept of the left-hand
    view-line of the obstacle and the lens line.  See diagram and
    workings in ./simulateTrig.pdf.
    """
    z: float = (
        ((obs['x'] - cam['x'])**2 + (obs['y'] - cam['y'])**2)**0.5)
    delta_x: float = obs['x'] - cam['x']
    phi1: float
    if delta_x == 0:
        phi1 = m.pi / 2
    else:
        phi1 = m.atan((obs['y'] - cam['y']) / (obs['x'] - cam['x']))
    phi2: float = m.asin(obs['radius'] / z)
    phi: float = phi1 + phi2
    u: float = cam['alpha'] - phi
    d1: float = cam['k'] * m.tan(u)
    s: float = (cam['k']**2 + d1**2)**0.5
    x2: float = s * m.cos(phi)
    x3: float = s * m.sin(phi)
    return {
        'x': cam['x'] + x2,
        'y': cam['y'] + x3}


def _calculate_C(
        cam: CamSpec,
        obs: Obstacle,
        ) -> Point:
    """
    It calculates the position of the intercept of the right-hand
    view-line of the obstacle and the lens line.  See diagram and
    workings in ./simulateTrig.pdf.
    """
    x1: float = (
        (obs['x'] - cam['x'])**2 + (obs['y'] - cam['y'])**2)**0.5
    x2: float = (x1**2 - obs['radius']**2)**0.5
    phi2: float = m.asin(obs['radius'] / x1)
    x5: float = obs['y'] - cam['y']
    phi5: float = m.asin(x5 / x1)
    phi1: float = phi5 - phi2
    phi3: float = cam['alpha'] - phi5
    phi4: float = phi2 + phi3
    s: float = cam['k'] / m.cos(phi4)
    x3: float = s * m.cos(phi1)
    x4: float = s * m.sin(phi1)
    return {
        'x': cam['x'] + x3,
        'y': cam['y'] + x4}


def _calculate_D(
        cam: CamSpec
        ) -> Point:
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
