"""
It works out what the camera images should be, given the state of the
simulated world.
"""

import math as m
from typing import Any, List, Tuple  # noqa: F401
from mypy_extensions import TypedDict
import numpy as np


class ImageParameters(TypedDict):
    """
    It specifies the view of an obstacle from a camera.
    :param x: The gap on the left of the obstacle.
    :param y: The width of the obstacle.
    """
    x: float
    y: float


class RoundedImageParameters(TypedDict):
    """
    Like the TypedDict ImageParameters, but with ints instead of floats.
    """
    x: int
    y: int


class Vector(TypedDict):
    """ A two-vector, used to represent positions and velocities etc. """
    x: float
    y: float


class Obstacle(TypedDict):
    """
    It represents an obstacle.  An obstacle is a vertical cylinder,
    of infinite height.
    """
    position: Vector
    velocity: Vector
    radius: float


class CamSpec(TypedDict):
    """
    It represents a camera.

            lens
        _____________
        \     |     /
         \    |    /
          \  k|   /
           \  |  /
            \ | /
             \|/
              V

    :param position: The position on the x, y plane of the inner
        corner of the camera.
    :param k: The distance between the inner corner of the camera
        and the lens.
    :param theta: The angle of the camera.
    :param alpha: The global orientation of the centre of the camera
        (the line marked k on the diagram above.
    """
    position: Vector
    k: float
    theta: float
    alpha: float


class FourPoints(TypedDict):
    """
    It represents the positions of four points on the line that goes
    along the camera lens.
    :param A: Left-hand end of lens, looking from above.
    :param D: Right-hand end of lens, looking from above.
    :param B: Intersection of lens-line and line from camera centre to
        left-hand side of obstacle.
    :param C: As with B, but for the right-hand side of the obstacle.
    """
    A: Vector
    B: Vector
    C: Vector
    D: Vector


class SixPoints(TypedDict):
    """
    It represents the positions of six points in the plane.
    :param A, B, C, D: as in FourPoints
    :param P: Furthest left visible point of the obstacle.
    :param N: Furthest right visible point of the obstacle.
    """
    A: Vector
    B: Vector
    C: Vector
    D: Vector
    P: Vector
    Q: Vector


class BikeState(TypedDict):
    """
    It represents the state of the bicycle.
    :param v: The velocity of the back-wheel contact point.
    :param psi: The angle of the main frame with respect to the global
        x-axis.
    :param phi: The lean angle to the right with respect to the vertical.
    :param phidot: The rate of change of phi with respect to time.
    :param delta: The steering angle with respect to straight ahead.  A right
        turn is positive.
    :param deltadot: The rate of change of delta with respect to time.
    :param Tdelta: The torque applied to the steering axis.
    :param Tm: The drive torque.
    :param position: The position of the back-wheel contact point.
    """
    v: float
    psi: float
    phi: float
    phidot: float
    delta: float
    deltadot: float
    Tdelta: float
    Tm: float
    position: Vector


class WorldState(TypedDict):
    """ It represents the state of the world. """
    bike: BikeState
    obstacles: List[Obstacle]


class ImageSet(TypedDict):
    """
    It represents the set of images seen by the cameras at one point
    in time.
    """
    front: 'np.ndarray[Any]'
    left: 'np.ndarray[Any]'
    back: 'np.ndarray[Any]'
    right: 'np.ndarray[Any]'


class AllCamSpecs(TypedDict):
    """ The specifications of the the set of cameras. """
    front: CamSpec
    left: CamSpec
    back: CamSpec
    right: CamSpec


def _camera_properties(x: float, y: float, orientation: float) -> AllCamSpecs:
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
    """ It creates the specification of a camera. """
    return {
        'position': {'x': x, 'y': y},
        'k': 0.1,
        'theta': m.pi/2,
        'alpha': alpha}


def calculate_small_images(
        obstacles: List[Obstacle],
        x: float,
        y: float,
        orientation: float) -> ImageSet:
    """
    It calculates the images of the surroundings for each of the four
    cameras.  Since the world has no vertical variation and is black-and-
    white, these images are expressed as 1 x 100 arrays of bools, with
    0s for obstructions and 1s for free.
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


def calculate_rgb_images(ims: ImageSet) -> ImageSet:
    """
    It converts the set of thin camera images, which are 1 x 100 arrays of
    bools, into square RGB images that are 100 x 100 x 3 arrays of
    unsigned 8-bit integers.
    """
    return {
        'front': _thin_image_to_thick(ims['front']),
        'left': _thin_image_to_thick(ims['left']),
        'back': _thin_image_to_thick(ims['back']),
        'right': _thin_image_to_thick(ims['right'])}


def _image_of_all_visible_obstacles(
        cam_spec: CamSpec,
        obstacle_list: List[Obstacle]) -> 'np.ndarray[Any]':
    """
    It makes an image of all the obstacles that are in the view of
    the camera.  The parameters describing the camera are contained
    in variable 'cam_spec'.
    """
    parameter_list_with_errs: List[Tuple[str, RoundedImageParameters]] = [
        _rounded_image_parameters(cam_spec, obstacle)
        for obstacle in obstacle_list]
    image_parameter_list_no_errs: List[RoundedImageParameters] = [
        i[1] for i in parameter_list_with_errs if i[0] is None]
    return _make_thin_composite_image(image_parameter_list_no_errs)


def _thin_image_to_thick(
        thin_im: 'np.ndarray[bool]') -> 'np.ndarray[np.uint8]':
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


Params = List[RoundedImageParameters]


def _make_thin_composite_image(
        image_parameter_list: Params) -> 'np.ndarray[bool]':
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
        np.ones(x, dtype=bool),  # type: ignore
        np.zeros(y, dtype=bool),  # type: ignore
        np.ones(100 - x - y, dtype=bool)))  # type: ignore


def _rounded_image_parameters(
        cam: CamSpec,
        obs: Obstacle) -> Tuple[str, RoundedImageParameters]:
    """ It makes the image parameters into an int between 0 and 100. """
    z: float = _width_of_camera_lens(cam)

    def n(a):
        """
        It normalises the image parameter to be a number between 0 and
        100.
        """
        return int(a * 100 / z)

    err, parameters = _obstacle_image_parameters(cam, obs)
    if err is not None:
        return err, None
    result: RoundedImageParameters = {
        'x': n(parameters['x']),
        'y': n(parameters['y'])}
    return None, result


def _obstacle_image_parameters(
        cam: CamSpec,
        obs: Obstacle) -> Tuple[str, ImageParameters]:
    """
    It takes in the position and orientation of a camera, and the
    position of an obstacle and calculates the parameters needed to
    construct the image in the camera.

    A diagram is shown in ./simulateTrig.pdf.  The required values are
    x and y (shown on the diagram).
    """
    err, points = _calculate_ABCD_coords(cam, obs)
    if err is not None:
        return err, None
    X = _flatten_points(points)
    A: float = X['A']
    B: float = X['B']
    C: float = X['C']
    D: float = X['D']
    # The alternatives for when the obstacle is in view are:
    #     ABCD -> x = B - A, y = C - B
    #     BACD -> x = 0, y = C - A
    #     BADC -> x = 0, y = 100
    #     ABDC -> x = B - A, y = D - B
    if A <= B and B <= C and C <= D:
        return (None, {
            'x': B - A,
            'y': C - B})
    if B <= A and A <= C and C <= D:
        return (None, {
            'x': 0,
            'y': C - A})
    if B <= A and A <= D and D <= C:
        return (None, {
            'x': 0,
            'y': 100})
    if A <= B and B <= D and D <= C:
        return (None, {
            'x': B - A,
            'y': D - B})
    return "Obstacle is out of sight.", None


def _width_of_camera_lens(cam: CamSpec) -> float:
    """
    It calculates the width of the camera lens.  See page 6 of
    ./simulateTrig.pdf for details.
    """
    return 2 * cam['k'] * m.tan(cam['theta'] / 2)


def _calculate_ABCD_coords(
        cam: CamSpec,
        obs: Obstacle) -> Tuple['str', FourPoints]:
    """
    It calculates the coordinates of the points A, B, C and D, which
    are points along the lens-line of the camera.  They are shown on
    the diagram in ./simulateTrig.pdf.
    """
    rel2cam: SixPoints = _solve_geometry(cam, obs)
    AL: Vector = rel2cam['A']
    BL: Vector = rel2cam['B']
    CL: Vector = rel2cam['C']
    DL: Vector = rel2cam['D']
    PL: Vector = rel2cam['P']
    QL: Vector = rel2cam['Q']
    BxP = BL['x']*PL['y'] - BL['y']*PL['x']
    QxC = QL['x']*CL['y'] - QL['y']*CL['x']
    if BxP < 0 or QxC < 0:
        return "Obstacle is out of sight.", None
    return (None, {
        'A': vectorSum(AL, cam['position']),
        'B': vectorSum(BL, cam['position']),
        'C': vectorSum(CL, cam['position']),
        'D': vectorSum(DL, cam['position'])})


def vectorSum(a: Vector, b: Vector) -> Vector:
    """ It calculates the sum of two two-vectors. """
    return {
        'x': a['x'] + b['x'],
        'y': a['y'] + b['y']}


class FlatPoints(TypedDict):
    """
    It specifies the positions of the four points defined in FourPoints,
    with respect to the line through A and D.  A is at zero and D is on
    the positive side.
    """
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
    def flatten(point: Vector) -> float:
        """
        It calculates the position of the point on the line that passes
        through A and D, with A at zero and D on the positive side of A.
        It assumes that the point is on the line.
        """
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


def _compare_to_AD(A: Vector, D: Vector, X: Vector) -> float:
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
    cosangle = m.cos(angle)
    sinangle = m.sin(angle)
    # The rotated point is found by multiplying it by the rotation
    # matrix:
    #
    # [rot11 rot12] [p1] = [rot11 * p1 + rot12 * p2]
    # [rot21 rot22] [p2]   [rot21 * p1 + rot22 * p2]
    #
    # The y-coordinate is thrown away, so the wanted result is
    #
    #     rot11 * p1 + rot12 * p2
    #
    # In this case:
    #
    #     rot11 = cos(ϴ)
    #     rot12 = sin(ϴ)

    def flatten(p: Vector) -> float:
        """
        It rotates the vector so it lies along the x-axis, and returns its
        x-coordinate.
        """
        return cosangle * p['x'] + sinangle * p['y']

    Anew, Xnew = flatten(A), flatten(X)
    return Xnew - Anew


@profile
def _solve_geometry(cam: CamSpec, obs: Obstacle) -> SixPoints:
    """
    It works out the vectors needed for creating the camera images, using
    the configuration of the camera and the position of the obstacle.

    The vectors are drawn in the diagram in w2s_vector_diagram.pdf. The
    z-axis is perpendicular to the page and positive when pointing towards
    the reader.  The unit vector pointing in the positive z-direction is
    k.  The required vectors are A, B, C, D, P and Q.  All the variables
    with lower-case names are known.

    The equations for finding A and D are:

                A + D - 2K = 0      v1

        cos(ϴ/2) - |K|/|A| = 0      s1
        cos(ϴ/2) - |K|/|D| = 0      s2

    From the diagram, the unknowns needed for finding C are:

        C, G, N, Q

    The equations are:

                C - G - K = 0
        n + Q - N - C - t = 0

                    G . K = 0
                    Q . N = 0
                    Q . C = 0
                      |Q| = r

    Let G + F = S, then the unknowns needed for finding B are:

        B, S, M, P

    The equations are:

                B - S - K = 0
        n + P - M - B - t = 0

                    S . K = 0
                    B . P = 0
                    M . P = 0
                      |P| = r

    These equations were solved using sympy, a symbolic numeric algebra
    library for Python.  The files containing the code are named
    'w2s_solve_for_*.py' where * is the vector name.  The corresponding
    solution files end in 'txt'.
    """
    t1: float = cam['position']['x']
    t12: float = t1**2
    t2: float = cam['position']['y']
    t22: float = t2**2
    n1: float = obs['position']['x']
    n12: float = n1**2
    n2: float = obs['position']['y']
    n22: float = n2**2
    k1: float = cam['k'] * m.cos(cam['alpha'])
    k12: float = k1**2
    k13: float = k1**3
    k2: float = cam['k'] * m.sin(cam['alpha'])
    k22: float = k2**2
    k23: float = k2**3
    r: float = obs['radius']
    r2: float = r**2
    cos_half_theta: float = m.cos(cam['theta']/2)
    sqrt = m.sqrt(n12 - 2*n1*t1 + n22 - 2*n2*t2 - r2 + t12 + t22)
    BCdenominator: float = (
        k12*n12 - 2*k12*n1*t1 - k12*r2 + k12*t12
        + 2*k1*k2*n1*n2 - 2*k1*k2*n1*t2 - 2*k1*k2*n2*t1
        + 2*k1*k2*t1*t2 + k22*n22 - 2*k22*n2*t2
        - k22*r2 + k22*t22)
    return {
        'P': {
            'x': (r*(-n1*r - n2*sqrt + r*t1 + t2*sqrt)
                  / (n12 - 2*n1*t1 + n22 - 2*n2*t2 + t12 + t22)),
            'y': (-r*(r*(n2 - t2) - (n1 - t1)*sqrt)
                  / (n12 - 2*n1*t1 + n22 - 2*n2*t2 + t12 + t22))},
        'Q': {
            'x': (r*(-n1*r + n2*sqrt + r*t1 - t2*sqrt)
                  / (n12 - 2*n1*t1 + n22 - 2*n2*t2 + t12 + t22)),
            'y': (-r*(r*(n2 - t2) + (n1 - t1) * sqrt)
                  / (n12 - 2*n1*t1 + n22 - 2*n2*t2 + t12 + t22))},
        'A': {
            'x': k1 - k2*m.sqrt(-cos_half_theta**2 + 1.0)/cos_half_theta,
            'y': k2 + k1*m.sqrt(-cos_half_theta**2 + 1.0)/cos_half_theta},
        'B': {
            'x': ((k13*n12 - 2*k13*n1*t1 - k13*r2 + k13*t12
                   + k12*k2*n1*n2 - k12*k2*n1*t2 - k12*k2*n2*t1
                   - k12*k2*r
                   * sqrt
                   + k12*k2*t1*t2 + k1*k22*n12 - 2*k1*k22*n1*t1
                   - k1*k22*r2 + k1*k22*t12 + k23*n1*n2
                   - k23*n1*t2 - k23*n2*t1 - k23*r
                   * sqrt
                   + k23*t1*t2)
                  / BCdenominator),
            'y': ((k13*n1*n2 - k13*n1*t2 - k13*n2*t1 + k13*r
                   * sqrt
                   + k13*t1*t2 + k12*k2*n22 - 2*k12*k2*n2*t2
                   - k12*k2*r2 + k12*k2*t22 + k1*k22*n1*n2
                   - k1*k22*n1*t2 - k1*k22*n2*t1 + k1*k22*r
                   * sqrt
                   + k1*k22*t1*t2 + k23*n22 - 2*k23*n2*t2
                   - k23*r2 + k23*t22)
                  / BCdenominator)},
        'C': {
            'x': ((k13*n12 - 2*k13*n1*t1 - k13*r2 + k13*t12
                   + k12*k2*n1*n2 - k12*k2*n1*t2 - k12*k2*n2*t1
                   + k12*k2*r*sqrt
                   + k12*k2*t1*t2 + k1*k22*n12 - 2*k1*k22*n1*t1
                   - k1*k22*r2 + k1*k22*t12 + k23*n1*n2
                   - k23*n1*t2 - k23*n2*t1 + k23*r*sqrt
                   + k23*t1*t2)
                  / BCdenominator),
            'y': ((k13*n1*n2 - k13*n1*t2 - k13*n2*t1 - k13*r
                   * sqrt
                   + k13*t1*t2 + k12*k2*n22 - 2*k12*k2*n2*t2
                   - k12*k2*r2 + k12*k2*t22 + k1*k22*n1*n2
                   - k1*k22*n1*t2 - k1*k22*n2*t1 - k1*k22*r
                   * sqrt
                   + k1*k22*t1*t2 + k23*n22 - 2*k23*n2*t2
                   - k23*r2 + k23*t22)
                  / BCdenominator)},
        'D': {
            'x': k1 + k2*m.sqrt(-cos_half_theta**2 + 1.0)/cos_half_theta,
            'y': k2 - k1*m.sqrt(-cos_half_theta**2 + 1.0)/cos_half_theta}}
