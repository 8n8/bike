"""
It exposes one function that is used for calculating the error of the
output of the navigation neural net.
"""


import math
from mypy_extensions import TypedDict
import numpy as np  # type: ignore
from typing import (List, Tuple)
import world2sensor as w


XY_RANGE = np.arange(-100, 100, 0.2)  # type: ignore
X_MATRIX, Y_MATRIX = np.meshgrid(XY_RANGE, XY_RANGE)  # type: ignore


# It defines a circle on the ground with the amount of time it will be
# free to drive over.
class Freeness(TypedDict):
    radius: float
    position: w.Point
    free_time: float


# Defines a circle with equation of the form (x - h)² + (y - k)² = r²,
# where (h, k) is the position of the centre and r is the radius.
class Circle(TypedDict):
    o: w.Point
    r: float


# Defines a straight line as a point and an angle.  The angle is measured
# anticlockwise from the positive x-axis.  The reason for not using
# y = mx + c is that this is not defined for vertical lines.
class Line(TypedDict):
    X: w.Point
    theta: float
    tan_theta: float  # gradient


def main(
        obstacles: List[w.Obstacle],
        freenesses: List[Freeness]
        ) -> float:
    """
    It uses the list of obstacles from the world state and the list of
    freenesses produced by the navigation neural net and calculates the
    error.  This error is then used to train the net.  The neural net works
    by placing circular markers of varying sizes on the ground, each
    rated with the time till it is occupied by an obstacle.  These markers
    have been given the name 'freenesses' in this program.  Areas that
    do not have freenesses in them are assumed to be free.

    The approach in this module is to limit the world area that is
    considered to a 200m x 200m square divided up into 20cm grid squares.
    This is represented by a 2D array.  The obstacle paths are mapped onto
    this array by putting the shortest time of arrival of an obstacle in
    each cell and a high value (1000) in all the cells that are not
    travelled over.  The freenesses created by the neural net are mapped
    onto a similarly-shaped array.  The two arrays are then subtracted
    from each other cell by cell, the absolute value is taken for each
    cell, and all the cells are added up to give the final error value.
    """
    obstacle_path_array: 'np.ndarray[np.float64]' = (
        _put_obstacles_in_array(obstacles))
    freeness_array: 'np.ndarray[np.float64]' = (
        _put_freenesses_in_array(freenesses))
    error_array: 'np.ndarray[np.float64]' = np.absolute(  # type: ignore
        obstacle_path_array - freeness_array)
    return np.sum(error_array)  # type: ignore


def _put_freenesses_in_array(
        fs: List[Freeness]
        ) -> 'np.ndarray[np.float64]':
    """
    It loops over the list of freenesses and puts them in the array
    representing the map of the surface.  If they overlap then the one
    with the lower time rating is used in the overlapping area.
    """
    array: 'np.ndarray[np.float64]' = (  # type: ignore
        np.ones((1000, 1000)) * 1000.0)
    for f in fs:
        array = np.minimum(  # type: ignore
            _put_freeness_in_array(f), array)
    return array


def _put_freeness_in_array(f: Freeness) -> 'np.ndarray[np.float32]':
    return (_mark_circle({'o': f['position'],  # type: ignore
                          'r': f['radius']}) * f['free_time'])


def _put_obstacles_in_array(
        os: List[w.Obstacle]
        ) -> 'np.ndarray[np.float64]':
    array: 'np.ndarray[np.float64]' = (  # type: ignore
        np.ones((1000, 1000)) * 1000.0)
    for o in os:
        array = np.minimum(  # type: ignore
            array,
            _put_obstacle_in_array(o))
    return array


def _put_obstacle_in_array(o: w.Obstacle) -> 'np.ndarray[np.float64]':
    # Note that the 1000 is not special.  It was chosen because
    # it is a high number, because of the assumption that unless an
    # area is marked as soon-to-be-occupied it is free for a long time.
    array: 'np.ndarray[np.float64]' = (  # type: ignore
        np.ones((1000, 1000)) * 1000.0)
    current_position: 'np.ndarray[np.bool]' = (  # type: ignore
        _mark_circle({'o': o['position'], 'r': o['radius']}))
    path: 'np.ndarray[np.bool]' = _mark_obstacle_path(o)  # type: ignore
    time_free: 'np.ndarray[np.float64]' = (
        _make_time_free_array(o))
    return (path * time_free +  # type: ignore
            array * ~current_position * ~path)


def _mark_circle(c: Circle) -> 'np.ndarray[np.bool]':  # type: ignore
    distance_from_circle: 'np.ndarray[np.float64]' = (
        (X_MATRIX - c['o']['x'])**2 + (Y_MATRIX - c['o']['y'])**2)
    return distance_from_circle < c['r']**2  # type: ignore


def _squared_distance_from_line(L: Line) -> 'np.ndarray[np.float64]':
    if _isclose(L['theta'] % math.pi, math.pi/2):
        # The line is vertical.
        return np.abs(X_MATRIX - L['X']['x'])  # type: ignore
    if _isclose(L['theta'] % math.pi, 0):
        # The line is horizontal.
        return np.abs(Y_MATRIX - L['X']['y'])  # type: ignore
    # The equation for the distance of a point from a straight line is
    # given on Wikipedia as
    #
    #     | a x_0 + b y_0 + d |
    #     ---------------------
    #        (a² + b²) ^ 0.5
    #
    # where the line is defined by
    #
    #     a x + b y + d = 0
    #
    # and the point is at (x_0, y_0).  Rearranging the equation of the
    # line:
    #
    #     a x + b y + d = 0
    #               b y = - a x + d
    #                 y = - (a/b) x + (d/b)
    #
    # So if the line is defined as y = mx + c then
    #
    #     m = - a / b
    #
    # and
    #
    #     c = d / b
    #
    # Choosing b=1, then
    #
    #     c = d
    #     d = c
    #
    # and
    #
    #     m = - a
    #     a = - m
    _, mxPc = _mx_plus_c(L)
    a = - mxPc['m']
    b = 1
    d = mxPc['c']
    sqrt_a2_plus_b2 = (a**2 + b**2) ** 0.5
    return abs(a * X_MATRIX + Y_MATRIX + d) / sqrt_a2_plus_b2


# def _squared_distance_from_line(L) -> 'np.ndarray[np.float64]':
#     """
#     It makes an array each element of which is the squared distance
#     of that element from the given straight line.
#     """
#     if _isclose(L['theta'] % math.pi, math.pi/2):
#         # The line is vertical.
#         return np.abs(X_MATRIX - L['X']['x'])  # type: ignore
#     if _isclose(L['theta'] % math.pi, 0):
#         # The line is horizontal.
#         return np.abs(Y_MATRIX - L['X']['y'])  # type: ignore
#     #
#     #        ^
#     #        |
#     #        |
#     #        |      (a,b)                        *
#     #        |      X                         *
#     #        |       *                     * y = mx + c
#     #        |     *                    *
#     #        |          * d          *
#     #        |                    *
#     #        |  h *        *   *
#     #        |              *
#     #        |           *
#     #        |  * α   *
#     #        |     *
#     #        |  *  ϴ = arctan(m)
#     #      c X - - - - - - -
#     #        |
#     #        |
#     #        +----------------------------------------->
#     #
#     #
#     #    Knowns:
#     #        ϴ, m, c, a, b
#     #
#     #    Wanted:
#     #        d
#     #
#     #    From the diagram it can be seen that
#     #
#     #        sin α = d / h
#     #            d = h sin α
#     #
#     #        h = (a² + (b - c)²) ^ 0.5
#     #
#     #        sin(α + ϴ) = (b - c) / h
#     #             α + ϴ = arcsin((b - c) / h)
#     #                 α = arcsin((b - c) / h) - ϴ
#
#     # Don't care about the error, since the case of the line being
#     # vertical has been dealt with already.
#     _, mxPc = _mx_plus_c(L)
#     b_c: 'np.ndarray[np.float64]' = Y_MATRIX - mxPc['c']
#     h: 'np.ndarray[np.float64]' = (X_MATRIX**2 + b_c**2)**0.5
#     alpha: 'np.ndarray[np.float64]' = (
#         np.arcsin(b_c / h) - math.atan(mxPc['m']))  # type: ignore
#     d: 'np.ndarray[np.float64]' = h * np.sin(alpha)  # type: ignore
#     return d


def _in_front_of_start(
        o: w.Obstacle) -> 'np.ndarray[np.bool]':  # type: ignore
    """
    It marks the area on the array that is in front of the centre of the
    obstacle in its start position.
    """
    ovx: float = o['velocity']['x']
    ovy: float = o['velocity']['y']
    # relx: 'np.ndarray[np.float64]' = X_MATRIX - o['position']['x']
    # rely: 'np.ndarray[np.float64]' = Y_MATRIX - o['position']['y']
    # return relx * ovx + rely * ovy > 0  # type: ignore
    return ((X_MATRIX - o['position']['x']) * ovx +
            (Y_MATRIX - o['position']['y']) * ovy) > 0


def _mark_obstacle_path(
        o: w.Obstacle
        ) -> 'np.ndarray[np.bool]':  # type: ignore
    """
    It marks the path of the obstacle on the big array that maps the
    surface.
    """
    L1, L2 = _obstacle_path_lines(o)
    d: float = _perpendicular_distance_between_paralell_lines(L1, L2)
    d2: float = d*d
    d1_from_line: 'np.ndarray[np.float64]' = (
        _squared_distance_from_line(L1))
    d2_from_line: 'np.ndarray[np.float64]' = (
        _squared_distance_from_line(L2))
    return ((d1_from_line < d2) * (d2_from_line < d2) *  # type: ignore
            _in_front_of_start(o))


def _velocity_magnitude_squared(v: w.Velocity) -> float:
    return (v['x']**2 + v['y']**2)


def _make_time_free_array(o: w.Obstacle) -> 'np.ndarray[np.float64]':
    """
    It makes a 2D array that represents a map of the ground.  Each
    cell has a number in it that is the amount of time that the obstacle
    would take to get there if it was travelling in that direction.
    """
    distance_from_centre_squared: 'np.ndarray[np.float64]' = (
        (X_MATRIX - o['position']['x'])**2 +
        (Y_MATRIX - o['position']['y'])**2)
    velocity_mag_sq: float = _velocity_magnitude_squared(o['velocity'])
    with np.errstate(divide='ignore'):  # type: ignore
        return np.nan_to_num(velocity_mag_sq   # type: ignore
                             / distance_from_centre_squared)


def _straight_lines_identical(L1: Line, L2: Line) -> bool:
    if _isclose(L1['theta'], L2['theta']):
        # The lines are not parallel so are not identical.
        return False
    err1, L1mc = _mx_plus_c(L1)
    err2, L2mc = _mx_plus_c(L2)
    if err1 is None:
        return _isclose(L1['X']['x'], L2['X']['x'])
    return _isclose(L1mc['c'], L2mc['c'])


def _perpendicular_distance_between_paralell_lines(
        L1: Line,
        L2: Line
        ) -> float:
    """
                         /
                        /                           /
                       /                           /
                      /                           /
                     /                         --X (x2,y2)
                    /                      ---/ /|
                   /                   ---/    / |
                  /            p   ---/       /  |
                 /             ---/          /   |
                /          ---/             /    | u
               /       ---/                /     |
              /    ---/                   /      |
             / ---/           q          / ϴ     |
    (x1,y1) X-/\- - - - - - - - - - - - - - - - -+
           /    ---\                ϴ  /    s
          /         ----\             /
         /            x  ---\        /
        /                    ---\   /
       / ϴ                       ---
      /_ _                        /
                                 /
                                /
                               / ϴ
                              /_ _

    The definition of a line used in this program is a point on the line
    and an angle anticlockwise from the positive x-axis.  Since the lines
    are parallel, the angle is the same for both.  The points are (a,b)
    and (c,d).

    Known: ϴ, x1, y1, x2, y2

    Required: x

    From the diagram it can be seen that if q = 0 then x = 0.  If q is not
    0 then:

        sin(ϴ) = x / q
             x = q sin(ϴ),

        q + s =  (p² - u²)^0.5
            q =  (p² - u²)^0.5 - s,

        u = y2 - y1,

        p = (u² + (x2 - x1)²)^0.5.


    If ϴ == π / 2 then s = 0, else:

        tan(ϴ) = u / s
             s = u / tan(ϴ)
    """
    x1: float = L1['X']['x']
    y1: float = L1['X']['y']
    x2: float = L2['X']['x']
    y2: float = L2['X']['y']
    theta: float = L1['theta']
    if _isclose(abs(theta % math.pi), 0):
        return abs(y2 - y1)
    if _isclose(abs(theta % math.pi), math.pi/2):
        return abs(x2 - x1)
    if _straight_lines_identical(L1, L2):
        return 0
    u: float = y2 - y1
    p: float = (u*u + (x2 - x1)**2)**0.5
    s: float = _calculate_s(theta, u)
    q: float = (p*p - u*u)**0.5 - s
    return q * math.sin(theta)


def _calculate_s(theta: float, u: float) -> float:
    if _isclose(theta, math.pi/2):
        return 0
    return u / math.tan(theta)


def _is_vertical(L: Line) -> bool:
    """ It calculates if the given line is vertical. """
    return _isclose(abs(L['theta']) % math.pi/2, math.pi/2)


class MxPlusC(TypedDict):
    m: float
    c: float


def _mx_plus_c(L: Line) -> Tuple[str, MxPlusC]:
    """
    It calculates the y-intercept and gradient of a straight line.  The
    output is a tuple.  The first element is an error message, or None
    if all is OK.  The second element is the desired value, or None if
    there is an error.
    """
    if _is_vertical(L):
        return "Line is vertical so no y-intercept.", None
    #
    #         ^
    #         |
    #         |           X (x,y)
    #         |          /'
    #         |         / '
    #         |        /  '
    #         |       /   '
    #         |      /    '
    #         |     /     '
    #         |    /      '
    #         |   /       '
    #         |  /        '
    #         | / ϴ       '
    #       c |/_ _ _ _ _ +
    #         |
    #         |
    #         +------------------>
    #
    #
    # tan(ϴ) = (y - c) / x
    #  y - c = x tan(ϴ)
    #      c = y - x tan(ϴ)
    if L['tan_theta'] is not None:
        return (
            None,
            {'m': L['tan_theta'],
             'c': L['X']['y'] - L['X']['x'] * L['tan_theta']})
    return "Line is vertical so no gradient", None


def _distance_between(a: w.Point, b: w.Point) -> float:
    """ It calculates the distance between two points. """
    return ((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2)**0.5


def _obstacle_path_lines(o: w.Obstacle) -> Tuple[Line, Line]:
    """
    It calculates the parameters of the straight lines, line1 and line2,
    that border the path of the obstacle.

                          *
                       * .....                  *
              (e,f) *_-^^^^^^^^^-_           *  B
                 *.X''           ``-.     *
     line2    * .-'                 `-.*
           *   .-'                  *`-.
        *     .-'                *    `-.
     *        ::       (a,b)  * ϴ      ::
              ::           o - - - - + ::           *
              ::        *    *       | ::        *  D
              `-.    *         * r   |.-'     *
               `-.*              * ϴ |-'   *
               *`-.                *.|' *
            *     `-..          ..-' X (c,d)
         *           ^-........-^ *
      *                  ''''  *
                            *
                         *  line1
                      *
                   *
                *
             *


    Knowns:
        a, b, r, velocity of obstacle

    Want to know:
        c, d, ϴ, e, f

    The line of length r between (a, b) and (c, d) is perpendicular to
    lines B and D, and to the velocity of the object.  The velocity of
    the object is known and has components vx and vy.

    If vx != 0 then:
        tan(ϴ) = vy / vx
             ϴ = arctan( vy / vx )
    else:
        ϴ = π / 2
    (or nπ / 2 where n = 2, 3, 4, ... but it doesn't matter because
    the line doesn't have direction).

    From the diagram it can be seen that:
        sin(ϴ) = (c - a) / r
    and
        cos(ϴ) = (b - d) / r
    so
        c - a = r sin(ϴ)
            c = r sin(ϴ) + a
    and
        b - d = r cos(ϴ)
            d = b - r cos(ϴ)
    By symmetry,
        a - e = c - a = r sin(ϴ)
            e = a - r sin(ϴ)
    and
        f - b = b - d = r cos(ϴ)
            f = r cos(ϴ) + b
    """
    vx: float = o['velocity']['x']
    vy: float = o['velocity']['y']
    a: float = o['position']['x']
    b: float = o['position']['y']
    r: float = o['radius']
    theta: float
    if vx != 0:
        theta = math.atan(vy / vx)
    else:
        theta = math.pi / 2
    c: float = r * math.sin(theta)
    d: float = b - r * math.cos(theta)
    e: float = a - r * math.sin(theta)
    f: float = r * math.cos(theta) + b
    tan_theta: float = _tan(theta)
    return (
        {'X': {'x': c, 'y': d},
         'theta': theta,
         'tan_theta': tan_theta},
        {'X': {'x': e, 'y': f},
         'theta': theta,
         'tan_theta': tan_theta})


def _tan(theta: float) -> float:
    if _isclose(abs(theta) % math.pi, math.pi/2):
        return None
    return math.tan(theta)


def _isclose(a: float, b: float) -> bool:
    diff: float = a - b
    return diff < 0.0000001 and diff > -0.0000001
