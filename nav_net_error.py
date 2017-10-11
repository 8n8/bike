from mypy_extensions import TypedDict
import math
import numpy as np
from typing import (List, Tuple)
import world2sensor as w


# It defines a circle on the ground with the amount of time it will be
# free to drive over.
class Freeness(TypedDict):
    radius: float
    position: w.Point
    free_time: float


def main(
        obstacles: List[w.Obstacle],
        freenesses: List[Freeness]
        ) -> float:
    obstacle_path_array: 'np.ndarray[np.float32]' =
        _put_obstacles_in_array(obstacles)
    freeness_array: 'np.ndarray[np.float32]' =
        _put_freenesses_in_array(freenesses)
    error_array = np.absolute(obstacle_path_array - freeness_array)
    return np.sum(error_array)


def _put_obstacles_in_array(
        os: List[w.Obstacle]
        ) -> 'np.ndarray[np.float32]':


def _put_obstacle_in_array(o: w.Obstacle) -> 'np.ndarray[np.float32]':
    blank = np.zeros((1000, 1000))
    path1, path2 = _obstacle_path_lines(o)
    perpendicular = _line_right_angle_to_path(o)



def coordinate_in_rect_path(
        x: float,
        y: float,
        o: w.Obstacle,
        path1: Line,
        path2: Line,
        perpendicular: Line) -> bool:
    """
    The function gives 'True' if (x, y) is in the area shaded by X's
    and false else:
    
                        :
                        : perpendicular
                        :
                        :
                       ___  _  _  _  _  _  _  _  _  _  path1
                     /  :  \ XXXXXXXXXXXXXXXXXXXXXXXXX
            ----> v |   o   | XXXXXXXXXXXXXXXXXXXXXXXX
                     \  :  / XXXXXXXXXXXXXXXXXXXXXXXXX
              obstacle ---  -  -  -  -  -  -  -  -  -  path2
                        :
                        :
                        :
                        :


    The lines marked 'path1' and 'path2' are the straight lines at the
    edges of the area travelled over by the obstacle as it moves along.
    The line marked 'perpendicular' is the line that is perpendicular
    to the path of the obstacle, and passes through the centre of it
    when it is at the start position.  The arrow marked 'v' denotes
    the velocity of the obstacle.
    """
    perpendicular_test: bool
    if o['velocity']['y'] 
    if o['velocity']['y'] > 0 and path1['c'] < path2['c']:
        return (
            y > path1['m'] * x + path1['c'] and
            y < path2['m'] * x + path2['c'] and
            y > perpendicular['m'] * x + perpendicular['c'])
    if o['velocity']['y'] < 0 and path1['c'] < path2['c']:
        return (
            y > path1['m'] * x + path1['c'] and
            y < path2['m'] * x + path2['c'] and
            y < perpendicular['m'] * x + perpendicular['c'])
    if o['velocity']['y'] > 0 and path1['c'] > path2['c']:
        return (
            y < path1['m'] * x + path1['c'] and
            y > path2['m'] * x + path2['c'] and
            y > perpendicular['m'] * x + perpendicular['c'])
    if o['velocity']['y'] < 0 and path1['c'] < path2['c']:
        return (
            y > path1['m'] * x + path1['c'] and
            y < path2['m'] * x + path2['c'] and
            y < perpendicular['m'] * x + perpendicular['c'])
            
    

def _line_right_angle_to_path(o: w.Obstacle) -> Line:
    vx = o['velocity']['x']
    vy = o['velocity']['y']
    # It's x-component over y-component because I want the line that
    # is perpendicular to the velocity.
    m = vx / vy 
    #
    #
    #          y ^
    #            |                 
    #    *       |                             /
    #         *  |                            /        
    #          c | *           1             /       /
    #            |      * - - - - - +       /       /
    #            |           * ϴ    '-m    /       /
    #            |-+              * '      ___    /
    #            | |              ϴ    * /     \  
    #          b |- - - - - - - - - - - ::  o  :: obstacle, 
    #            |                       \  '  /*   centre at (a, b)
    #            |                    /    ---       *
    #            |                   /      /             *
    #            |                  /      /'                  * 
    #            |                          '                       *
    #          0 +--------------------------------------------------->
    #            0                          a                        x
    #
    #    From the diagram:
    #        tan(ϴ) = m
    #    and
    #        tan(ϴ) = (c - b) / a
    #         c - b = a tan(ϴ)
    #             c = a tan(ϴ) + b
    #               = a m + b
    #
    a = o['position']['x']
    b = o['position']['y']
    return {
        'm': m,
        'c': a * m + b }


def _intersecting_obstacles_err(
        os: List[w.Obstacle],
        fs: List[Freeness]
        ) -> float:
    """
    It works out the error for all the obstacles that have paths that
    intersect freenesses.  This function loops over all the freenesses
    for each obstacle and works out the 
    """



def _non_intersecting_freenesses_err(fs: List[Freeness]) -> float:
    """
    It calculates the error for all the freenesses that do not intersect
    the paths of obstacles.  This type of freenesses are always erroneous
    because the whole point of a freeness is that it marks places that
    will soon not be free.  The error is bigger when the area of the
    freeness is greater and bigger when the rating of the freeness is
    smaller.
    """
    return sum((math.pi * f['radius']**2 / f['free_time']  for f in fs))


def _non_intersecting_obstacles_err(os: List[w.Obstacles]) -> float:
    """
    It calculates the error for all the obstacles with paths that do not
    intersect freenesses.
        
       time till occupied
        ^
        |
        |     default time till occupied
        |        (as high as possible)
      d | - - - - - - - - - - - - - - - - - - -
        |                              ' 
        |                              ' 
      k |   -    error area (a)   -    *
        |                          *   '
        |                      *       '
        |                  *           '
        |              * actual time   '
        |          * till occupied     '
        |      * curve                 '  distance from 
        |  *                           '  start position
      0 +---------------------------------------------->
        0                              |         
                                     100 m
                           (max distance considered)
    
    From the diagram, it can be seen that
        a = 100d - (100k/2)
    where
        k = 100/v
        v = velocity of obstacle
    so 
        a = 100d - (100 (100/v) / 2)
          = 100d - (5000 / v)
          = 100 ( d - (50 / v) )
    """
    d: float = 1000_000.0
    return sum((100 * (d - (50 / magnitude(o['velocity']))) for o in os))


def magnitude(v: w.Velocity) -> float:
    return (v['x']**2 + v['y']**2)**0.5


def _compare_intersecting(
        os: List[w.Obstacle],
        fs: List[Freeness]
        ) -> float
    


def _freeness_intersects_an_obstacle(
        os: List[w.Obstacle],
        f: Freeness
        ) -> bool
   return any((_obstacle_intersects_freeness(o, f) for o in os))  


def _obstacle_intersects_a_freeness(
        o: w.Obstacle,
        fs: List[Freeness]
        ) -> bool:
    return any((_obstacle_intersects_freeness(o, f) for f in fs))


def _distance_between(a: w.Point, b: w.Point) -> float:
    """ It calculates the distance between two points. """
    return ((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2)**0.5


def _obstacle_currently_intersects_freeness(
        o: w.Obstacle,
        f: Freeness
        ) -> bool:
    """
    It works out if the obstacle in its current position intersects
    the freeness.
    """
    return (o['radius'] + f['radius'] >
            _distance_between(o['position'], f['position']))


def _obstacle_going_towards_freeness(o: w.Obstacle, f: Freeness) -> bool:
    """
    It works out if the obstacle is travelling towards the freeness.
    """
    relative_position = {
        'x': f['position']['x'] - o['position']['x'],
        'y': f['position']['y'] - o['position']['y']}
    dot_product_of_rel_pos_vel = (
        relative_position['x'] * o['velocity']['x'] +
        relative_position['y'] * o['velocity']['y'])
    return dot_product_of_rel_pos_vel > 0


# Defines a straight line with equation of the form y = mx + c where m is
# the gradient and c is the y-intercept.
class Line(TypedDict):
    m: float
    c: float


# Defines a circle with equation of the form (x - h)² + (y - k)² = r²,
# where (h, k) is the position of the centre and r is the radius.
class Circle(TypedDict):
    h: float
    k: float
    r: float


class ComplexPoint(TypedDict):
    x: complex
    y: complex


def _intersections_of_line_and_circle(
        L: Line,
        C: Circle
        ) -> Tuple[ComplexPoint, ComplexPoint]:
    """
    The computer algebra program wxmaxima was used to find the
    intersections of a straight line and a circle, as follows:
    (%i1)	eq1: (x-h)^2 + (y-k)^2 = r^2;
    (eq1)	(y-k)^2+(x-h)^2=r^2
    (%i2)	eq2: y = m*x+c;
    (eq2)	y=m*x+c
    (%i3)	solve([eq1, eq2], [x, y]);
    (%o3)	[[x=(-sqrt((m^2+1)*r^2-h^2*m^2+k*(2*h*m+2*c)-2*c*h*m-k^2-c^2)
                 +k*m-c*m+h)
                /(m^2+1),
              y=(-m*sqrt((m^2+1)*r^2-h^2*m^2+k*(2*h*m+2*c)-2*c*h*m-k^2-c^2)
                 +k*m^2+h*m+c)
                /(m^2+1)],
             [x=(sqrt((m^2+1)*r^2-h^2*m^2+k*(2*h*m+2*c)-2*c*h*m-k^2-c^2)
                 +k*m-c*m+h)
                /(m^2+1),
              y=(m*sqrt((m^2+1)*r^2-h^2*m^2+k*(2*h*m+2*c)-2*c*h*m-k^2-c^2)
                 +k*m^2+h*m+c)
                /(m^2+1)]]
    """
    c: float = L['c']
    m: float = L['m']
    h: float = C['h']
    k: float = C['k']
    r: float = C['r']
    c2: float = c*c
    m2: float = m*m
    h2: float = h*h
    k2: float = k*k
    r2: float = r*r
    big_square_root: complex = ((m2 + 1)*r2 - h2*m2 + k*(2*h*m + 2*c) -
                                2*c*h*m - k2 - c2)**0.5
    denominator: float = m2 + 1
    kxm_cxmPh: float = k * m - c * m + h
    kxm2PhxmPc: float = k * m2 + h * m + c
    x1: complex = (- big_square_root + kxm_cxmPh) / denominator
    y1: complex = (- m * big_square_root + kxm2PhxmPc) / denominator
    x2: complex = (big_square_root + kxm_cxmPh) / denominator
    y2: complex = (m * big_square_root + kxm2PhxmPc) / denominator
    return (
        {'x': x1, 'y': y1},
        {'x': x2, 'y': y2})


def _line_intersects_circle(L: Line, C: Circle) -> bool:
    X1, X2 = _intersections_of_line_and_circle(L, C)
    # Case where there are no intersections.  From the formula, if one
    # of the four coordinates of the two coordinates is imaginary then
    # they all are, so there are no intersections.
    if math.isclose(X1['x'].imag, 0):
        return False
    # Case where the line just touches the circle at a tangent.  This is
    # a false because I am only interested when the areas actually
    # overlap.
    if (math.isclose(X1['x'].real, X2['x'].real) and
            math.isclose(X1['y'].real, X2['y'].real)):
        return False
    return True


def _obstacle_path_lines(o: w.Obstacle) -> Tuple[Line, Line]:
    """
    It calculates the equations of the straight lines that mark the
    edges of the path of the obstacle.  Let these lines be of the form
    y = mx + c then this function calculates m, c1 and c3.

              y ^
                |                 *___
                |              */      \ obstacle (plan view)
                |           *  (   *o   )  centre at (a, b)
                |        *      *      /*
                |     *      *    `--*
                |  *      *       *
            c1 -*      *       * |
                |   *       *    | m
            c2 _|*       * ϴ     |
                |     *- - - - - +
                |  *        1
            c3 -*
                |
                +--------------------------> x

    """
    vx: float = o['velocity']['x']
    vy: float = o['velocity']['y']
    m: float = vy / vx
    theta: float = math.tan(m)
    # Since:
    #     tan(ϴ) = (b - c2) / a
    # then:
    #     b - c2 = a tan(ϴ)
    #     - c2 = a tan(ϴ) - b
    #     c2 = b - a tan(ϴ)
    c2: float = vy - vx * math.tan(theta)
    # Calculations for c3:
    #
    #                 .....
    #             _-^^^^^^^^^-_
    #          .-''           ``-.
    #        .-'                 `-.
    #       .-'                   `-.
    #      .-'                     `-.
    #      ::                       ::
    #      ::           o           ::           *
    #      ::           | *         ::        *
    #      `-.          |ϴ  * r    .-'     *
    #       `-.         |     *   .-'   *
    #        `-.      q |       *.-' *
    #          `-..     |    ..-' *
    #             ^-....|...-^ *
    #                 ''|'  *  ϴ
    #                    * -----------------
    #                 *
    #              *
    #           *
    #        *
    #     *
    #
    # cos(ϴ) = r / q
    #
    #      q = r / cos(ϴ)
    q: float = o['radius'] / math.cos(theta)
    c3: float = c2 - q
    c1: float = c2 + q
    return (
        {'m': m, 'c': c1},
        {'m': m, 'c': c3})


def _obstacle_path_lines_intersect_freeness(
        o: w.Obstacle,
        f: Freeness
        ) -> bool:
    path1, path2 = _obstacle_path_lines(o)
    circle: Circle = {
        'h': f['position']['x'],
        'k': f['position']['y'],
        'r': f['radius']}
    return (
        _line_intersects_circle(path1, circle) or
        _line_intersects_circle(path2, circle))


def _obstacle_intersects_freeness(
        f: Freeness,
        o: w.Obstacle
        ) -> bool:
    if _obstacle_currently_intersects_freeness(o, f):
        return True
    if not _obstacle_going_towards_freeness(o, f):
        return False
    return _obstacle_path_lines_intersect_freeness(o, f)
