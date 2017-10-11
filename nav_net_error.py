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



def _point_in_circle(p: w.Point, c: Circle) -> bool:
    """ It decides if the point is inside the circle. """
    return _distance_between(p, c['o']) < c['r']


def _straight_lines_identical(L1: Line, L2: Line) -> bool:
    """ It works out if the straight lines are the same. """
    if not math.isclose(L1['theta'], L2['theta']):
        # The lines are not parallel so are not identical.
        return False
    x1 = L1['X']['x']
    y1 = L1['X']['y']
    x2 = L2['X']['x']
    y2 = L2['X']['y']
    if math.isclose(x1, x2) and math.isclose(y1, y2):
        # The reference points for the lines are in the same place.
        return True
    # So if the function has reached this point then we know that the two
    # points are not in the same place and are on lines that are parallel
    # or identical.
    #
    #                      /
    #                     /
    #                    X (x2,y2)
    #                   /|
    #                  / |
    #                 /  |
    #                /   | y2 - y1
    #               /    |
    #              /     |
    #             / ϴ    | 
    #    (x1,y1) X-------+
    #           / x2 - x1
    #          /
    #         /
    #
    # If (ϴ - π/2) % π = 0 and x2 - x1 = 0 then the lines are
    # perpendicular and identical.
    #
    # If x2 - x1 != 0 then the lines are identical if
    #
    #     tan(ϴ) = (y2 - y1) / (x2 - x1)
    vertical = math.isclose(abs(L1['theta']) % math.pi, math.pi/2)
    same_x = math.isclose(x1, x2)
    if vertical and same_x:
        return True
    if vertical and not same_x:
        return False
    return math.isclose(math.tan(L1['theta']), (y2 - y1) / (x2 - x1))


def _perpendicular_distance_between_paralell_lines(
        L1: Line,
        L2: Line
        ) -> float:
    """ 
                             /                           
                            /                            
                           /                           / 
                          /                           /  
                         /                           /   
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
    assert math.isclose(L1['theta'], L2['theta'])
    theta: float = L1['theta']
    if _straight_lines_identical(L1, L2):
        return 0
    u: float = y2 - y1
    p: float = (u*u + (x2 - x1)**2)**0.5
    s: float = _calculate_s(theta, u)
    q: float = (p*p - u*u)**0.5 - s
    return q * math.sin(theta)


def _calculate_s(theta: float, u: float) -> float:
    if math.isclose(theta, math.pi/2):
        return 0
    return u / math.tan(theta)


def _is_vertical(L: Line) -> bool:
    return math.isclose(abs(L['theta']) % math.pi/2, math.pi/2)


class MxPlusC(TypedDict):
    m: float
    c: float


def _mx_plus_c(L: Line) -> Tuple[str, MxPlusC]:
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
    tan_theta = math.tan(L['theta'])
    return (
        None,
        {'m': tan_theta,
         'c': L['X']['y'] - L['X']['x'] * tan_theta})


def _solve_MxPlusC(a: MxPlusC, b: MxPlusC) -> w.Point:
    """
    It finds the intersection of two non-vertical and non-parallel 
    straight lines defined by equations of the form y = mx + c.

    Putting y = m1 * x + c1 and y = m2 * x + c2 into wxmaxima gives:
        x = -(c1 - c2) / (m1 - m2)  
    and
        y = (c2 * m1 - c1 * m2) / (m1 - m2).
    """
    return {
        'x': -(a['c'] - b['c']) / (a['m'] - b['c']),
        'y': (b['c'] * a['m'] - a['c'] * b['m']) / (a['m'] - b['m'])}


def _intersection_of(L1: Line, L2: Line) -> Tuple[str, w.Point]:
    """ It finds the point at the intersection of two straight lines. """
    err1, L1mc = _mx_plus_c(L1)
    err2, L2mc = _mx_plus_c(L2)
    if err1 is not None and err2 is not None:
        return "Both lines are vertical so no intersection.", None
    if err1 is not None and err2 is None:
        return (
            None,
            {'x': L1['X']['x'],
             L2mc['m'] * L1['X']['x'] + L2mc['c']})
    if err1 is None and err2 is not None:
        return (
            None,
            {'x': L2['X']['x'],
             L1mc['m'] * L1['X']['x'] + L1mc['c']})
    if math.isclose(L1mc['m'], L2mc['m']):
        return "Lines are parallel so do not intersect.", None
    if err1 is None and err2 is None:
        return None, _solve_MxPlusC(L1mc, L2mc)

         
def _perpendicular_distance_of_point_from_line(
        L: Line,
        p: w.Point
        ) -> Tuple[str, float]:
    err, X = _intersection_of(
        {'X': p, 'theta': L['theta'] + math.pi/2},
        L)
    if err is not None:
        return err, None
    return None, _distance_between(X, p)


def _point_between_parallel_lines(L1: Line, L2: Line, p: w.Point) -> bool:
    """ It decides if the point is between the two parallel lines. """
    d = _perpendicular_distance_between_paralell_lines(L1, L2)
    p1 = _perpendicular_distance_of_point_from_line(L1, p)
    p2 = _perpendicular_distance_of_point_from_line(L2, p)
    return p1 < d and p2 < d


def _point_in_front_of_start(o: w.Obstacle, p: w.Point) -> bool:
    px = p['x']
    py = p['y']
    opx = o['position']['x']
    opy = o['position']['y']
    ovx = o['velocity']['x']
    ovy = o['velocity']['y']
    relx = px - opx
    rely = py - opy
    return relx * ovx + rely * opy > 0



def _point_in_path(
        p: w.Point
        o: w.Obstacle,
        path1: Line,
        path2: Line,
        perpendicular: Line
        ) -> bool:
    """
    The function gives 'True' if (x, y) is in the area shaded by X's
    and false else.
    
                    :
                    : perpendicular
                    :
                    :
                   ___  _  _  _  _  _  _  _  _  _  path1
                 /  :  \ XXXXXXXXXXXXXXXXXXXXXXXXX
        ---> v  |   o   | XXXXXXXXXXXXXXXXXXXXXXXX
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
    circle: Circle = {'o': o['position'], 'r': o['radius']}
    if _point_in_circle(p, circle):
        return False
    if not _point_between_parallel_lines(path1, path2, p):
        return False
    return _point_in_front_of_start(o, p)
            
    
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


# Defines a straight line as a point and an angle.  The angle is measured
# anticlockwise from the positive x-axis.  The reason for not using 
# y = mx + c is that this is not defined for vertical lines.
class Line(TypedDict):
    X: w.Point
    theta: float


# Defines a circle with equation of the form (x - h)² + (y - k)² = r²,
# where (h, k) is the position of the centre and r is the radius.
class Circle(TypedDict):
    o: w.Point
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
    (%i1)   eq1: (x-h)^2 + (y-k)^2 = r^2;
    (eq1)   (y-k)^2+(x-h)^2=r^2
    (%i2)   eq2: y = m*x+c;
    (eq2)   y=m*x+c
    (%i3)   solve([eq1, eq2], [x, y]);
    (%o3)   [[x=(-sqrt((m^2+1)*r^2-h^2*m^2+k*(2*h*m+2*c)-2*c*h*m-k^2-c^2)
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
        c, d, ϴ


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
    c = r * math.sin(theta)
    d = b - r * math.cos(theta)
    return (
        {'X': {'x': c, 'y': d},
         'theta': theta},
        {'X': {'x': e, 'y': f},
         'theta': theta})
            

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
    #                   +*------------------
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
