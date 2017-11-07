import math
from PIL import ImageTk, Image  # type: ignore
from typing import List, Tuple
import game_gui as g
import update_obstacle_pop as u
import world2sensor as s


XOFFSET: float = 330
YOFFSET: float = 800
SCALE: float = 8


def _distance_between(a: s.Vector, b: s.Vector) -> float:
    return ((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2)**0.5


def _crashed_into_obstacle(position, obstacles) -> bool:
    """ It works out if the robot has crashed into an obstacle. """
    return any([
        _distance_between(position, o['position']) < 2
        for o in obstacles])


def _angle_mod(angle: float) -> float:
    """ It makes sure the angle is between 0 and 2π. """
    if angle < 0:
        return _angle_mod(angle + 2 * math.pi)
    return angle % (2 * math.pi)


def _speed_mod(speed: float) -> float:
    """ It makes sure the speed stays in the interval (-10, 10). """
    if speed > 10:
        return 10
    if speed < -10:
        return -10
    return speed


def _update_velocity(key: g.KeyPress, v: g.Velocity) -> g.Velocity:
    """ It changes the velocity in response to a key press. """
    speed_step = 0.7
    angle_step = math.pi/46
    if key == g.KeyPress.UP:
        return g.Velocity(
            angle=v.angle,
            speed=_speed_mod(v.speed + speed_step))
    if key == g.KeyPress.DOWN:
        return g.Velocity(
            angle=v.angle,
            speed=_speed_mod(v.speed - speed_step))
    if key == g.KeyPress.LEFT:
        return g.Velocity(
            angle=_angle_mod(v.angle + angle_step),
            speed=v.speed)
    if key == g.KeyPress.RIGHT:
        return g.Velocity(
            angle=_angle_mod(v.angle - angle_step),
            speed=v.speed)
    return v


def _update_position(v: g.Velocity, p: s.Vector, t: float) -> s.Vector:
    velx: float = v.speed * math.cos(v.angle)
    vely: float = v.speed * math.sin(v.angle)
    return {
        'x': p['x'] + t * velx,
        'y': p['y'] + t * vely}


def update_world(
        init: g.WorldState,
        rand: u.RandomData,
        timestep: float) -> g.WorldState:
    return g.WorldState(
        crashed=_crashed_into_obstacle(init.position, init.obstacles),
        velocity=_update_velocity(init.keyboard, init.velocity),
        position=_update_position(init.velocity, init.position, timestep),
        target_velocity=init.target_velocity,
        obstacles=u.main(init.obstacles, timestep, init.position, rand),
        keyboard=g.KeyPress.NONE)


def _circle(x: float, y: float, r: float, colour: str) -> g.TkOval:
    return g.TkOval(
        top_left_x=x - r,
        top_left_y=y - r,
        bottom_right_x=x + r,
        bottom_right_y=y + r,
        fill_colour=colour)


def polar2cart(v: g.Velocity) -> s.Vector:
    return {
        'x': v.speed * math.cos(v.angle),
        'y': v.speed * math.sin(v.angle)}


def _arrow(v: g.Velocity, colour: str) -> g.TkArrow:
    vcart = polar2cart(v)
    return g.TkArrow(
        start_x=XOFFSET - SCALE*(vcart['x']),
        start_y=YOFFSET + SCALE*(vcart['y']),
        stop_x=XOFFSET + SCALE*vcart['x'],
        stop_y=YOFFSET - SCALE*vcart['y'],
        colour=colour,
        width=1)


def _plot_obstacle(o: s.Obstacle) -> g.TkOval:
    return _circle(
        SCALE * o['position']['x'] + XOFFSET,
        - SCALE * o['position']['y'] + YOFFSET,
        SCALE * o['radius'],
        'black')


def update_obstacle(
        o: s.Obstacle,
        bikepos: s.Vector,
        bikevel: g.Velocity) -> s.Obstacle:
    shifted: s.Vector = {
        'x': o['position']['x'] - bikepos['x'],
        'y': o['position']['y'] - bikepos['y']}
    # The rotation matrix is:
    #
    #     [ cos ϴ   - sin ϴ ]
    #     [ sin ϴ   cos ϴ   ]
    #
    # So the rotated position is:
    #
    #    [ x cos ϴ - y sin ϴ ]
    #    [ x sin ϴ + y cos ϴ ]
    theta = bikevel.angle - math.pi/2
    sintheta = math.sin(-theta)
    costheta = math.cos(-theta)
    x = shifted['x']
    y = shifted['y']
    rotated: s.Vector = {
        'x': x * costheta - y * sintheta,
        'y': x * sintheta + y * costheta}
    return {
        'position': rotated,
        'velocity': o['velocity'],
        'radius': o['radius']}


def draw_arrows(
        actual_v: g.Velocity,
        target_v: g.Velocity) -> Tuple[g.TkArrow, g.TkArrow]:
    # The rotation matrix is:
    #
    #     [ cos ϴ   - sin ϴ ]
    #     [ sin ϴ   cos ϴ   ]
    #
    # So the rotated position is:
    #
    #    [ x cos ϴ - y sin ϴ ]
    #    [ x sin ϴ + y cos ϴ ]
    rotated_target = g.Velocity(
        speed=target_v.speed,
        angle=_angle_mod(target_v.angle - actual_v.angle + math.pi/2))
    rotated_actual = g.Velocity(
        speed=actual_v.speed,
        angle=math.pi/2)
    return (_arrow(rotated_actual, 'red'),
            _arrow(rotated_target, 'black'))


def numpy_to_TKimage(i: s.ImageSet):
    """
    It converts a set of four camera images from the four cameras from
    m x n x 3 numpy arrays to the right format for displaying in
    Tkinter.
    """
    def f(im):
        return ImageTk.PhotoImage(
            Image.fromarray(im).resize((200, 200), Image.ANTIALIAS))
    return {
        'front': f(i['front']),
        'left': f(i['left']),
        'back': f(i['back']),
        'right': f(i['right'])}


def make_images(w: g.WorldState) -> List[g.TkImage]:
    """
    It calculates the images from the worldstate and converts them into
    the correct format for displaying in a Tkinter window.
    """
    images = numpy_to_TKimage(s._calculate_rgb_images(
        w.obstacles,
        w.position['x'],
        w.position['y'],
        w.velocity.angle))
    return [
        g.TkImage(image=images['front'], x=320, y=110),
        g.TkImage(image=images['back'], x=320, y=330),
        g.TkImage(image=images['left'], x=110, y=220),
        g.TkImage(image=images['right'], x=530, y=220)]


def world2view(w: g.WorldState) -> List[g.TkPicture]:
    robot: g.TkPicture = _circle(XOFFSET, YOFFSET, SCALE * 1.2, 'red')
    shifted_obstacles: List[s.Obstacle] = [
        update_obstacle(o, w.position, w.velocity) for o in w.obstacles]
    obstacles: List[g.TkPicture] = [
        _plot_obstacle(o) for o in shifted_obstacles]
    arrows: Tuple[g.TkArrow, g.TkArrow] = draw_arrows(
        w.velocity, w.target_velocity)
    arrow1: g.TkPicture = arrows[0]
    arrow2: g.TkPicture = arrows[1]
    images: List[g.TkImage] = make_images(w)
    return [robot, arrow1, arrow2] + obstacles + images  # type: ignore
