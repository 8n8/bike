"""
It provides two functions for updating the game GUI.  One of them,
update_world, is used to change the world state between frames, and the
other, world2view, uses the world state to calculate the parameters
needed to plot the world state in the GUI window.
"""


import enum
import math
from typing import List, Tuple
import numpy as np
from PIL import ImageTk, Image  # type: ignore
import game_gui as g
import update_obstacle_pop as u
import world2sensor as s


def _squared_distance_between(a: s.Vector, b: s.Vector) -> float:
    """ It calculates the square of the distance between two points. """
    return (a.x - b.x)**2 + (a.y - b.y)**2


def _crashed_into_obstacle(position, obstacles) -> bool:
    """ It works out if the robot has crashed into an obstacle. """
    return any([
        _squared_distance_between(position, o.position) < 4
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


SPEED_STEP: float = 0.7
ANGLE_STEP: float = math.pi/46


def _update_velocity_manual(key: g.KeyPress, v: g.Velocity) -> g.Velocity:
    """ It changes the velocity in response to a key press. """
    if key == g.KeyPress.UP:
        return g.Velocity(
            angle=v.angle,
            speed=_speed_mod(v.speed + SPEED_STEP))
    if key == g.KeyPress.DOWN:
        return g.Velocity(
            angle=v.angle,
            speed=_speed_mod(v.speed - SPEED_STEP))
    if key == g.KeyPress.LEFT:
        return g.Velocity(
            angle=_angle_mod(v.angle + ANGLE_STEP),
            speed=v.speed)
    if key == g.KeyPress.RIGHT:
        return g.Velocity(
            angle=_angle_mod(v.angle - ANGLE_STEP),
            speed=v.speed)
    return v


def i_for_n_seconds_ago(
        timestamps: 'np.ndarray[np.float64]',
        n: float,
        now: float) -> int:
    """
    It finds the index of the timestamp that is close to being n
    seconds ago.
    """
    comparison = np.ones_like(timestamps)*(now - n)
    matching = abs(timestamps - comparison) < 0.1  # type: ignore
    indices = matching.nonzero()
    if indices[0].shape[0] == 0:  # i.e., if the list is empty
        return None
    return indices[0][0]


IMAGE_TIMES: Tuple[float, float, float, float] = (0, 0.5, 1, 1.5)


def imageset2numpy(i: s.ImageSet) -> 'np.ndarray[bool]':
    """ It converts a dictionary of images into a numpy array.  """
    return np.stack(  # type: ignore
        [i.front, i.left, i.back, i.right],
        axis=1)


def make_recent_images(
        ws: List[g.WorldState]) -> Tuple[str, 'np.ndarray[bool]']:
    """
    It works out the set of recent images for feeding into the neural
    network, given the history of world states.  The output array is of
    shape 100 x 4 x 4.
    """
    now: float = ws[-1].timestamp
    timestamps = np.array([w.timestamp for w in ws])
    i_s: List[int] = [i_for_n_seconds_ago(timestamps, t, now)
                      for t in IMAGE_TIMES]
    nones: List[bool] = [i is None for i in i_s]
    if any(nones):
        return "Not possible to make this batch.", None
    batch: List['np.ndarray[bool]'] = [
        imageset2numpy(ws[i].thin_view) for i in i_s]
    return None, np.stack(batch, axis=2)  # type: ignore


def array2velocity(arr) -> g.Velocity:
    """ It converts a normalised numpy array into a velocity. """
    return g.Velocity(
        speed=arr[0][0]*20 - 10,
        angle=arr[0][1]*2*math.pi)


def velocity2array(v: g.Velocity) -> 'np.ndarray[np.float64]':
    """
    It converts velocity into a numpy array and normalises it into
    the range [0, 1] so it can be compared with the neural net output.
    """
    return np.array([(v.speed+10)/20, v.angle/(2*math.pi)])


def _update_velocity_auto(
        target_velocity: g.Velocity,
        current_velocity: g.Velocity,
        recent_images: 'np.ndarray[bool]',
        model) -> g.Velocity:
    """ It changes the velocity using the neural network. """
    return array2velocity(model.predict(
        {'image_in': np.expand_dims(recent_images, axis=0),  # type: ignore
         'velocity_in': np.expand_dims(  # type: ignore
             velocity2array(target_velocity),
             axis=0),
         'target_in': np.expand_dims(  # type: ignore
             velocity2array(current_velocity),
             axis=0)},
        batch_size=1))


def _update_position(v: g.Velocity, p: s.Vector, t: float) -> s.Vector:
    """
    It calculates the next position given the current one, the velocity,
    and a timestep.
    """
    return s.Vector(
        x=p.x + t * v.speed * math.cos(v.angle),
        y=p.y + t * v.speed * math.sin(v.angle))


@enum.unique
class Mode(enum.Enum):
    """ It is for specifying if update_world is automatic or manual. """
    AUTO = enum.auto()
    MANUAL = enum.auto()


def auto_update_world(
        history: List[g.WorldState],
        rand: u.RandomData,
        timestep: float,
        model) -> Tuple['np.ndarray[bool]', g.WorldState]:
    """
    It provides a wrapper around the update_world function with the AUTO
    flag set.
    """
    return update_world(history, rand, Mode.AUTO, timestep, model)


def manual_update_world(
        history: List[g.WorldState],
        rand: u.RandomData,
        timestep: float,
        model) -> Tuple['np.ndarray[bool]', g.WorldState]:
    """
    It provides a wrapper around the update_world function with the
    MANUAL flag set.
    """
    return update_world(history, rand, Mode.MANUAL, timestep, model)


def update_world(
        history: List[g.WorldState],
        rand: u.RandomData,
        mode: Mode,
        timestep: float,
        model) -> Tuple['np.ndarray[bool]', g.WorldState]:
    """
    It predicts the new state of the world in a short amount of time,
    given the current state and some random data for calculating the
    obstacle parameters.
    """
    err, recent_images = make_recent_images(history)
    init = history[-1]
    if mode == Mode.AUTO:
        if err is None:
            velocity: g.Velocity = _update_velocity_auto(
                init.target_velocity, init.velocity, recent_images, model)
        else:
            velocity = init.velocity
    if mode == Mode.MANUAL:
        velocity = _update_velocity_manual(init.keyboard, init.velocity)
    return (  # type: ignore
        recent_images,
        g.WorldState(
            crashed=_crashed_into_obstacle(init.position, init.obstacles),
            velocity=velocity,
            position=_update_position(init.velocity, init.position, timestep),
            target_velocity=init.target_velocity,
            obstacles=u.main(init.obstacles, timestep, init.position, rand),
            thin_view=s.calculate_small_images(
                init.obstacles,
                init.position.x,
                init.position.y,
                init.velocity.angle),
            keyboard=g.KeyPress.NONE,
            timestamp=init.timestamp + timestep))


def _circle(x: float, y: float, r: float, colour: str) -> g.TkOval:
    """
    It calculates the parameters needed to draw a circle in TKinter, with
    centre at (x, y) and radius r.
    """
    return g.TkOval(
        top_left_x=x - r,
        top_left_y=y - r,
        bottom_right_x=x + r,
        bottom_right_y=y + r,
        fill_colour=colour)


def _polar2cart(v: g.Velocity) -> s.Vector:
    """ It converts a vector from polar to cartesian form. """
    return s.Vector(
        x=v.speed * math.cos(v.angle),
        y=v.speed * math.sin(v.angle))


XOFFSET: float = 330
YOFFSET: float = 800
SCALE: float = 8


def _arrow(v: g.Velocity, colour: str) -> g.TkArrow:
    """
    It calculates the parameters needed to draw an arrow in Tkinter,
    centred at the origin, with a length proportional to the magnitude
    of the velocity.
    """
    vcart = _polar2cart(v)
    return g.TkArrow(
        start_x=XOFFSET - SCALE*(vcart.x),
        start_y=YOFFSET + SCALE*(vcart.y),
        stop_x=XOFFSET + SCALE*vcart.x,
        stop_y=YOFFSET - SCALE*vcart.y,
        colour=colour,
        width=1)


def _plot_obstacle(o: s.Obstacle) -> g.TkOval:
    """
    It calculates the parameters needed to plot an obstacle as a circle
    in the Tkinter window.
    """
    return _circle(
        SCALE * o.position.x + XOFFSET,
        - SCALE * o.position.y + YOFFSET,
        SCALE * o.radius,
        'black')


def _shift_and_centre(
        o: s.Obstacle,
        bikepos: s.Vector,
        bikevel: g.Velocity) -> s.Obstacle:
    """
    It shifts and rotates the obstacle so that the robot can be plotted
    at the centre of the window and the obstacles move around it instead
    of the robot moving through the obstacles.
    """
    shifted: s.Vector = s.Vector(
        x=o.position.x - bikepos.x,
        y=o.position.y - bikepos.y)
    # The rotation matrix is:
    #
    #     [ cos ϴ   - sin ϴ ]
    #     [ sin ϴ   cos ϴ   ]
    #
    # So the rotated position is:
    #
    #    [ x cos ϴ - y sin ϴ ]
    #    [ x sin ϴ + y cos ϴ ]
    theta = - bikevel.angle + math.pi/2
    sintheta = math.sin(theta)
    costheta = math.cos(theta)
    x = shifted.x
    y = shifted.y
    rotated: s.Vector = s.Vector(
        x=x * costheta - y * sintheta,
        y=x * sintheta + y * costheta)
    return s.Obstacle(
        position=rotated,
        velocity=o.velocity,
        radius=o.radius)


def _numpy_x1_to_TKimage(image: 'np.ndarray[np.uint8]'):
    """
    It converts a single numpy array into an image in the right format for
    displaying in Tkinter.
    """
    return ImageTk.PhotoImage(
        Image.fromarray(image).resize((200, 200), Image.ANTIALIAS))


def _numpy_x4_to_TKimage(i: s.ImageSet):
    """
    It converts a set of four camera images from the four cameras from
    m x n x 3 numpy arrays to the right format for displaying in
    Tkinter.
    """
    return s.ImageSet(
        front=_numpy_x1_to_TKimage(i.front),
        left=_numpy_x1_to_TKimage(i.left),
        back=_numpy_x1_to_TKimage(i.back),
        right=_numpy_x1_to_TKimage(i.right))


def _make_tk_images(small_images: s.ImageSet) -> List[g.TkImage]:
    """
    It calculates the images from the worldstate and converts them into
    the correct format for displaying in a Tkinter window.
    """
    images = _numpy_x4_to_TKimage(s.calculate_rgb_images(small_images))
    return [
        g.TkImage(image=images.front, x=320, y=110),
        g.TkImage(image=images.back, x=320, y=330),
        g.TkImage(image=images.left, x=110, y=220),
        g.TkImage(image=images.right, x=530, y=220)]


def world2view(w: g.WorldState) -> List[g.TkPicture]:
    """
    It works out how to make the view in the Tkinter window, given the
    state of the world.
    """
    robot: g.TkPicture = _circle(XOFFSET, YOFFSET, SCALE * 1.2, 'red')
    arrow_actual: g.TkPicture = _arrow(
        g.Velocity(speed=w.velocity.speed, angle=math.pi/2),
        'red')
    arrow_target: g.TkPicture = _arrow(
        g.Velocity(speed=w.target_velocity.speed,
                   angle=(w.target_velocity.angle
                          - w.velocity.angle + math.pi/2)),
        'black')
    obstacles: List[g.TkPicture] = [
        _plot_obstacle(_shift_and_centre(o, w.position, w.velocity))
        for o in w.obstacles]
    images: List[g.TkPicture] = _make_tk_images(w.thin_view)  # type: ignore
    return [robot, arrow_actual, arrow_target] + obstacles + images
