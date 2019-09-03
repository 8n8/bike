"""
It provides two functions for updating the game GUI.  One of them,
update_world, is used to change the world state between frames, and the
other, world2view, uses the world state to calculate the parameters
needed to plot the world state in the GUI window.
"""


import enum
import math
from typing import List, Tuple, NamedTuple, Any, Union, Callable
import numpy as np  # type: ignore
from PIL import ImageTk, Image  # type: ignore
# import game_gui as g
import update_obstacle_pop as u
import world2sensor as s


@enum.unique
class KeyPress(enum.Enum):
    """
    For representing which arrow key is pressed on the keyboard.
    """
    UP = enum.auto()
    DOWN = enum.auto()
    LEFT = enum.auto()
    RIGHT = enum.auto()
    NONE = enum.auto()


class TkOval(NamedTuple):
    """
    It contains the parameters needed for drawing an oval in the GUI
    window.
    """
    top_left_x: float
    top_left_y: float
    bottom_right_x: float
    bottom_right_y: float
    fill_colour: str


class TkArrow(NamedTuple):
    """
    It contains the parameters needed for drawing an arrow in the GUI
    window.
    """
    start_x: float
    start_y: float
    stop_x: float
    stop_y: float
    colour: str
    width: float


class TkImage(NamedTuple):
    """ It contains an image and its placement in the GUI window. """
    image: Any
    x: float
    y: float


TkPicture = Union[TkOval, TkArrow, TkImage]


class Velocity(NamedTuple):
    """
    The velocity of a point in the x, y plane.
    """
    angle: float  # between 0 and 2π
    speed: float


class WorldState(NamedTuple):
    """ The state of the simulated world. """
    crashed: bool
    velocity: Velocity
    position: s.Vector
    target_velocity: Velocity
    obstacles: List[s.Obstacle]
    keyboard: KeyPress
    timestamp: float
    thin_view: s.ImageSet


UpdateWorld = Callable[[List[WorldState], u.RandomData, float, Any],
                       Tuple['np.ndarray[bool]', WorldState]]


def array2velocity(arr) -> Velocity:
    """ It converts a normalised numpy array into a velocity. """
    x, y = arr[0][0], arr[0][1]
    x1 = x*20 - 10
    y1 = y*20 - 10
    speed = (x1**2 + y1**2)**0.5
    angle = math.atan2(y1, x1)
    if angle < 0:
        angle += math.pi
    return Velocity(speed=speed, angle=angle)


def velocity2array(v: Velocity) -> 'np.ndarray[np.float64]':
    """
    It converts velocity into a numpy array and normalises it into
    the range [0, 1] so it can be compared with the neural net output.
    """
    x = (v.speed * math.cos(v.angle) + 10)/20
    y = (v.speed * math.sin(v.angle) + 10)/20
    return np.array([x, y])


def update_keyboard(key: KeyPress, w: WorldState) -> WorldState:
    """ It updates the 'keyboard' element of the world state. """
    return WorldState(
        crashed=w.crashed,
        velocity=w.velocity,
        position=w.position,
        target_velocity=w.target_velocity,
        obstacles=w.obstacles,
        keyboard=key,
        timestamp=w.timestamp,
        thin_view=w.thin_view)


class DataSet(NamedTuple):
    """ It represents a whole game's worth of data. """
    images: 'np.ndarray[bool]'  # n x 100 x 4 x 4
    target_velocity: 'np.ndarray[np.float64]'  # n x 2
    velocity_in: 'np.ndarray[np.float64]'  # n x 2
    velocity_out: 'np.ndarray[np.float64]'  # n x 2


MODEL_FILE: str = 'nav_net.h5'


def prepare_for_save(
        recent_images_set: List['np.ndarray[bool]'],
        history: List[WorldState]) -> DataSet:
    """
    It converts the data gathered over the course of the game into
    numpy arrays.
    """
    good_indices = [i for i, element in enumerate(recent_images_set)
                    if element is not None
                    and i % 5 == 0]
    target_velocities = [
        velocity2array(history[i].target_velocity)
        for i in good_indices]
    velocity_outs = [
        velocity2array(history[i].velocity) for i in good_indices]
    velocity_ins = [
        velocity2array(Velocity(speed=0, angle=0))] + velocity_outs[:-1]
    images = [np.expand_dims(recent_images_set[i], axis=1)
              for i in good_indices]
    return DataSet(
        target_velocity=np.stack(target_velocities, axis=0),  # type: ignore
        velocity_out=np.stack(velocity_outs, axis=0),  # type: ignore
        velocity_in=np.stack(velocity_ins, axis=0),  # type: ignore
        images=np.stack(images, axis=0))  # type: ignore


def _squared_distance_between(a: s.Vector, b: s.Vector) -> float:
    """ It calculates the square of the distance between two points. """
    return (a['x'] - b['x'])**2 + (a['y'] - b['y'])**2


def _crashed_into_obstacle(position, obstacles) -> bool:
    """ It works out if the robot has crashed into an obstacle. """
    return any([
        _squared_distance_between(position, o['position']) < 4
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


def _update_velocity_manual(key: KeyPress, v: Velocity) -> Velocity:
    """ It changes the velocity in response to a key press. """
    if key == KeyPress.UP:
        return Velocity(
            angle=v.angle,
            speed=_speed_mod(v.speed + SPEED_STEP))
    if key == KeyPress.DOWN:
        return Velocity(
            angle=v.angle,
            speed=_speed_mod(v.speed - SPEED_STEP))
    if key == KeyPress.LEFT:
        return Velocity(
            angle=_angle_mod(v.angle + ANGLE_STEP),
            speed=v.speed)
    if key == KeyPress.RIGHT:
        return Velocity(
            angle=_angle_mod(v.angle - ANGLE_STEP),
            speed=v.speed)
    return v


IMAGE_TIMES: Tuple[float, float, float, float] = (0, 0.5, 1, 1.5)


def imageset2numpy(i: s.ImageSet) -> 'np.ndarray[bool]':
    """ It converts a dictionary of images into a numpy array.  """
    return np.stack(  # type: ignore
        [i['front'], i['left'], i['back'], i['right']],
        axis=1)


def make_recent_images(
        ws: List[WorldState]) -> 'np.ndarray[bool]':
    """
    It works out the set of recent images for feeding into the neural
    network, given the history of world states.  The output array is of
    shape 100 x 4 x 4.
    """
    # <<<<<<< HEAD
    #     now: float = ws[-1].timestamp
    #     timestamps = np.array([w.timestamp for w in ws])
    #     i_s: List[int] = [i_for_n_seconds_ago(timestamps, t, now)
    #                       for t in IMAGE_TIMES]
    #     nones: List[bool] = [i is None for i in i_s]
    #     if any(nones):
    #         return "Not possible to make this batch.", None
    #     batch: List['np.ndarray[bool]'] = [
    #         imageset2numpy(ws[i].thin_view) for i in i_s]
    #     return None, np.stack(batch, axis=2)  # type: ignore
    # =======
    return ws[-1].thin_view
    # now: float = ws[-1].timestamp
    # timestamps = np.array([w.timestamp for w in ws])
    # i_s: List[int] = [i_for_n_seconds_ago(timestamps, t, now)
    #                   for t in IMAGE_TIMES]
    # nones: List[bool] = [i is None for i in i_s]
    # if any(nones):
    #     return "Not possible to make this batch.", None
    # batch: List['np.ndarray[bool]'] = [
    #     imageset2numpy(ws[i].thin_view) for i in i_s]
    # return None, np.stack(batch, axis=2)  # type: ignore


def _update_velocity_auto(
        target_velocity: Velocity,
        velocity_in: Velocity,
        recent_images: 'np.ndarray[bool]',
        model) -> Velocity:
    """ It changes the velocity using the neural network. """
    print(velocity2array(velocity_in))
    result = array2velocity(model.predict(
        {'image_in': np.expand_dims(np.expand_dims(
              recent_images, axis=0), axis=2),  # type: ignore
         'target_in': np.expand_dims(  # type: ignore
             velocity2array(target_velocity),
             axis=0),
         'velocity_in': np.expand_dims(  # type: ignore
             velocity2array(velocity_in),
             axis=0)},
        batch_size=1))
    # print(result)
    return result


def _update_position(v: Velocity, p: s.Vector, t: float) -> s.Vector:
    """
    It calculates the next position given the current one, the velocity,
    and a timestep.
    """
    return {
        'x': p['x'] + t * v.speed * math.cos(v.angle),
        'y': p['y'] + t * v.speed * math.sin(v.angle)}


@enum.unique
class Mode(enum.Enum):
    """ It is for specifying if update_world is automatic or manual. """
    AUTO = enum.auto()
    MANUAL = enum.auto()


def auto_update_world(
        history: List[WorldState],
        rand: u.RandomData,
        timestep: float,
        model) -> Tuple['np.ndarray[bool]', WorldState]:
    """
    It provides a wrapper around the update_world function with the AUTO
    flag set.
    """
    return update_world(history, rand, Mode.AUTO, timestep, model)


def manual_update_world(
        history: List[WorldState],
        rand: u.RandomData,
        timestep: float,
        model) -> Tuple['np.ndarray[bool]', WorldState]:
    """
    It provides a wrapper around the update_world function with the
    MANUAL flag set.
    """
    return update_world(history, rand, Mode.MANUAL, timestep, model)


def update_world(
        history: List[WorldState],
        rand: u.RandomData,
        mode: Mode,
        timestep: float,
        model) -> Tuple['np.ndarray[bool]', WorldState]:
    """
    It predicts the new state of the world in a short amount of time,
    given the current state and some random data for calculating the
    obstacle parameters.
    """
    recent_images = make_recent_images(history)
    init = history[-1]
    if mode == Mode.AUTO:
        velocity: Velocity = _update_velocity_auto(
        init.target_velocity, init.velocity, recent_images, model)
    if mode == Mode.MANUAL:
        velocity = _update_velocity_manual(init.keyboard, init.velocity)
    return (  # type: ignore
        recent_images,
        WorldState(
            crashed=_crashed_into_obstacle(init.position, init.obstacles),
            velocity=velocity,
            position=_update_position(init.velocity, init.position, timestep),
            target_velocity=init.target_velocity,
            obstacles=u.main(init.obstacles, timestep, init.position, rand),
            thin_view=s.calculate_small_images(
                init.obstacles,
                init.position['x'],
                init.position['y'],
                init.velocity.angle),
            keyboard=KeyPress.NONE,
            timestamp=init.timestamp + timestep))


def _circle(x: float, y: float, r: float, colour: str) -> TkOval:
    """
    It calculates the parameters needed to draw a circle in TKinter, with
    centre at (x, y) and radius r.
    """
    return TkOval(
        top_left_x=x - r,
        top_left_y=y - r,
        bottom_right_x=x + r,
        bottom_right_y=y + r,
        fill_colour=colour)


def _polar2cart(v: Velocity) -> s.Vector:
    """ It converts a vector from polar to cartesian form. """
    return {
        'x': v.speed * math.cos(v.angle),
        'y': v.speed * math.sin(v.angle)}


XOFFSET: float = 330
YOFFSET: float = 600
SCALE: float = 7


def _arrow(v: Velocity, colour: str) -> TkArrow:
    """
    It calculates the parameters needed to draw an arrow in Tkinter,
    centred at the origin, with a length proportional to the magnitude
    of the velocity.
    """
    vcart = _polar2cart(v)
    return TkArrow(
        start_x=XOFFSET - SCALE*(vcart['x']),
        start_y=YOFFSET + SCALE*(vcart['y']),
        stop_x=XOFFSET + SCALE*vcart['x'],
        stop_y=YOFFSET - SCALE*vcart['y'],
        colour=colour,
        width=1)


def _plot_obstacle(o: s.Obstacle) -> TkOval:
    """
    It calculates the parameters needed to plot an obstacle as a circle
    in the Tkinter window.
    """
    return _circle(
        SCALE * o['position']['x'] + XOFFSET,
        - SCALE * o['position']['y'] + YOFFSET,
        SCALE * o['radius'],
        'black')


def _shift_and_centre(
        o: s.Obstacle,
        bikepos: s.Vector,
        bikevel: Velocity) -> s.Obstacle:
    """
    It shifts and rotates the obstacle so that the robot can be plotted
    at the centre of the window and the obstacles move around it instead
    of the robot moving through the obstacles.
    """
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
    theta = - bikevel.angle + math.pi/2
    sintheta = math.sin(theta)
    costheta = math.cos(theta)
    x = shifted['x']
    y = shifted['y']
    rotated: s.Vector = {
        'x': x * costheta - y * sintheta,
        'y': x * sintheta + y * costheta}
    return {
        'position': rotated,
        'velocity': o['velocity'],
        'radius': o['radius']}


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
    return {
        'front': _numpy_x1_to_TKimage(i['front']),
        'left': _numpy_x1_to_TKimage(i['left']),
        'back': _numpy_x1_to_TKimage(i['back']),
        'right': _numpy_x1_to_TKimage(i['right'])}


def _make_tk_images(small_images: s.ImageSet) -> List[TkImage]:
    """
    It calculates the images from the worldstate and converts them into
    the correct format for displaying in a Tkinter window.
    """
    # <<<<<<< HEAD
    #     images = _numpy_x4_to_TKimage(s.calculate_rgb_images(small_images))
    #     return [
    #         TkImage(image=images['front'], x=320, y=110),
    #         TkImage(image=images['back'], x=320, y=330),
    #         TkImage(image=images['left'], x=110, y=220),
    #         TkImage(image=images['right'], x=530, y=220)]
    # =======
    return TkImage(
        image=_numpy_x1_to_TKimage(s.calculate_rgb_images(small_images)['front']),
        x=320,
        y=110)


def world2view(w: WorldState) -> List[TkPicture]:
    """
    It works out how to make the view in the Tkinter window, given the
    state of the world.
    """
    robot: TkPicture = _circle(XOFFSET, YOFFSET, SCALE * 1.2, 'red')
    arrow_actual: TkPicture = _arrow(
        Velocity(speed=w.velocity.speed, angle=math.pi/2),
        'red')
    arrow_target: TkPicture = _arrow(
        Velocity(speed=w.target_velocity.speed,
                 angle=(w.target_velocity.angle
                        - w.velocity.angle + math.pi/2)),
        'black')
    # <<<<<<< HEAD
    #     obstacles: List[TkPicture] = [
    #         _plot_obstacle(_shift_and_centre(o, w.position, w.velocity))
    #         for o in w.obstacles]
    #     images: List[TkPicture] = _make_tk_images(w.thin_view)  # type: ignore
    #     return [robot, arrow_actual, arrow_target] + images # + obstacles
    # =======
    image: TkPicture = _make_tk_images(w.thin_view)  # type: ignore
    return [robot, arrow_actual, arrow_target, image]
