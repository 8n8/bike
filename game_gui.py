""" It provides a function for creating a Tkinter GUI. """

from enum import Enum, auto, unique
import os
import uuid
from typing import Any, Callable, NamedTuple, List, Union, Tuple
import tkinter as k
from keras.models import load_model  # type: ignore
import numpy as np
import update_obstacle_pop as u
import world2sensor as w


class Velocity(NamedTuple):
    """
    The velocity of a point in the x, y plane.
    """
    angle: float  # between 0 and 2π
    speed: float


@unique
class KeyPress(Enum):
    """
    For representing which arrow key is pressed on the keyboard.
    """
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    NONE = auto()


class WorldState(NamedTuple):
    """ The state of the simulated world. """
    crashed: bool
    velocity: Velocity
    position: w.Vector
    target_velocity: Velocity
    obstacles: List[w.Obstacle]
    keyboard: KeyPress
    timestamp: float
    thin_view: w.ImageSet


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


UpdateWorld = Callable[[List[WorldState], u.RandomData, float, Any],
                       Tuple['np.ndarray[bool]', WorldState]]


IMAGE_TIMES = (0, 0.5, 1, 1.5)


def update_keyboard(key: KeyPress, s: WorldState) -> WorldState:
    """ It updates the 'keyboard' element of the world state. """
    return WorldState(
        crashed=s.crashed,
        velocity=s.velocity,
        position=s.position,
        target_velocity=s.target_velocity,
        obstacles=s.obstacles,
        keyboard=key,
        timestamp=s.timestamp,
        thin_view=s.thin_view)


class DataSet(NamedTuple):
    """ It represents a whole game's worth of data. """
    images: 'np.ndarray[bool]'  # n x 100 x 4 x 4
    target_velocity: 'np.ndarray[np.float64]'  # n x 2
    velocity_in: 'np.ndarray[np.float64]'  # n x 2
    velocity_out: 'np.ndarray[np.float64]'  # n x 2


def prepare_for_save(
        recent_images_set: List['np.ndarray[bool]'],
        history: List[WorldState]) -> DataSet:
    """
    It converts the data gathered over the course of the game into
    numpy arrays.
    """
    good_indices = [i for i, element in enumerate(recent_images_set)
                    if element is not None]
    target_velocities = [history[i].target_velocity for i in good_indices]
    velocity_outs = [history[i].velocity for i in good_indices]
    velocity_ins = [Velocity(speed=0, angle=0)] + velocity_outs[:-1]
    images = [recent_images_set[i] for i in good_indices]
    return DataSet(
        target_velocity=np.stack(target_velocities, axis=0),  # type: ignore
        velocity_out=np.stack(velocity_outs, axis=0),  # type: ignore
        velocity_in=np.stack(velocity_ins, axis=0),  # type: ignore
        images=np.stack(images, axis=0))  # type: ignore


MODEL_FILE: str = 'nav_net.h5'


class _World:
    """
    It holds the state of the world and various functions for updating it.
    """
    def __init__(
            self,
            canvas,
            init: WorldState,
            timestep: float,
            update_world: UpdateWorld,
            world2view: Callable[[WorldState], List[TkPicture]]) -> None:
        self.state = init
        self.canvas = canvas
        self.timestep = timestep
        self.history: List[WorldState] = []
        self.world2view = world2view
        self.update_world = update_world
        self.time = 0
        self.imref = []  # type: ignore
        self.recent_images: List['np.ndarray[bool]'] = []
        if os.path.isfile(MODEL_FILE):
            self.model = load_model(MODEL_FILE)
        else:
            self.model = None

    def update(self):
        """ It updates the gui window and the world state. """
        if self.state.crashed:
            print('Robot has crashed into obstacle.')
            return
        self.canvas.delete('all')
        for i in self.world2view(self.state):
            if isinstance(i, TkOval):
                self.canvas.create_oval(
                    i.top_left_x,
                    i.top_left_y,
                    i.bottom_right_x,
                    i.bottom_right_y,
                    fill=i.fill_colour)
                continue
            if isinstance(i, TkArrow):
                self.canvas.create_line(
                    i.start_x,
                    i.start_y,
                    i.stop_x,
                    i.stop_y,
                    arrow=k.LAST,
                    fill=i.colour,
                    width=i.width)
                continue
            if isinstance(i, TkImage):
                self.canvas.create_image(i.x, i.y, image=i.image)
                self.imref.append(i.image)
                self.imref = self.imref[-4:]
        self.history.append(self.state)
        recent_images, self.state = self.update_world(
            self.history,
            u.generate_params(),
            self.timestep,
            self.model)
        self.recent_images.append(recent_images)
        self.time += self.timestep
        self.canvas.after(int(1000 * self.timestep), self.update)

    def keyboard_up(self, _):
        """ Records the pressing of the up arrow key. """
        self.state = update_keyboard(KeyPress.UP, self.state)

    def keyboard_down(self, _):
        """ Records the pressing of the down arrow key. """
        self.state = update_keyboard(KeyPress.DOWN, self.state)

    def keyboard_left(self, _):
        """ Records the pressing of the left arrow key. """
        self.state = update_keyboard(KeyPress.LEFT, self.state)

    def keyboard_right(self, _):
        """ Records the pressing of the right arrow key. """
        self.state = update_keyboard(KeyPress.RIGHT, self.state)

    def save(self, _):
        """ It saves the data gathered so far to file. """
        filename: str = 'game_data/' + str(uuid.uuid4())
        dat: DataSet = prepare_for_save(self.recent_images, self.history)
        np.savez(filename, dat.images, dat.target_velocity,
                 dat.velocity_in, dat.velocity_out)


def main(
        init: WorldState,
        timestep: float,
        update_world: UpdateWorld,
        world2view: Callable[[WorldState], List[TkPicture]]) -> None:
    """
    It creates a GUI window with moving objects in it.

    :param init: The initial state of the world.
    :param timestep: The amount of time between frames.
    :param update_world: A function that takes in the state of the
        world, a timestep and a set of random parameters, and returns
        the new world state.
    :param world2view: A function that takes in the world state and
        calculates the parameters needed to draw objects on the GUI
        window.
    """
    root = k.Tk()
    root.title('Robot navigation simulator')
    root.geometry('800x600')
    canvas = k.Canvas(root, width=650, height=1000, bg='grey')
    canvas.pack()

    world = _World(canvas, init, timestep, update_world, world2view)

    root.bind('<Up>', world.keyboard_up)
    root.bind('<Down>', world.keyboard_down)
    root.bind('<Left>', world.keyboard_left)
    root.bind('<Right>', world.keyboard_right)
    root.bind('s', world.save)

    def close_window(_):
        """ Closes the GUI window. """
        root.destroy()

    root.bind('q', close_window)

    world.update()

    root.mainloop()
