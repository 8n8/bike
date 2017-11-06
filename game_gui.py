"""
It provides a function for creating a Tkinter GUI.
"""

from enum import Enum, auto, unique
import json
import tkinter as k
from typing import Any, Callable, NamedTuple, List, Union
import uuid
import update_obstacle_pop as u
import world2sensor as w


class Velocity(NamedTuple):
    angle: float  # between 0 and 2π
    speed: float


@unique
class KeyPress(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    NONE = auto()


class WorldState(NamedTuple):
    crashed: bool
    velocity: Velocity
    position: w.Vector
    target_velocity: Velocity
    obstacles: List[w.Obstacle]
    keyboard: KeyPress


class DataPoint(NamedTuple):
    state: WorldState
    timestamp: float


class TkOval(NamedTuple):
    top_left_x: float
    top_left_y: float
    bottom_right_x: float
    bottom_right_y: float
    fill_colour: str


class TkArrow(NamedTuple):
    start_x: float
    start_y: float
    stop_x: float
    stop_y: float
    colour: str
    width: float


class TkImage(NamedTuple):
    image: Any
    x: float
    y: float


TkPicture = Union[TkOval, TkArrow, TkImage]


UpdateWorld = Callable[[WorldState, u.RandomData, float], WorldState]


def update_keyboard(k: KeyPress, w: WorldState) -> WorldState:
    return WorldState(
        crashed=w.crashed,
        velocity=w.velocity,
        position=w.position,
        target_velocity=w.target_velocity,
        obstacles=w.obstacles,
        keyboard=k)


def last4(L):
    if len(L) < 4:
        return L
    return L[-4:]


def close_window(root):
    root.destroy('all')


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
        self.data: List[DataPoint] = []
        self.world2view = world2view
        self.update_world = update_world
        self.time = 0
        self.images: List[TkImage] = []

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
                self.images.append(i)
                self.images = last4(self.images)
                self.canvas.create_image(i.x, i.y, image=i.image)
        self.data.append(
            DataPoint(state=self.state, timestamp=self.time))
        self.state: WorldState = self.update_world(
            self.state,
            u.generate_params(),
            self.timestep)
        self.time += self.timestep
        self.canvas.after(int(self.timestep * 1000), self.update)

    def keyboard_up(self, _):
        """
        Updates the world state because the up arrow key has been pressed.
        """
        self.state = update_keyboard(KeyPress.UP, self.state)

    def keyboard_down(self, _):
        self.state = update_keyboard(KeyPress.DOWN, self.state)

    def keyboard_left(self, _):
        self.state = update_keyboard(KeyPress.LEFT, self.state)

    def keyboard_right(self, _):
        self.state = update_keyboard(KeyPress.RIGHT, self.state)

    def save(self, _):
        """ It saves the data gathered so far to file. """
        filename: str = 'game_data/' + str(uuid.uuid4())
        with open(filename, 'a') as the_file:
            json.dump(self.data, the_file)


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

    root.bind("<Up>", world.keyboard_up)
    root.bind("<Down>", world.keyboard_down)
    root.bind("<Left>", world.keyboard_left)
    root.bind("<Right>", world.keyboard_right)
    root.bind("x", world.save)

    def close_window(_):
        """ Closes the GUI window. """
        root.destroy()

    root.bind("q", close_window)

    world.update()

    root.mainloop()
