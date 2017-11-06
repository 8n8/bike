""" It provides a function for creating a Tkinter GUI. """

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
    state = WorldState(
        crashed=init.crashed,
        velocity=init.velocity,
        position=init.position,
        target_velocity=init.target_velocity,
        obstacles=init.obstacles,
        keyboard=init.keyboard)
    data: List[DataPoint] = []
    the_time: float = 0
    while True:
        print('hoho')
        if state.crashed:
            print('Robot has crashed into obstacle.')
            root.destroy()
            return
        canvas.delete('all')
        for i in world2view(state):
            if isinstance(i, TkOval):
                canvas.create_oval(
                    i.top_left_x,
                    i.top_left_y,
                    i.bottom_right_x,
                    i.bottom_right_y,
                    fill=i.fill_colour)
                continue
            if isinstance(i, TkArrow):
                canvas.create_line(
                    i.start_x,
                    i.start_y,
                    i.stop_x,
                    i.stop_y,
                    arrow=k.LAST,
                    fill=i.colour,
                    width=i.width)
                continue
            if isinstance(i, TkImage):
                canvas.create_image(i.x, i.y, image=i.image)
        data.append(DataPoint(state=state, timestamp=the_time))
        the_time += timestep
        state = update_world(state, u.generate_params(), timestep)

    def keyboard_up(_):
        """
        Updates the world state because the up arrow key has been pressed.
        """
        state.keyboard = KeyPress.UP

    def keyboard_down(_):
        state.keyboard = KeyPress.DOWN

    def keyboard_left(_):
        state.keyboard = KeyPress.LEFT

    def keyboard_right(_):
        state.keyboard = KeyPress.RIGHT

    def save(_):
        """ It saves the data gathered so far to file. """
        filename: str = 'game_data/' + str(uuid.uuid4())
        with open(filename, 'a') as the_file:
            json.dump(data, the_file)

    root.bind("<Up>", keyboard_up)
    root.bind("<Down>", keyboard_down)
    root.bind("<Left>", keyboard_left)
    root.bind("<Right>", keyboard_right)
    root.bind("x", save)

    def close_window(_):
        """ Closes the GUI window. """
        root.destroy()

    root.bind("q", close_window)

    root.mainloop()
