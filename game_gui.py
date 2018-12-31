""" It provides a function for creating a Tkinter GUI. """

import os
import uuid
from typing import Callable, List
import tkinter as k
from keras.models import load_model  # type: ignore
import numpy as np
import update_obstacle_pop as u
import game_functions as g


if os.path.isfile(g.MODEL_FILE):
    MODEL = load_model(g.MODEL_FILE)
else:
    MODEL = None


class _World:
    """
    It holds the state of the world and various functions for updating it.
    """
    def __init__(
            self,
            canvas,
            init: g.WorldState,
            timestep: float,
            update_world: g.UpdateWorld,
            world2view: Callable[[g.WorldState], List[g.TkPicture]]) -> None:
        self.state = init
        self.canvas = canvas
        self.timestep = timestep
        self.history: List[g.WorldState] = []
        self.world2view = world2view
        self.update_world = update_world
        self.time = 0
        self.imref = []  # type: ignore
        self.recent_images: List['np.ndarray[bool]'] = []

    def update(self):
        """ It updates the gui window and the world state. """
        if self.state.crashed:
            print('Robot has crashed into obstacle.')
            return
        self.canvas.delete('all')
        for i in self.world2view(self.state):
            if isinstance(i, g.TkOval):
                self.canvas.create_oval(
                    i.top_left_x,
                    i.top_left_y,
                    i.bottom_right_x,
                    i.bottom_right_y,
                    fill=i.fill_colour)
                continue
            if isinstance(i, g.TkArrow):
                self.canvas.create_line(
                    i.start_x,
                    i.start_y,
                    i.stop_x,
                    i.stop_y,
                    arrow=k.LAST,
                    fill=i.colour,
                    width=i.width)
                continue
            if isinstance(i, g.TkImage):
                self.canvas.create_image(i.x, i.y, image=i.image)
                self.imref.append(i.image)
                self.imref = self.imref[-4:]
        self.history.append(self.state)
        recent_images, self.state = self.update_world(
            self.history,
            u.generate_params(),
            self.timestep,
            MODEL)
        self.recent_images.append(recent_images)
        self.time += self.timestep
        self.canvas.after(int(1000 * self.timestep), self.update)

    def keyboard_up(self, _):
        """ Records the pressing of the up arrow key. """
        self.state = g.update_keyboard(g.KeyPress.UP, self.state)

    def keyboard_down(self, _):
        """ Records the pressing of the down arrow key. """
        self.state = g.update_keyboard(g.KeyPress.DOWN, self.state)

    def keyboard_left(self, _):
        """ Records the pressing of the left arrow key. """
        self.state = g.update_keyboard(g.KeyPress.LEFT, self.state)

    def keyboard_right(self, _):
        """ Records the pressing of the right arrow key. """
        self.state = g.update_keyboard(g.KeyPress.RIGHT, self.state)

    def save(self, _):
        """ It saves the data gathered so far to file. """
        filename: str = 'game_data/' + str(uuid.uuid4())
        dat: g.DataSet = g.prepare_for_save(self.recent_images, self.history)
        np.savez(filename, dat.images, dat.target_velocity,
                 dat.velocity_in, dat.velocity_out)


def main(
        init: g.WorldState,
        timestep: float,
        update_world: g.UpdateWorld,
        world2view: Callable[[g.WorldState], List[g.TkPicture]]) -> None:
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
