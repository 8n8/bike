""" It provides a function for creating a Tkinter GUI. """

import os
import uuid
from typing import Callable, List
import tkinter as k
from keras.models import load_model  # type: ignore
import numpy as np  # type: ignore
import update_obstacle_pop as u
import game_functions as g
import random


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
            world2view: Callable[[g.WorldState], List[g.TkPicture]]):

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
        i = self.world2view(self.state)
        self.canvas.create_image(250, 250, image=i.image)
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
        newv = g.Velocity(speed=random.uniform(3, 7), angle=0)
        self.state = g.WorldState(
            crashed=self.state.crashed,
            velocity=newv,
            position=self.state.position,
            target_velocity=newv,
            obstacles=self.state.obstacles,
            keyboard=self.state.keyboard,
            timestamp=self.state.timestamp,
            thin_view=self.state.thin_view)
        filename: str = 'game_data/' + str(uuid.uuid4())
        dat: g.DataSet = g.prepare_for_save(
                self.recent_images, self.history)
        np.savez(filename, dat.images, dat.target_velocity,
                 dat.velocity_in, dat.velocity_out)


def main(
        init: g.WorldState,
        timestep: float,
        update_world: g.UpdateWorld,
        world2view: Callable[[g.WorldState], List[g.TkPicture]]):
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
    root.geometry('500x500')
    canvas = k.Canvas(root, width=500, height=500, bg='grey')
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
