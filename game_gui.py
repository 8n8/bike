"""
It provides a function for creating a Tkinter GUI.
"""

from enum import Enum, auto, unique
import json
import math
import tkinter as k
import game_gui as g
from typing import Any, Callable, NamedTuple, List, Union, Tuple
import uuid
import data_gen_game as dg
import numpy as np
from keras.models import load_model  # type: ignore
import train_nav_net as tr
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


def update_keyboard(key: KeyPress, s: WorldState) -> WorldState:
    return WorldState(
        crashed=s.crashed,
        velocity=s.velocity,
        position=s.position,
        target_velocity=s.target_velocity,
        obstacles=s.obstacles,
        keyboard=key)


def last4(L):
    if len(L) < 4:
        return L
    return L[-4:]


def convert_data_point(d: DataPoint) -> dg.DataPoint:
    return {
        'world': {
            'velocity': {
                'speed': d.state.velocity.speed,
                'angle': d.state.velocity.angle},
            'position': d.state.position,
            'obstacles': d.state.obstacles},
        'target_velocity': {
            'speed': d.state.target_velocity.speed,
            'angle': d.state.target_velocity.angle},
        'timestamp': d.timestamp}


def convert_data_points(ds: List[DataPoint]) -> List[dg.DataPoint]:
    return [convert_data_point(d) for d in ds]


model = load_model('nav_net_Adam_001_3e5_catcross.h5')


def array2velocity(arr) -> Velocity:
    return Velocity(
        speed=arr[0][0]*20 - 10,
        angle=arr[0][1]*2*math.pi)


def pad_data_point(oldest: dg.DataPoint, timestep: float) -> dg.DataPoint:
    return {
        'world': oldest['world'],
        'target_velocity': oldest['target_velocity'],
        'timestamp': oldest['timestamp'] - timestep}


def pad_data(
        ds: List[dg.DataPoint],
        timestep: float) -> List[dg.DataPoint]:
    if ds == []:
        return []
    lends = len(ds)
    if lends > 1200:
        return ds
    for i in range(1200-lends):
        oldest = ds[0]
        ds = [pad_data_point(oldest, timestep)] + ds
    return ds


def i_for_n_seconds_ago(
        timestamps: 'np.ndarray[np.float64]',
        n: float,
        now: float) -> int:
    """
    It finds the index of the timestamp that is close to being n
    seconds ago.
    """
    comparison = np.ones_like(timestamps)*(now - n)
    matching = abs(timestamps - comparison) < 0.1
    indices = matching.nonzero()
    if len(indices[0]) == 0:
        return None
    return indices[0][0]
    # for i, t in enumerate(timestamps):
    #     if isclose(t, now - n): 
    #         return i
    # return None


def isclose(a: float, b: float) -> bool:
    return abs(a - b) < 0.1


def make_batch(
        ds: List[g.DataPoint],
        timestamps: 'np.ndarray[np.float64]',
        i: int
        ) -> Tuple[str, 'np.ndarray[np.uint8]']:
    """
    It makes a single data point for the neural network.  The
    network takes in 5 images going back in time between 0.1
    and 8.1 seconds.
    """
    now: float = timestamps[i]
    times: List[float] = [2.7, 0.9, 0.3, 0.1]
    i_s: List[int] = [i_for_n_seconds_ago(timestamps, t, now) for t in times]
    nones: List[bool] = [i is None for i in i_s]
    if any(nones):
        return "Not possible to make this batch.", None
    batch: List['np.ndarray[np.uint8]'] = [tr.worldstate2images(ds[i]['world']) for i in i_s]
    result = None, np.stack(batch, axis=2)  # type: ignore
    return result


def convert_data(
        data_batch: List[g.DataPoint]
        ) -> Tuple[str, tr.TrainingData]:
    """
    It converts the data from the game format into numpy arrays
    ready for feeding into the neural network.
    """
    vs: List['np.ndarray[np.float64]'] = [
        tr.velocity2array(i['world']['velocity']) for i in data_batch]
    target_vs: List['np.ndarray[np.float64]'] = [
        tr.velocity2array(i['target_velocity']) for i in data_batch]
    timestamps: List[float] = [i['timestamp'] for i in data_batch]
    numpoints: int = len(timestamps)
    imbatches_with_errors_and_indices: IndexErrImage = []
    np_timestamps = np.array(timestamps, dtype=np.float32)
    for i in range(numpoints):
        batch = (i, make_batch(data_batch, np_timestamps, i))
        if batch[1][0] is None:
            imbatches_with_errors_and_indices.append(batch)
            break
    i_s: Set[int] = {i for i, (err, _) in imbatches_with_errors_and_indices
                     if err is None}
    vs_used: List['np.ndarray[np.float64]'] = [
        v for i, v in enumerate(vs) if i in i_s]
    target_vs_used: List['np.ndarray[np.float64]'] = [
        t for i, t in enumerate(target_vs) if i in i_s]
    imbatches: List['np.ndarray[np.uint8]'] = [
        i for _, (err, i) in imbatches_with_errors_and_indices
        if err is None]
    if imbatches == []:
        return "No useful data in batch.", None
    image_in: 'np.ndarray[np.uint8]' = (
        np.stack([imbatches[-1]], axis=0))  # type: ignore
    v_in: 'np.ndarray[np.float64]' = (
        np.stack(target_vs_used, axis=0))  # type: ignore
    v_out: 'np.ndarray[np.float64]' = np.stack(vs_used, axis=0)  # type: ignore
    return (None, {
        'image_in': image_in,
        'v_in': v_in,
        'v_out': v_out})


def predict_velocity(
        ds: List[dg.DataPoint],
        timestep: float) -> Tuple[str, Velocity]:
    err, dat = convert_data(ds)
    if err is not None:
        return err, None
    result = model.predict(
        {'image_in': dat['image_in'],
         'velocity_in': dat['v_in']},
        batch_size=1)
    return None, array2velocity(result)


def update_velocity(v: Velocity, s: WorldState) -> WorldState:
    return WorldState(
        crashed=s.crashed,
        velocity=v,
        position=s.position,
        target_velocity=s.target_velocity,
        obstacles=s.obstacles,
        keyboard=s.keyboard)


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
        self.counter = 0

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
        self.counter += 1
        if self.counter % 100 == 0:
            chopped = self.data[-300:]
            err, velocity = predict_velocity(
                convert_data_points(chopped),
                self.timestep)
            if err is None:
                self.state = update_velocity(velocity, self.state)
        if self.counter > 1000:
            print('ending')
            return
        self.canvas.after(1, self.update)

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
