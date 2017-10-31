"""
It is a simple game that is played autonomously with a neural network.
"""

import json
import math
import random
import time
import uuid
from typing import List
import tkinter as k
import numpy as np
import train_nav_net as tr
import update_obstacle_pop as u
import world2sensor as w
from keras.models import load_model  # type: ignore
from mypy_extensions import TypedDict
from PIL import ImageTk, Image  # type: ignore


XOFFSET: float = 330
YOFFSET: float = 800
SCALE: float = 8


class Velocity(TypedDict):
    """ It defines the polar velocity. """
    angle: float  # between 0 and 2π
    speed: float


class WorldState(TypedDict):
    """ The state of the world. """
    velocity: Velocity
    position: w.Vector
    obstacles: List[w.Obstacle]


class DataPoint(TypedDict):
    """ It defines a single data point. """
    world: WorldState
    target_velocity: Velocity
    timestamp: float


def angle_mod(angle: float) -> float:
    """ It makes sure the angle is between 0 and 2π. """
    if angle < 0:
        return angle_mod(angle + 2 * math.pi)
    return angle % (2 * math.pi)


def speed_mod(speed: float) -> float:
    """ It makes sure the speed stays in the interval (-10, 10). """
    if speed > 10:
        return 10
    if speed < -10:
        return -10
    return speed


def update_velocity(key: str, velocity: Velocity) -> Velocity:
    """ It changes the velocity in response to a key press. """
    speed_step = 0.7
    angle_step = math.pi/26
    if key == 'up':
        return {
            'angle': velocity['angle'],
            'speed': speed_mod(velocity['speed'] + speed_step)}
    if key == 'down':
        return {
            'angle': velocity['angle'],
            'speed': speed_mod(velocity['speed'] - speed_step)}
    if key == 'left':
        return {
            'angle': angle_mod(velocity['angle'] + angle_step),
            'speed': velocity['speed']}
    if key == 'right':
        return {
            'angle': angle_mod(velocity['angle'] - angle_step),
            'speed': velocity['speed']}
    return velocity


def polar_velocity_to_cartesian(velocity: Velocity) -> w.Vector:
    """ It converts a polar two-vector to a cartesian one. """
    return {
        'x': velocity['speed'] * math.cos(velocity['angle']),
        'y': velocity['speed'] * math.sin(velocity['angle'])}


def numpy_to_single_tkimage(image):
    """
    It converts the image from a numpy array to the format for
    displaying in Tkinter.
    """
    return ImageTk.PhotoImage(
        Image.fromarray(image).resize((200, 200), Image.ANTIALIAS))


def numpy_to_tkimages(i: w.ImageSet):
    """
    It converts a set of four camera images from the four cameras from
    m x n x 3 numpy arrays to the right format for displaying in
    Tkinter.
    """
    return {
        'front': numpy_to_single_tkimage(i['front']),
        'left': numpy_to_single_tkimage(i['left']),
        'back': numpy_to_single_tkimage(i['back']),
        'right': numpy_to_single_tkimage(i['right'])}


def make_images(state: WorldState):
    """
    It calculates the images from the worldstate and converts them into
    the correct format for displaying in a Tkinter window.
    """
    return numpy_to_tkimages(w.calculate_images(
        state['obstacles'],
        state['position']['x'],
        state['position']['y'],
        state['velocity']['angle']))


def update_world(
        state: WorldState,
        step: float,
        obstacle_params: List[u.ObstacleParams],
        max_new_obstacles: int) -> WorldState:
    """
    It calculates the next state of the world, given the previous one and
    a time interval.
    """
    # cos(ϴ) = x / mag
    #      x = mag * cos(ϴ)
    velx = state['velocity']['speed'] * math.cos(state['velocity']['angle'])
    vely = state['velocity']['speed'] * math.sin(state['velocity']['angle'])
    return {
        'velocity': state['velocity'],
        'position': {
            'x': state['position']['x'] + step * velx,
            'y': state['position']['y'] + step * vely},
        'obstacles': u.main(
            state['obstacles'],
            step,
            state['position'],
            max_new_obstacles,
            obstacle_params)}


def distance_between(one: w.Vector, two: w.Vector) -> float:
    """ It calculates the distance between two points. """
    return ((one['x'] - two['x'])**2 + (one['y'] - two['y'])**2)**0.5


def crashed_into_obstacle(state: WorldState) -> bool:
    """ It works out if the robot has crashed into an obstacle. """
    return any([
        distance_between(state['position'], o['position']) < 2
        for o in state['obstacles']])


def chop_data(datapoints: List[DataPoint]) -> List[DataPoint]:
    """
    It shortens the list of data points to save unnecessary processing.
    """
    tnew: float = datapoints[-1]['timestamp']
    needed: List[bool] = [tnew - d['timestamp'] < 9 for d in datapoints]
    return [d for d, good in zip(datapoints, needed) if good]


def pad_data(datapoints: List[DataPoint]) -> List[DataPoint]:
    """
    It increases the length of the list of data points so that
    it is long enough to be processed by the neural network.
    """
    newest: float = datapoints[-1]['timestamp']
    oldest: float = datapoints[0]['timestamp']
    if oldest - newest > 9:
        return datapoints
    extra_needed: float = 9 - oldest + newest
    extra_timestamps: List[float] = [
        oldest + 0.1 * i for i in range(int(extra_needed/0.1))]
    return [{
        'world': {
            'velocity': {
                'angle': datapoints[0]['world']['velocity']['angle'],
                'speed': 0},
            'position': datapoints[0]['world']['position'],
            'obstacles': []},
        'target_velocity': datapoints[0]['target_velocity'],
        'timestamp': t} for t in extra_timestamps]


class World:
    """
    It represents the state of the game and the various methods
    for manipulating it.
    """
    def __init__(self, canvas, root):
        self.state = {
            'velocity': {
                'angle': 0,
                'speed': 0},
            'position': {
                'x': 0,
                'y': 0},
            'obstacles': []}
        self.canvas = canvas
        self.images = make_images(self.state)
        self.target_v = {
            'speed': - random.uniform(0, 10),
            'angle': math.pi/2}
        self.data = []
        self.root = root
        self.model = load_model('nav_net1.h5')

    def update(self):
        """ It updates the GUI window. """
        datapoint: DataPoint = {
            'world': self.state,
            'target_velocity': self.target_v,
            'timestamp': time.time()}
        self.data.append(datapoint)
        rate = 0.04
        self.canvas.delete('all')
        if crashed_into_obstacle(self.state):
            print('Robot has crashed into obstacle.')
            return
        plot_objects(self.canvas, self.state)
        self.calculate_velocity()
        cart_target_v = {'x': self.target_v['speed'], 'y': 0}
        draw_arrows(self.canvas, self.state['velocity'], cart_target_v)
        max_new_obstacles, obstacle_params = u.generate_params()
        self.state = update_world(
            self.state, rate, obstacle_params, max_new_obstacles)
        self.images = make_images(self.state)
        self.canvas.create_image(320, 110, image=self.images['front'])
        self.canvas.create_image(320, 330, image=self.images['back'])
        self.canvas.create_image(110, 220, image=self.images['left'])
        self.canvas.create_image(530, 220, image=self.images['right'])
        self.canvas.after(int(1/rate), self.update)

    def calculate_velocity(self):
        """ It uses the neural network to decide on the velocity. """
        self.data = pad_data(chop_data(self.data))
        err, dat = tr.convert_data(self.data)
        if err is not None:
            return
        imin = np.expand_dims(dat['image_in'][0], axis=0)
        vin = np.expand_dims(dat['v_in'][0], axis=0)
        npvel = self.model.predict(
            {'image_in': imin,
             'velocity_in': vin},
            batch_size=1)[0]
        print(npvel)
        self.state['velocity']['speed'] = (npvel[0] * 20) - 10
        self.state['velocity']['angle'] = npvel[1] * 2 * math.pi


def save(data):
    """ It writes the game data to a file in json format. """
    filename = 'game_data/' + str(uuid.uuid4())
    with open(filename, 'a') as file_handle:
        json.dump(data, file_handle)


def circle(canvas, x_coord, y_coord, radius, colour):
    """ It draws a circle on the canvas in the GUI window. """
    canvas.create_oval(
        x_coord - radius,
        y_coord - radius,
        x_coord + radius,
        y_coord + radius,
        fill=colour)


def draw_arrow(
        canvas,
        velocity: w.Vector,
        colour: str):
    """ It draws an arrow in the GUI window at (XOFFSET, YOFFSET). """
    startx: float = XOFFSET + SCALE*(-velocity['x'])
    starty: float = YOFFSET - SCALE*(-velocity['y'])
    stopx: float = XOFFSET + SCALE*velocity['x']
    stopy: float = YOFFSET - SCALE*velocity['y']
    canvas.create_line(startx, starty, stopx, stopy, arrow=k.LAST,
                       fill=colour, width=1)


def draw_arrows(
        canvas,
        actual_v: Velocity,
        target_v: w.Vector):
    """
    It draws the arrows in the GUI window that represent the actual
    velocity and the target velocity.
    """
    # The rotation matrix is:
    #
    #     [ cos ϴ   - sin ϴ ]
    #     [ sin ϴ   cos ϴ   ]
    #
    # So the rotated position is:
    #
    #    [ x cos ϴ - y sin ϴ ]
    #    [ x sin ϴ + y cos ϴ ]
    theta = actual_v['angle'] + math.pi/2
    sintheta = math.sin(-theta)
    costheta = math.cos(-theta)
    x_coord = target_v['x']
    y_coord = target_v['y']
    rotated: w.Vector = {
        'x': x_coord * costheta - y_coord * sintheta,
        'y': x_coord * sintheta + y_coord * costheta}
    actual_v_cart: w.Vector = {
        'x': 0,
        'y': actual_v['speed']}
    draw_arrow(canvas, actual_v_cart, 'red')
    draw_arrow(canvas, rotated, 'black')


def subtract_vector(one: w.Vector, two: w.Vector) -> w.Vector:
    """ It subtracts one vector from another. """
    return {
        'x': one['x'] - two['x'],
        'y': one['y'] - two['y']}


def plot_obstacle(canvas, obstacle: w.Obstacle):
    """ It plots an obstacle on the GUI window. """
    circle(
        canvas,
        SCALE * obstacle['position']['x'] + XOFFSET,
        - SCALE * obstacle['position']['y'] + YOFFSET,
        SCALE * obstacle['radius'],
        'black')


def update_obstacle(
        obstacle: w.Obstacle,
        bikepos: w.Vector,
        bikevel: Velocity) -> w.Obstacle:
    """
    It repositions the obstacles such that the frame of reference
    is the robot.
    """
    shifted: w.Vector = {
        'x': obstacle['position']['x'] - bikepos['x'],
        'y': obstacle['position']['y'] - bikepos['y']}
    # The rotation matrix is:
    #
    #     [ cos ϴ   - sin ϴ ]
    #     [ sin ϴ   cos ϴ   ]
    #
    # So the rotated position is:
    #
    #    [ x cos ϴ - y sin ϴ ]
    #    [ x sin ϴ + y cos ϴ ]
    theta = bikevel['angle'] - math.pi/2
    sintheta = math.sin(-theta)
    costheta = math.cos(-theta)
    x_coord = shifted['x']
    y_coord = shifted['y']
    rotated: w.Vector = {
        'x': x_coord * costheta - y_coord * sintheta,
        'y': x_coord * sintheta + y_coord * costheta}
    return {
        'position': rotated,
        'velocity': obstacle['velocity'],
        'radius': obstacle['radius']}


def plot_objects(canvas, world: WorldState):
    """ It plots the obstacles and the robot on the GUI window. """
    circle(canvas, XOFFSET, YOFFSET, SCALE * 1.0, 'red')
    centred_obstacles = [
        update_obstacle(obstacle, world['position'], world['velocity'])
        for obstacle in world['obstacles']]
    for obstacle in centred_obstacles:
        plot_obstacle(canvas, obstacle)


def main():
    """ It creates the GUI window. """
    root = k.Tk()
    root.title('NavNet data generator game')
    root.geometry('800x600')
    canvas = k.Canvas(root, width=650, height=1000, bg='grey')
    canvas.pack()

    world = World(canvas, root)

    # def exit(_):
    #     save(world.data)

    # root.bind("<Up>", world.increase_velocity)
    # root.bind("<Down>", world.decrease_velocity)
    # root.bind("<Left>", world.velocity_left)
    # root.bind("<Right>", world.velocity_right)
    # root.bind("x", exit)

    world.update()

    root.mainloop()
