import json
import math
from mypy_extensions import TypedDict
from PIL import ImageTk, Image  # type: ignore
import random
import time
import tkinter as k
from typing import List
import update_obstacle_pop as u
import uuid
import world2sensor as w


XOFFSET: float = 330
YOFFSET: float = 800
SCALE: float = 8


class Velocity(TypedDict):
    angle: float  # between 0 and 2π
    speed: float


class WorldState(TypedDict):
    velocity: Velocity
    position: w.Vector
    obstacles: List[w.Obstacle]


class DataPoint(TypedDict):
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


def update_velocity(key: str, v: Velocity) -> Velocity:
    """ It changes the velocity in response to a key press. """
    speed_step = 0.7 
    angle_step = math.pi/26
    if key == 'up':
        return {
            'angle': v['angle'],
            'speed': speed_mod(v['speed'] + speed_step)}
    if key == 'down':
        return {
            'angle': v['angle'],
            'speed': speed_mod(v['speed'] - speed_step)}
    if key == 'left':
        return {
            'angle': angle_mod(v['angle'] + angle_step),
            'speed': v['speed']}
    if key == 'right':
        return {
            'angle': angle_mod(v['angle'] - angle_step),
            'speed': v['speed']}
    return v


def polar_velocity_to_cartesian(v: Velocity) -> w.Vector:
    return {
        'x': v['speed'] * math.cos(v['angle']),
        'y': v['speed'] * math.sin(v['angle'])}


def numpy_to_TKimage(i: w.ImageSet):
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


def make_images(s: WorldState):
    """
    It calculates the images from the worldstate and converts them into
    the correct format for displaying in a Tkinter window.
    """
    return numpy_to_TKimage(w._calculate_images(
        s['obstacles'],
        s['position']['x'],
        s['position']['y'],
        s['velocity']['angle']))


def update_world(
        s: WorldState,
        t: float,
        obstacle_params: List[u.ObstacleParams],
        max_new_obstacles: int
        ) -> WorldState:
    """
    It calculates the next state of the world, given the previous one and
    a time interval.
    """
    # cos(ϴ) = x / mag
    #      x = mag * cos(ϴ)
    velx = s['velocity']['speed'] * math.cos(s['velocity']['angle'])
    vely = s['velocity']['speed'] * math.sin(s['velocity']['angle'])
    return {
        'velocity': s['velocity'],
        'position': {
            'x': s['position']['x'] + t * velx,
            'y': s['position']['y'] + t * vely},
        'obstacles': u.main(
            s['obstacles'],
            t,
            s['position'],
            max_new_obstacles,
            obstacle_params)}


def distance_between(a: w.Vector, b: w.Vector) -> float:
    return ((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2)**0.5


def crashed_into_obstacle(w: WorldState) -> bool:
    """ It works out if the robot has crashed into an obstacle. """
    return any([
        distance_between(w['position'], o['position']) < 2
        for o in w['obstacles']])


class World:
    def __init__(self, canvas, root):
        self.w = {
            'velocity': {
                'angle': 0,
                'speed': 0},
            'position': {
                'x': 0,
                'y': 0},
            'obstacles': []}
        self.canvas = canvas
        self.images = make_images(self.w)
        self.target_v = {
            'speed': - random.uniform(0, 10),
            'angle': math.pi/2}
        self.data = []
        self.root = root

    def update(self):
        datapoint: DataPoint = {
            'world': self.w,
            'target_velocity': self.target_v,
            'timestamp': time.time()}
        self.data.append(datapoint)
        rate = 0.04
        self.canvas.delete('all')
        if crashed_into_obstacle(self.w):
            print('Robot has crashed into obstacle.')
            return
        plot_objects(self.canvas, self.w)
        cart_target_v = {'x': self.target_v['speed'], 'y': 0}
        draw_arrows(self.canvas, self.w['velocity'], cart_target_v,
                    self.w['position'])
        max_new_obstacles, obstacle_params = u.generate_params()
        self.w = update_world(
            self.w, rate, obstacle_params, max_new_obstacles)
        self.images = make_images(self.w)
        self.canvas.create_image(320, 110, image=self.images['front'])
        self.canvas.create_image(320, 330, image=self.images['back'])
        self.canvas.create_image(110, 220, image=self.images['left'])
        self.canvas.create_image(530, 220, image=self.images['right'])
        self.canvas.after(int(1/rate), self.update)

    def increase_velocity(self, _):
        self.w['velocity'] = update_velocity('up', self.w['velocity'])

    def decrease_velocity(self, _):
        self.w['velocity'] = update_velocity('down', self.w['velocity'])

    def velocity_left(self, _):
        self.w['velocity'] = update_velocity('left', self.w['velocity'])

    def velocity_right(self, _):
        self.w['velocity'] = update_velocity('right', self.w['velocity'])


def save(data):
    filename = 'game_data/' + str(uuid.uuid4())
    with open(filename, 'a') as f:
        json.dump(data, f)


def circle(canvas, x, y, r, colour):
    canvas.create_oval(x - r, y - r, x + r, y + r, fill=colour)


def draw_arrow(
        canvas,
        v: w.Vector,
        p: w.Vector,
        colour: str):
    startx: float = XOFFSET + SCALE*(-v['x'])
    starty: float = YOFFSET - SCALE*(-v['y'])
    stopx: float = XOFFSET + SCALE*v['x']
    stopy: float = YOFFSET - SCALE*v['y']
    canvas.create_line(startx, starty, stopx, stopy, arrow=k.LAST,
                       fill=colour, width=1)


def draw_arrows(
        canvas,
        actual_v: Velocity,
        target_v: w.Vector,
        p: w.Vector):
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
    x = target_v['x']
    y = target_v['y']
    rotated: w.Vector = {
        'x': x * costheta - y * sintheta,
        'y': x * sintheta + y * costheta}
    actual_v_cart: w.Vector = {
        'x': 0,
        'y': actual_v['speed']}
    draw_arrow(canvas, actual_v_cart, p, 'red')
    draw_arrow(canvas, rotated, p, 'black')


def subtract_vector(a: w.Vector, b: w.Vector) -> w.Vector:
    return {
        'x': a['x'] - b['x'],
        'y': a['y'] - b['y']}


def plot_obstacle(canvas, o: w.Obstacle):
    circle(
        canvas,
        SCALE * o['position']['x'] + XOFFSET,
        - SCALE * o['position']['y'] + YOFFSET,
        SCALE * o['radius'],
        'black')


def update_obstacle(
        o: w.Obstacle,
        bikepos: w.Vector,
        bikevel: Velocity
        ) -> w.Obstacle:
    shifted: w.Vector = {
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
    theta = bikevel['angle'] - math.pi/2
    sintheta = math.sin(-theta)
    costheta = math.cos(-theta)
    x = shifted['x']
    y = shifted['y']
    rotated: w.Vector = {
        'x': x * costheta - y * sintheta,
        'y': x * sintheta + y * costheta}
    return {
        'position': rotated,
        'velocity': o['velocity'],
        'radius': o['radius']}


def plot_objects(canvas, s: WorldState):
    circle(canvas, XOFFSET, YOFFSET, SCALE * 1.0, 'red')
    centred_obstacles = [update_obstacle(o, s['position'], s['velocity'])
                         for o in s['obstacles']]
    [plot_obstacle(canvas, o) for o in centred_obstacles]


def main():
    root = k.Tk()
    root.title('NavNet data generator game')
    root.geometry('800x600')
    canvas = k.Canvas(root, width=650, height=1000, bg='grey')
    canvas.pack()

    world = World(canvas, root)

    def exit(_):
        save(world.data)

    root.bind("<Up>", world.increase_velocity)
    root.bind("<Down>", world.decrease_velocity)
    root.bind("<Left>", world.velocity_left)
    root.bind("<Right>", world.velocity_right)
    root.bind("x", exit)

    world.update()

    root.mainloop()
