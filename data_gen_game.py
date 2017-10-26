import math
from mypy_extensions import TypedDict
from PIL import ImageTk, Image  # type: ignore
import tkinter as k
from typing import List
import update_obstacle_pop as u
import world2sensor as w


XOFFSET: float = 300
YOFFSET: float = 600
SCALE: float = 5


class Velocity(TypedDict):
    angle: float  # between 0 and 2π
    speed: float


def angle_mod(angle: float) -> float:
    if angle < 0:
        return angle_mod(angle + 2 * math.pi)
    return angle % (2 * math.pi)


def speed_mod(speed: float) -> float:
    if speed > 10:
        return 10
    if speed < -10:
        return -10
    return speed


def update_velocity(key: str, v: Velocity) -> Velocity:
    speed_step = 0.2
    angle_step = math.pi/36
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


class WorldState(TypedDict):
    velocity: Velocity
    position: w.Vector
    obstacles: List[w.Obstacle]


def numpy_to_TKimage(i: w.ImageSet):
    def f(im):
        return ImageTk.PhotoImage(
            Image.fromarray(im).resize((200, 200), Image.ANTIALIAS))
    return {
        'front': f(i['front']),
        'left': f(i['left']),
        'back': f(i['back']),
        'right': f(i['right'])}


def make_images(s: WorldState):
    return numpy_to_TKimage(w._calculate_images(
        s['obstacles'],
        s['position']['x'],
        s['position']['y'],
        s['velocity']['angle']))


def update_world(s: WorldState, t: float) -> WorldState:
    # cos(ϴ) = x / mag
    #      x = mag * cos(ϴ)
    velx = s['velocity']['speed'] * math.cos(s['velocity']['angle'])
    vely = s['velocity']['speed'] * math.sin(s['velocity']['angle'])
    return {
        'velocity': s['velocity'],
        'position': {
            'x': s['position']['x'] + t * velx,
            'y': s['position']['y'] + t * vely},
        'obstacles': u.main(s['obstacles'], t, s['position'])}


def distance_between(a: w.Vector, b: w.Vector) -> float:
    return ((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2)**0.5


def crashed_into_obstacle(w: WorldState) -> bool:
    return any([
        distance_between(w['position'], o['position']) < 2
        for o in w['obstacles']])


class World:
    def __init__(self, canvas):
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
        self.target_v = u._random_obstacle_velocity()

    def update(self):
        rate = 0.05
        self.canvas.delete('all')
        if crashed_into_obstacle(self.w):
            print('Robot has crashed into obstacle.')
            return
        plot_objects(self.canvas, self.w)
        draw_arrows(self.canvas, self.w['velocity'], self.target_v,
                    self.w['position'])
        self.w = update_world(self.w, rate)
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


def circle(canvas, x, y, r, colour):
    canvas.create_oval(x - r, y - r, x + 2 * r, y + 2 * r, fill=colour)


def draw_arrow(
        canvas,
        v: w.Vector,
        p: w.Vector,
        colour: str):
    print(v)
    startx: float = XOFFSET + SCALE*(p['x'] - (v['x']))
    starty: float = YOFFSET - SCALE*(p['y'] - (v['y']))
    stopx: float = XOFFSET + SCALE*(p['x'] + (v['x']))
    stopy: float = YOFFSET - SCALE*(p['y'] + (v['y']))
    canvas.create_line(startx, starty, stopx, stopy, arrow=k.LAST,
                       fill=colour, width=1)


def draw_arrows(
        canvas,
        actual_v: Velocity,
        target_v: w.Vector,
        p: w.Vector):
    actual_v_cart: w.Vector = {
        'x': actual_v['speed'] * math.cos(actual_v['angle']),
        'y': actual_v['speed'] * math.sin(actual_v['angle'])}
    draw_arrow(canvas, actual_v_cart, p, 'red')
    draw_arrow(canvas, target_v, p, 'black')


def plot_obstacle(canvas, o: w.Obstacle):
    circle(
        canvas,
        SCALE * o['position']['x'] + XOFFSET,
        - SCALE * o['position']['y'] + YOFFSET,
        SCALE * o['radius'],
        'black')


def plot_objects(canvas, w: WorldState):
    circle(
        canvas,
        XOFFSET + SCALE * w['position']['x'],
        YOFFSET - SCALE * w['position']['y'],
        SCALE * 1.0,
        'red')
    [plot_obstacle(canvas, o)
     for o in w['obstacles']]


def main():
    root = k.Tk()
    root.title('NavNet data generator game')
    root.geometry('800x600')
    canvas = k.Canvas(root, width=650, height=1000, bg='grey')
    canvas.pack()

    world = World(canvas)

    root.bind("<Up>", world.increase_velocity)
    root.bind("<Down>", world.decrease_velocity)
    root.bind("<Left>", world.velocity_left)
    root.bind("<Right>", world.velocity_right)

    world.update()

    root.mainloop()


main()
