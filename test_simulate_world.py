import math as m
import numpy as np
from PIL import Image  # type: ignore
import simulate_world as s
import unittest as u


def world1() -> s.WorldState:
    return {
        'obstacles': [],
        'x': 0,
        'y': 0,
        'velocity': 0,
        'orientation': 0,
        'lean': 0,
        'lean_acceleration': 0,
        'steer': 0}


def world2() -> s.WorldState:
    return {
        'obstacles': [
            {
                'x': 0,
                'y': 0.2,
                'radius': 0.1}],
        'x': 0,
        'y': 0,
        'velocity': 0,
        'orientation': m.pi / 2,
        'lean': 0,
        'lean_acceleration': 0,
        'steer': 0}


def world3() -> s.WorldState:
    return {
        'obstacles': [
            {
                'x': 0.2,
                'y': 0,
                'radius': 0.1}],
        'x': 0,
        'y': 0,
        'velocity': 0,
        'orientation': m.pi / 2,
        'lean': 0,
        'lean_acceleration': 0,
        'steer': 0}


def readings3() -> s.SensorReadings:
    white_image: 'np.ndarray[np.uint8]' = (  # type: ignore
        255 * np.ones((100, 100, 3), dtype=np.uint8))
    right_image = np.concatenate((  # type: ignore
        255 * np.ones((100, 21, 3), dtype=np.uint8),
        np.zeros((100, 57, 3), dtype=np.uint8),
        255 * np.ones((100, 22, 3), dtype=np.uint8)), axis=1)
    return {
        'cameras': {
            'front': white_image,
            'left': white_image,
            'right': right_image,
            'back': white_image},
        'lean_acceleration': 0,
        'steer': 0,
        'velocity': 0,
        'gps': {'x': 0, 'y': 0}}


def readings2() -> s.SensorReadings:
    white_image: 'np.ndarray[np.uint8]' = (  # type: ignore
        255 * np.ones((100, 100, 3), dtype=np.uint8))
    front_image: 'np.ndarray[np.uint8]' = np.concatenate((  # type: ignore
        255 * np.ones((100, 21, 3), dtype=np.uint8),
        np.zeros((100, 57, 3), dtype=np.uint8),
        255 * np.ones((100, 22, 3), dtype=np.uint8)), axis=1)
    return {
        'cameras': {
            'front': front_image,
            'left': white_image,
            'right': white_image,
            'back': white_image},
        'lean_acceleration': 0,
        'steer': 0,
        'velocity': 0,
        'gps': {'x': 0, 'y': 0}}


def readings1() -> s.SensorReadings:
    white_image: 'np.ndarray[np.uint8]' = (  # type: ignore
        255 * np.ones((100, 100, 3), dtype=np.uint8))
    return {
        'cameras': {
            'front': white_image,
            'left': white_image,
            'right': white_image,
            'back': white_image},
        'lean_acceleration': 0,
        'steer': 0,
        'velocity': 0,
        'gps': {'x': 0, 'y': 0}}


def equal_nparray(
        one: 'np.ndarray[np.uint8]',
        two: 'np.ndarray[np.uint8]'
        ) -> bool:
    equalness: 'np.ndarray[bool]' = one == two  # type: ignore
    answer: bool = equalness.all()  # type: ignore
    return answer


def equal_sensor_state(
        one: s.SensorReadings,
        two: s.SensorReadings
        ) -> bool:
    onecams: s.ImageSet = one['cameras']
    twocams: s.ImageSet = two['cameras']
    if not equal_nparray(onecams['front'], twocams['front']):
        print('Front image was not correct.')
        return False
    if not equal_nparray(onecams['left'], twocams['left']):
        img = Image.fromarray(onecams['left'], 'RGB')
        img.show()
        print('Left-hand image was not correct.')
        return False
    if not equal_nparray(onecams['right'], twocams['right']):
        print('Right-hand image was not correct.')
        return False
    if not equal_nparray(onecams['back'], twocams['back']):
        print('Rear image was not correct.')
        return False
    if one['velocity'] != two['velocity']:
        print('Velocity was not correct.')
        return False
    if one['lean_acceleration'] != two['lean_acceleration']:
        print('Lean acceleration was not correct.')
        return False
    if one['steer'] != two['steer']:
        print('Steer angle was not correct.')
        return False
    if one['gps']['x'] != two['gps']['x']:
        print('GPS x-coordinate was not correct.')
        return False
    if one['gps']['y'] != two['gps']['y']:
        print('GPS y-coordinate was not correct.')
        return False
    return True


class test_calculate_sensor_readings(u.TestCase):

    def test_calculate_sensor_readings(self):
        # self.assertTrue(equal_sensor_state(
        #     s.calculate_sensor_readings(world1()),
        #     readings1()))
        # self.assertTrue(equal_sensor_state(
        #     s.calculate_sensor_readings(world2()),
        #     readings2()))
        self.assertTrue(equal_sensor_state(
            s.calculate_sensor_readings(world3()),
            readings3()))


u.main()
