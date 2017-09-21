import numpy as np
import simulate_world as s
from typing import (Any, Dict, List, Union)
import unittest as u


def world1() -> Dict[str, Union[float, List[Dict[str, float]]]]:
    return {
        'obstacles': [],
        'x': 0,
        'y': 0,
        'velocity': 0,
        'orientation': 0,
        'lean': 0,
        'lean acceleration': 0,
        'steer': 0}


Sensors = Dict[str, Any]


def readings1() -> Sensors:
    white_image = 255 * np.ones(
        (100, 100, 3), dtype=np.uint8)
    set_of_white_images = {
        '0.1': white_image,
        '0.3': white_image,
        '0.9': white_image,
        '2.7': white_image,
        '8.1': white_image}
    return {
        'cameras': {
            'front': set_of_white_images,
            'left': set_of_white_images,
            'right': set_of_white_images,
            'back': set_of_white_images},
        'lean acceleration': 0,
        'steer': 0,
        'velocity': 0,
        'gps': {'x': 0, 'y': 0}}


def equal_sensor_state(
        one: Sensors,
        two: Sensors
        ) -> bool:
    onecams: Dict[str, Any] = one['cameras']
    twocams: Dict[str, Any] = two['cameras']
    for location in ['front', 'left', 'right', 'back']:
        test = not (onecams[location] == twocams[location]).all()
        if test:
            print('camera {} was not as expected'.format(location))
            print(type(onecams[location]))
            print(type(twocams[location]))
            return False
    for key, _ in one:
        if key == 'cameras':
            continue
        if one[key] != two[key]:
            print('{} was not as expected'.format(key))
            return False
    return True


class test_calculate_sensor_readings(u.TestCase):

    def test_types(self):
        self.assertTrue(equal_sensor_state(
            s.calculate_sensor_readings(world1()),
            readings1()))


u.main()
