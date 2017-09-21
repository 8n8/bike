import numpy as np
import simulate_world as s
import unittest as u

def world1():
    return {
        'obstacles': [],
        'x': 0,
        'y': 0,
        'velocity': 0,
        'orientation': 0,
        'lean': 0,
        'lean acceleration': 0,
        'steer': 0 }

def readings1():
    white_image = 255 * np.ones((100, 100, 3), dtype=uint8)
    set_of_white_images = {
        '0.1': white_image,
        '0.3': white_image,
        '0.9': white_image,
        '2.7': white_image,
        '8.1': white_image }
    return {
        'cameras': {
            'front': set_of_white_images,
            'left': set_of_white_images,
            'right': set_of_white_images,
            'back': set_of_white_images},
        'lean acceleration': 0,
        'steer': 0,
        'velocity': 0,
        'gps': {'x': 0, 'y': 0} }

class test_calculate_sensor_readings(u.TestCase):
    
    def test_types(self):
        self.assertEqual(
            s.calculate_sensor_readings(world1()),
            readings1())

u.main()
