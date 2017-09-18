import simulate_world as s
import unittest as u

def world1():
    return {
        'obstacles': [],
        'bike_x': 0,
        'bike_y': 0,
        'bike_orientation': 0,
        'bike_lean': 0,
        'lean_acceleration': 0,
        'bike_steer': 0 }

def readings1():
    return {
        'front': [
            {0: np.ones

class test_calculate_sensor_readings(u.TestCase):
    
    def test_types(self):
        self.assertEqual(
            s.calculate_sensor_readings(world1()),
            readings1())

unittest.main()
