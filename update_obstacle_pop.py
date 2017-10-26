"""
It maintains the population of obstacles in the world, keeping a
reasonable number moving around near the bicycle and deleting ones
that are too far away.  The only function exposed is main().
"""


import math as m
import random
from typing import List
import world2sensor as w


def main(
        obstacle_list: List[w.Obstacle],
        t: float,
        bike_position: w.Vector,
        ) -> List[w.Obstacle]:
    """
    Given the current obstacles, a time period and the current bike
    position, it creates new obstacles near to the bike if necessary
    and removes the ones that are a long way away.
    """
    updated_obstacles: List[w.Obstacle] = _update_current_obstacles(
        obstacle_list,
        bike_position,
        t)
    new_obstacles: List[w.Obstacle] = _make_new_obstacles(
        _num_close_obstacles(bike_position, updated_obstacles),
        bike_position)
    return updated_obstacles + new_obstacles


def _num_close_obstacles(pos: w.Vector, obs: List[w.Obstacle]) -> int:
    return len([o for o in obs
                if _distance_between(o['position'], pos) < 40])


def _num_new_obstacles(current_obstacle_count: int) -> int:
    """ It decides how many new obstacles to make. """
    diff: int = random.randint(0, 20) - current_obstacle_count
    if diff < 0:
        return 0
    return diff


def _random_obstacle_position(bike_position: w.Vector) -> w.Vector:
    """ It randomly decides on a position for a new obstacle. """
    distance_from_bike: float = random.uniform(30, 50)
    angle: float = random.uniform(0, 2*m.pi)
    return {
        'x': bike_position['x'] + distance_from_bike * m.cos(angle),
        'y': bike_position['y'] + distance_from_bike * m.sin(angle)}


def _random_obstacle_velocity() -> w.Vector:
    """ It randomly decides on a velocity for a new obstacle. """
    magnitude: float = random.uniform(0, 5)
    direction: float = random.uniform(0, 2*m.pi)
    return {
        'x': magnitude * m.cos(direction),
        'y': magnitude * m.sin(direction)}


def _new_random_obstacle(bike_position: w.Vector) -> w.Obstacle:
    """ It randomly creates a new obstacle. """
    return {
        'position': _random_obstacle_position(bike_position),
        'velocity': _random_obstacle_velocity(),
        'radius': 0.5}


def _make_new_obstacles(
        num_obstacles: int,
        bike_position: w.Vector,
        ) -> List[w.Obstacle]:
    """
    It creates a random number of new obstacles, taking into account
    the number of existing obstacles and the position of the bicycle.
    """
    num_new: int = _num_new_obstacles(num_obstacles)
    return [
        _new_random_obstacle(bike_position) for _ in range(num_new)]


def _update_current_obstacles(
        obstacle_list: List[w.Obstacle],
        bike_position: w.Vector,
        t: float
        ) -> List[w.Obstacle]:
    """
    It deletes the obstacles that are too far away from the bicycle
    and calculates the new positions of the ones that are near by.
    """
    return [
        _move_obstacle(o, t) for o in obstacle_list
        if _obstacle_near_to_bike(bike_position, o)]


def _move_obstacle(
        obstacle_state: w.Obstacle,
        t: float
        ) -> w.Obstacle:
    """ It calculates the new positions of the existing obstacles. """
    newpos: w.Vector = {
        'x': (obstacle_state['position']['x'] +
              obstacle_state['velocity']['x'] * t),
        'y': (obstacle_state['position']['y'] +
              obstacle_state['velocity']['y'] * t)}
    return {
        'position': newpos,
        'velocity': obstacle_state['velocity'],
        'radius': obstacle_state['radius']}


def _distance_between(a: w.Vector, b: w.Vector) -> float:
    """ It calculates the distance between two points. """
    return ((b['x'] - a['x'])**2 + (b['y'] - a['y'])**2)**0.5


def _obstacle_near_to_bike(
        bike_position: w.Vector,
        obstacle: w.Obstacle
        ) -> bool:
    """ It decides if the obstacle is near to the bicycle or not. """
    return _distance_between(bike_position, obstacle['position']) < 150
