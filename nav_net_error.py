from mypy_extensions import TypedDict
from typing import List
import world2sensor as w


# It defines a circle on the ground with the amount of time it will be
# free to drive over.
class Freeness(TypedDict):
    radius: float
    position: w.Point
    free_time: float


def main(
        obstacles: List[w.Obstacle],
        freenesses: List[Freeness]
        ) -> float:
    return sum([
        _compare_freeness_with_obstacle(f, o)
        for f in freenesses
        for o in obstacles])


def _compare_freeness_with_obstacle(
        freeness: Freeness,
        obstacle: w.Obstacle
        ) -> float:
    if _obstacle_intersects_freeness(freeness, obstacle):
        return abs(
            (freeness['free_time'] -
             _time_till_collision(freeness, obstacle)) /
            _proportion_of_freeness_swept_by_obstacle(freeness, obstacle))


def _obstacle_intersects_freeness(
        freeness: Freeness,
        obstacle: w.obstacle
        ) -> bool:
    
