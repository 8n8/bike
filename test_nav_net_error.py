import nav_net_error as v
import world2sensor as w  # noqa: F401


def test_main_simple():
    assert v.main([], []) >= 0


def test_main_1():
    obstacle: w.Obstacle = {
        'position': {
            'x': 0,
            'y': 0},
        'velocity': {
            'x': 5,
            'y': 0},
        'radius': 0.5}
    smallfreeness: v.Freeness = {
        'radius': 0.1,
        'position': {
            'x': 2,
            'y': 0},
        'free_time': 2}
    largefreeness: v.Freeness = {
        'radius': 0.3,
        'position': {
            'x': 2,
            'y': 0},
        'free_time': 2}
    assert (v.main([obstacle], [largefreeness]) <
            v.main([obstacle], [smallfreeness]))


def test_main_2():
    slowobstacle: w.Obstacle = {
        'position': {
            'x': 0,
            'y': 0},
        'velocity': {
            'x': 5,
            'y': 0},
        'radius': 0.5}
    fastobstacle: w.Obstacle = {
        'position': {
            'x': 0,
            'y': 0},
        'velocity': {
            'x': 10,
            'y': 0},
        'radius': 0.5}
    freeness: v.Freeness = {
        'radius': 0.1,
        'position': {
            'x': 2,
            'y': 0},
        'free_time': 2}
    assert (v.main([slowobstacle], [freeness]) <
            v.main([fastobstacle], [freeness]))
