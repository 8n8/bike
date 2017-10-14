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


def test_main_3():
    smallobstacle: w.Obstacle = {
        'position': {
            'x': 0,
            'y': 0},
        'velocity': {
            'x': 5,
            'y': 0},
        'radius': 0.5}
    largeobstacle: w.Obstacle = {
        'position': {
            'x': 0,
            'y': 0},
        'velocity': {
            'x': 5,
            'y': 0},
        'radius': 1}
    assert v.main([smallobstacle], []) < v.main([largeobstacle], [])


def test_main_4():
    obstacleone: w.Obstacle = {
        'position': {
            'x': -150,
            'y': 150},
        'velocity': {
            'x': 5,
            'y': -5},
        'radius': 3}
    obstacletwo: w.Obstacle = {
        'position': {
            'x': 0,
            'y': 0},
        'velocity': {
            'x': 5,
            'y': 0},
        'radius': 3}
    assert v.main([obstacleone], []) > v.main([obstacletwo], [])


def test_main_5():
    obstacle: w.Obstacle = {
        'position': {
            'x': 0,
            'y': 0},
        'velocity': {
            'x': 5,
            'y': 0},
        'radius': 3}
    freeness1: v.Freeness = {
        'radius': 2,
        'position': {
            'x': -4,
            'y': -10},
        'free_time': 2}
    freeness2: v.Freeness = {
        'radius': 4,
        'position': {
            'x': -20,
            'y': 20},
        'free_time': 55}
    assert (v.main([obstacle], [freeness1, freeness2]) >
            v.main([obstacle], []))
