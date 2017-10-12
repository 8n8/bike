import cProfile
import nav_net_error as v
import world2sensor as w  # noqa: F401


def main():
    obstacle: w.Obstacle = {
        'position': {
            'x': 0,
            'y': 0},
        'velocity': {
            'x': 5,
            'y': 0},
        'radius': 0.5}
    # smallfreeness: v.Freeness = {
    #     'radius': 0.1,
    #     'position': {
    #         'x': 2,
    #         'y': 0},
    #     'free_time': 2}
    largefreeness: v.Freeness = {
        'radius': 0.3,
        'position': {
            'x': 2,
            'y': 0},
        'free_time': 2}
    v.main([obstacle], [largefreeness])


cProfile.run('main()')
