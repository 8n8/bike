import numpy as np
import obstacle_map as o


def test_output_shape():
    i = np.random.randint(0, 256, size=(1, 200, 200, 5, 3), dtype=np.uint8)
    assert o.main(i).shape == (1, 11040)

#     i = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
#     is = {
#         'left': i,
#         'right': i,
#         'back': i,
#         'right': i}
#     images = {
#         't100': is,
#         't300': is,
#         't900': is,
#         't2700': is,
#         't8100': is}
#     assert o.main(images).shape == (100, 100)
