import numpy as np
import obstacle_map as o


def testTime():
    a = np.zeros((1, 1))
    assert o.main(a, a, a, a, 1).shape == (100, 100)
