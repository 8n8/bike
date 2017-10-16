import numpy as np
import obstacle_map as o
import time


def test_output_shape():
    nn = o.first_net()
    nn.summary()
    i = np.random.randint(0, 256, size=(1, 200, 200, 5, 3), dtype=np.uint8)
    nn.predict(i)
    start = time.time()
    num = 100
    for _ in range(num):
        i = np.random.randint(0, 256, size=(1, 200, 200, 5, 3), dtype=np.uint8)
        nn.predict(i)
    stop = time.time()
    print('time taken: {}'.format((stop - start)/num))


test_output_shape()
