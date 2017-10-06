import numpy as np
import obstacle_map as o
import time


def gen_input1():
    return np.random.randint(
        0,
        256,
        size=(1, 200, 200, 3),
        dtype=np.uint8)


def gen_input2():
    return np.random.randint(
        0,
        256,
        size=(1, 2940, 5),
        dtype=np.uint8)


numruns = 100
inputs = [gen_input1() for _ in range(numruns)]
nn = o.first_net()
start = time.time()
for i in inputs:
    print(nn.predict(i).shape)
print((time.time() - start)/numruns)
