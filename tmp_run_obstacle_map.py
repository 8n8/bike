import numpy as np
import obstacle_map as o
import time

def gen_input():
    return np.random.randint(
        0,
        256,
        size=(1, 200, 200, 3, 1), 
        dtype=np.uint8)

numruns = 30
inputs = [gen_input() for _ in range(numruns)]
nn = o.neural_net()
start = time.time()
for i in inputs:
    print(nn.predict(i).shape)
print((time.time() - start)/numruns)
