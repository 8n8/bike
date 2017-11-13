""" It works out the time taken to run the navigation neural network. """

import time
import conv_net
import numpy as np

num = 100

iins = [np.random.rand(1, 100, 4, 4) for _ in range(num)]  # type: ignore
vins = [np.random.rand(1, 2) for _ in range(num)]  # type: ignore

model = conv_net.main()
model.summary()
start = time.time()
for vin, iin in zip(vins, iins):
    model.predict({'image_in': iin, 'velocity_in': vin})
print((time.time() - start) / num)
