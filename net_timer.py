import func_net
from keras.models import load_model
import numpy as np
import time

num = 100

iins = [np.random.rand(1, 100, 4, 4) for _ in range(num)]
vins = [np.random.rand(1, 2) for _ in range(num)]

model = load_model('nav_net_Adam_001_3e5_catcross.h5')
model.summary()
start = time.time()
for vin, iin in zip(vins, iins): 
    model.predict({'image_in': iin, 'velocity_in': vin})
print((time.time() - start) / num)
