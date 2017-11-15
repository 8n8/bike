""" It evaluates the neural network.  """


import evaluate_net
from keras.models import load_model  # type: ignore

for _ in range(10):
    print(evaluate_net.main(load_model("nav_net.h5")))
