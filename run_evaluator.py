""" It evaluates the neural network.  """


import evaluate_net
from keras.models import load_model  # type: ignore

evaluate_net.main(load_model("nav_net.h5"))
