""" It builds the navigation neural network. """


import keras as k  # type: ignore
from keras.layers import TimeDistributed as td


conv1 = td(k.layers.convolutional.Conv2D(
    10,
    3,
    strides=(2, 2),
    padding='same',
    activation='relu'))


conv2 = td(k.layers.convolutional.Conv2D(
    20,
    3,
    strides=(2, 2),
    padding='same',
    activation='relu'))


conv3 = td(k.layers.convolutional.Conv2D(
    40,
    3,
    strides=(2, 2),
    padding='same',
    activation='relu'))


conv4 = td(k.layers.convolutional.Conv2D(
    80,
    3,
    strides=(2, 2),
    padding='same',
    activation='relu'))


conv5 = td(k.layers.convolutional.Conv2D(
    80,
    3,
    strides=(2, 2),
    padding='same',
    activation='relu'))


conv6 = td(k.layers.convolutional.Conv2D(
    40,
    3,
    strides=(2, 2),
    padding='same',
    activation='relu'))


flat = td(k.layers.core.Flatten())

dense0 = k.layers.Dense(100, activation='relu')

dense1 = k.layers.Dense(50, activation='relu')

dense2 = k.layers.Dense(25, activation='relu')

dense3 = k.layers.Dense(2, activation='softmax')

bn = k.layers.BatchNormalization

gru = k.layers.GRU(200)

def main():
    """ It creates the neural network. """
    imin = k.layers.Input(shape=(100, 200, 4), name='image_in')
    overtime = gru(imin)
    # velin = k.layers.Input(shape=(2,), name='velocity_in')
    # targetin = k.layers.Input(shape=(2,), name='target_in')
    # conv = flat(conv6(bn()(conv5(bn()(conv4(bn()(conv3(
    #     bn()(conv2(bn()(conv1(bn()(overtime)))))))))))))
    # midin = k.layers.concatenate([conv, targetin, velin])
    # dense = dense1(bn()(gru(bn()(midin))))
    return k.models.Model(inputs=[imin], outputs=[overtime])
