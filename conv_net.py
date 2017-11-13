""" It builds the navigation neural network. """


import keras as k  # type: ignore


conv1 = k.layers.convolutional.Conv2D(
    5,
    5,
    strides=(2, 2),
    padding='same',
    activation='relu')


conv2 = k.layers.convolutional.Conv2D(
    10,
    5,
    strides=(2, 2),
    padding='same',
    activation='relu')


conv3 = k.layers.convolutional.Conv2D(
    20,
    5,
    strides=(2, 2),
    padding='same',
    activation='relu')


conv4 = k.layers.convolutional.Conv2D(
    30,
    5,
    strides=(2, 2),
    padding='same',
    activation='relu')


conv5 = k.layers.convolutional.Conv2D(
    40,
    5,
    strides=(2, 2),
    padding='same',
    activation='relu')


conv6 = k.layers.convolutional.Conv2D(
    60,
    5,
    strides=(2, 2),
    padding='same',
    activation='relu')


flat = k.layers.core.Flatten()

dense1 = k.layers.Dense(50, activation='relu')

dense2 = k.layers.Dense(25, activation='relu')

dense3 = k.layers.Dense(2, activation='softmax')

bn = k.layers.BatchNormalization


def main():
    """ It creates the neural network. """
    imin = k.layers.Input(shape=(100, 4, 4), name='image_in')
    velin = k.layers.Input(shape=(2,), name='velocity_in')
    targetin = k.layers.Input(shape=(2,), name='target_in')
    conv = flat(bn()(conv6(bn()(conv5(bn()(conv4(bn()(conv3(
        bn()(conv2(bn()(conv1(bn()(imin))))))))))))))
    midin = k.layers.concatenate([conv, targetin, velin])
    dense = dense3(bn()(dense2(bn()(dense1(bn()(midin))))))
    return k.models.Model(inputs=[imin, targetin, velin], outputs=[dense])