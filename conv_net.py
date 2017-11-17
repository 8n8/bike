""" It builds the navigation neural network. """


import keras as k  # type: ignore


conv1 = k.layers.convolutional.Conv2D(
    3,
    5,
    strides=(2, 2),
    padding='same',
    activation='relu')


conv2 = k.layers.convolutional.Conv2D(
    5,
    5,
    strides=(2, 2),
    padding='same',
    activation='relu')


conv3 = k.layers.convolutional.Conv2D(
    5,
    5,
    strides=(2, 2),
    padding='same',
    activation='relu')


conv4 = k.layers.convolutional.Conv2D(
    5,
    3,
    padding='same',
    activation='relu')


conv5 = k.layers.convolutional.Conv2D(
    5,
    3,
    padding='same',
    activation='relu')


flat = k.layers.core.Flatten()

dense0 = k.layers.Dense(1164, activation='relu')

dense1 = k.layers.Dense(100, activation='relu')

dense2 = k.layers.Dense(50, activation='relu')

dense3 = k.layers.Dense(10, activation='relu')

dense4 = k.layers.Dense(2, activation='softmax')

bn = k.layers.BatchNormalization


def main():
    """ It creates the neural network. """
    imin = k.layers.Input(shape=(100, 4, 4), name='image_in')
    velin = k.layers.Input(shape=(2,), name='velocity_in')
    targetin = k.layers.Input(shape=(2,), name='target_in')
    conv = flat(conv5(bn()(conv4(bn()(conv3(
        bn()(conv2(bn()(conv1(bn()(imin)))))))))))
    midin = k.layers.concatenate([conv, targetin, velin])
    dense = dense4(bn()(dense3(bn()(dense2(bn()(dense1(bn()(dense0(bn()(
        midin))))))))))
    return k.models.Model(inputs=[imin, targetin, velin], outputs=[dense])
