""" It builds the navigation neural network. """


import keras as k  # type: ignore


conv1 = k.layers.convolutional.Conv1D(
    6,
    5,
    strides=1,
    padding='same',
    activation='relu')


conv2 = k.layers.convolutional.Conv1D(
    10,
    5,
    strides=1,
    padding='same',
    activation='relu')


conv3 = k.layers.convolutional.Conv1D(
    10,
    5,
    strides=1,
    padding='same',
    activation='relu')


conv4 = k.layers.convolutional.Conv1D(
    10,
    3,
    strides=1,
    padding='same',
    activation='relu')


conv5 = k.layers.convolutional.Conv1D(
    10,
    3,
    strides=1,
    padding='same',
    activation='relu')


flat = k.layers.core.Flatten()

dense1 = k.layers.Dense(20, activation='relu')

dense2 = k.layers.Dense(2, activation='softmax')

bn = k.layers.BatchNormalization


def main():
    """ It creates the neural network. """
    imin = k.layers.Input(shape=(100, 1), name='image_in')
    velin = k.layers.Input(shape=(2,), name='velocity_in')
    targetin = k.layers.Input(shape=(2,), name='target_in')
    conv = flat(conv5(bn()(conv4(bn()(conv3(
        bn()(conv2(bn()(conv1(bn()(imin)))))))))))
    midin = k.layers.concatenate([conv, targetin, velin])
    dense = dense2(bn()(dense1(bn()(midin))))
    return k.models.Model(inputs=[imin, targetin, velin],
                          outputs=[dense])
