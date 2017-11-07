import keras as k  # type: ignore
import numpy as np
import time


conv1 = k.layers.convolutional.Conv2D(
    5,
    3,
    strides=(2, 2),
    padding='same',
    activation='relu')


conv2 = k.layers.convolutional.Conv2D(
    10,
    3,
    strides=(2, 2),
    padding='same',
    activation='relu')


conv3 = k.layers.convolutional.Conv2D(
    20,
    3,
    strides=(2, 2),
    padding='same',
    activation='relu')


conv4 = k.layers.convolutional.Conv2D(
    30,
    3,
    strides=(2, 2),
    padding='same',
    activation='relu')


conv5 = k.layers.convolutional.Conv2D(
    40,
    3,
    strides=(2, 2),
    padding='same',
    activation='relu')


conv6 = k.layers.convolutional.Conv2D(
    30,
    3,
    strides=(2, 2),
    padding='same',
    activation='relu')


flat = k.layers.core.Flatten()

dense1 = k.layers.Dense(50, activation='relu')

dense2 = k.layers.Dense(25, activation='relu')

dense3 = k.layers.Dense(2, activation='softmax')

bn = k.layers.BatchNormalization


def velnet():
    imin = k.layers.Input(shape=(100, 4, 4), name='image_in')
    velin = k.layers.Input(shape=(2,), name='velocity_in')
    conv = flat(bn()(conv6(bn()(conv5(bn()(conv4(bn()(conv3(
        bn()(conv2(bn()(conv1(bn()(imin))))))))))))))
    midin = k.layers.concatenate([conv, velin])
    dense = dense3(bn()(dense2(bn()(dense1(bn()(midin))))))
    return k.models.Model(inputs=[imin, velin], outputs=[dense])


def main():
    iin = np.random.rand(1, 100, 4, 4)  # type: ignore
    vin = np.random.rand(1, 2)  # type: ignore

    start = time.time()
    num = 100
    model = velnet()
    model.summary()
    for _ in range(num):
        model.predict({'image_in': iin, 'velocity_in': vin})
    print((time.time() - start) / num)
