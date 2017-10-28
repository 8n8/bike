import keras as k  # type: ignore
import numpy as np
import time


conv1 = k.layers.convolutional.Conv3D(
    5,
    3,
    strides=(2, 2, 2),
    padding='same',
    activation='relu',
    data_format='channels_last')


conv2 = k.layers.convolutional.Conv3D(
    10,
    3,
    strides=(2, 2, 2),
    padding='same',
    activation='relu',
    data_format='channels_last')


conv3 = k.layers.convolutional.Conv3D(
    20,
    3,
    strides=(2, 2, 2),
    padding='same',
    activation='relu',
    data_format='channels_last')

conv4 = k.layers.convolutional.Conv3D(
    20,
    3,
    strides=(2, 2, 2),
    padding='same',
    activation='relu',
    data_format='channels_last')

conv5 = k.layers.convolutional.Conv3D(
    20,
    3,
    strides=(2, 2, 2),
    padding='same',
    activation='relu',
    data_format='channels_last')

conv6 = k.layers.convolutional.Conv3D(
    20,
    3,
    strides=(2, 2, 2),
    padding='same',
    activation='relu',
    data_format='channels_last')

conv7 = k.layers.convolutional.Conv3D(
    25,
    3,
    strides=(2, 2, 2),
    padding='same',
    activation='relu',
    data_format='channels_last')

flat = k.layers.core.Flatten()

dense1 = k.layers.Dense(50, activation='relu')

dense2 = k.layers.Dense(25, activation='relu')

dense3 = k.layers.Dense(2, activation='softmax')


def velnet():
    imin = k.layers.Input(shape=(200, 200, 5, 3), name='image_in')
    velin = k.layers.Input(shape=(2,), name='velocity_in')
    conv = flat(conv7(conv6(conv5(conv4(conv3(conv2(conv1(imin))))))))
    midin = k.layers.concatenate([conv, velin])
    dense = dense3(dense2(dense1(midin)))
    return k.models.Model(inputs=[imin, velin], outputs=[dense])


def main():
    iin = np.random.rand(1, 200, 200, 5, 3)  # type: ignore
    vin = np.random.rand(1, 2)  # type: ignore

    start = time.time()
    num = 100
    model = velnet()
    model.summary()
    for _ in range(num):
        model.predict({'image_in': iin, 'velocity_in': vin})
    print((time.time() - start) / num)
