import keras as k  # type: ignore
from mypy_extensions import TypedDict
import numpy as np  # noqa: F401


class ImageOverTime(TypedDict):
    t100: 'np.ndarray[np.uint8]'
    t300: 'np.ndarray[np.uint8]'
    t900: 'np.ndarray[np.uint8]'
    t2700: 'np.ndarray[np.uint8]'
    t8100: 'np.ndarray[np.uint8]'


class ImagesOverTime(TypedDict):
    left: ImageOverTime
    right: ImageOverTime
    front: ImageOverTime
    back: ImageOverTime


def first_net():
    return k.models.Sequential([
        k.layers.convolutional.Conv2D(
            5,  # number of output layers
            3,  # kernel size
            strides=(1, 1),
            padding='same',
            activation='relu',
            input_shape=(200, 200, 3),
            data_format='channels_last'),
        k.layers.convolutional.Conv2D(
            10,
            3,
            strides=(2, 2),
            padding='same',
            activation='relu'),
        k.layers.convolutional.Conv2D(
            15,
            3,
            strides=(2, 2),
            padding='same',
            activation='relu'),
        k.layers.convolutional.Conv2D(
            20,
            3,
            strides=(2, 2),
            padding='same',
            activation='relu'),
        k.layers.convolutional.Conv2D(
            30,
            3,
            strides=(2, 2),
            padding='same',
            activation='relu'),
        k.layers.convolutional.Conv2D(
            60,
            3,
            strides=(2, 2),
            padding='same',
            activation='relu'),
        k.layers.normalization.BatchNormalization(),
        k.layers.core.Flatten()])


def second_net():
    return k.models.Sequential([
        k.layers.convolutional.Conv2D(
            20,  # number of output layers
            3,  # kernel size
            strides=(2, 2),
            padding='same',
            activation='relu',
            input_shape=(2940, 5, 1)),
        k.layers.normalization.BatchNormalization(),
        k.layers.core.Flatten(),
        k.layers.core.Dense(5000, activation='relu')])
