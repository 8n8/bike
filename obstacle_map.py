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
        k.layers.convolutional.Conv3D(
            15,  # number of output layers
            3,  # kernel size
            strides=(2, 2, 2),
            padding='same',
            activation='relu',
            input_shape=(200, 200, 5, 3),
            data_format='channels_last'),
        k.layers.convolutional.Conv3D(
            200,  # number of output layers
            3,  # kernel size
            strides=(2, 2, 2),
            padding='same',
            activation='relu',
            data_format='channels_last'),
        k.layers.core.Flatten()])

# Note that this config takes about 0.07s per run.  It has 82,430 parameters.
