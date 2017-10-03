import keras as k
from mypy_extensions import TypedDict
from keras.optimizers import SGD


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


def neural_net():
    return k.models.Sequential([
        k.layers.convolutional.Conv3D(
            5,  # number of output layers
            4,  # kernel size
            strides=(1, 1, 1),
            padding='same',
            activation='relu',
            input_shape=(200, 200, 3, 1),
            data_format='channels_last'),
        k.layers.convolutional.Conv3D(
            10,
            3,
            strides=(2, 2, 2),
            padding='same',
            activation='relu'),
        k.layers.convolutional.Conv3D(
            15,
            3,
            strides=(2, 2, 2),
            padding='same',
            activation='relu'),
        k.layers.convolutional.Conv3D(
            20,
            3,
            strides=(2, 2, 2),
            padding='same',
            activation='relu'),
        k.layers.core.Flatten(),
        k.layers.normalization.BatchNormalization(),
        k.layers.core.Dense(11040, activation='relu'),
        k.layers.core.Dense(11040, activation='relu')])


def main(i):
    nn = neural_net()
    return nn.predict(i)
