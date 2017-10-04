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
            activation='relu')])
        #k.layers.normalization.BatchNormalization(),  # (1, 3380)
#        k.layers.core.Dense(3000, activation='relu')])
#        k.layers.core.Flatten()])
#        k.layers.normalization.BatchNormalization(),
#        k.layers.core.Dense(6000, activation='relu'),
#        k.layers.core.Dense(3000, activation='relu')])


def second_net(): 
    return k.models.Sequential([
        k.layers.core.Dense(
            5000,
            activation='relu',
            input_shape=(2940, 5))])
        # k.layers.core.Dense(5000, activation='relu')])


def main(i):
    nn = neural_net()
    return nn.predict(i)
