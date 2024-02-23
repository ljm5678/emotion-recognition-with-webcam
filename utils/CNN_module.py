# import packages
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import ELU
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2
from keras import backend as K

class CNN_module:
    @staticmethod
    def build(width, height, depth, classes, reg = 0.0005):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we use "channels first", update the input shape and channels dimensions
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # Block #1: first CONV => ELU => CONV => ELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding = "same",
            kernel_initializer = "he_normal", input_shape = inputShape))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(32, (3, 3), padding = "same",
            kernel_initializer = "he_normal"))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        # Block #2: second CONV => ELU => CONV => ELU => POOL layer set
        model.add(Conv2D(64, (3, 3), padding = "same",
            kernel_initializer = "he_normal", input_shape = inputShape))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(64, (3, 3), padding = "same",
            kernel_initializer = "he_normal"))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))




        # Block #3: third CONV => ELU => CONV => ELU => POOL
        model.add(Conv2D(128, (3, 3), padding = "same",
            kernel_initializer = "he_normal", input_shape = inputShape))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(128, (3, 3), padding = "same",
            kernel_initializer = "he_normal"))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        # Block #4: second set of FC => ELU layer set
        model.add(Flatten())
        model.add(Dense(64, kernel_initializer = "he_normal"))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(Dropout(0.5))

        # Block #5: second set of FC => ELU layer set
        model.add(Dense(64, kernel_initializer = "he_normal"))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(Dropout(0.5))

        # Block #6: softmax classifier
        model.add(Dense(classes, kernel_initializer = "he_normal"))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model