import tensorflow
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D,Flatten
import numpy as np
from tensorflow.keras import Model


# functional approach : function that returns a model
def functional_model():
    # while using the functional approach we need to pass the output of the previous layer as an input of the next layer
    # so we store the output of each layer in a variable and pass it as a parameter for the next layer

    my_input = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu')(my_input)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = tensorflow.keras.Model(inputs=my_input, outputs=x)

    return model


# tensorflow.keras.Model : inherit from this class
class MyCustomModel(tensorflow.keras.Model):

    def __init__(self):
        super().__init__()

        self.conv1 = Conv2D(32, (3, 3), activation='relu')
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.maxpool1 = MaxPool2D()
        self.batchnorm1 = BatchNormalization()

        self.conv3 = Conv2D(128, (3, 3), activation='relu')
        self.maxpool2 = MaxPool2D()
        self.batchnorm2 = BatchNormalization()

        self.globalavgpool1 = GlobalAvgPool2D()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

    # a function that will call all the layer components defined above
    def call(self, my_input):

        # follow the same procedure as described in functional way
        x = self.conv1(my_input)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)
        x = self.globalavgpool1(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x


def streetsigns_model(no_of_classes):
    my_input = Input(shape=(60, 60, 3))

    x = Conv2D(32, (3, 3), activation='relu')(my_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    # x = GlobalAvgPool2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(no_of_classes, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)


if __name__ == '__main__':
    model = streetsigns_model(10)
    model.summary()

# if __name__ == '__main__':
#
#     (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
#
#     print("x_train.shape = ", x_train.shape)
#     print("y_train.shape = ", y_train.shape)
#     print("x_test.shape = ", x_test.shape)
#     print("y_test.shape = ", y_test.shape)
#
#     # if False:
#     # display_some_examples(x_train, y_train)
#
#     # Normalising the Training and Testing data bw 0 and 1 so that GD converges faster to the global minima of Cost function
#     # We need to convert the unsigned data to float
#     x_train = x_train.astype('float32') / 255
#     x_test = x_test.astype('float32') / 255
#
#     # Making training and testing data compatible with input dimensions that it can accept
#     # The following is equivalent to x[:, np.newaxis] i.e. we are adding a new dimension at the end
#     x_train = np.expand_dims(x_train, axis=-1)
#     x_test = np.expand_dims(x_test, axis=-1)
#
#     # Modifying y_train and y_test to one hot encoding if we want to use categorical_crossentropy loss fucntion
#     y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
#     y_test = tensorflow.keras.utils.to_categorical(y_test, 10)
#
#     # # model = functional_model()
#     model = MyCustomModel()
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
#
#     # model training
#     # batch size represents how many images our model will see each time
#     # So batch_size=64 means our model will see 64 images (64 training examples) each time
#     # One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE
#     # validation_split=0.2 means 20% of training data will be used for cross validation
#     model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2)
#
#     # Evaluation on test set
#     model.evaluate(x_test, y_test, batch_size=64)

