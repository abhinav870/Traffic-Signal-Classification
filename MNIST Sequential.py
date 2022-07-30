import tensorflow
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D
from my_utils import display_some_examples


"""
1. BATCH NORMALIZATION is a layer that allows every layer of the network to do learning more independently.
It is used to normalize the output of the previous layers.
The layer is added to the sequential model to standardize the input or the outputs. 
It can be used at several points in between the layers of the model
"""

"""
2. FLATTENING is converting the data into a 1-dimensional array for inputting it to the next layer.
We flatten the output of the convolutional layers to create a single long feature vector.
And it is connected to the final classification model, which is called a fully-connected layer.
"""

"""
3. GLOBAL AVERAGE POOLING is a pooling operation designed to replace fully connected layers in classical CNNs.
The idea is to generate one feature map for each corresponding category of the classification task
in the last mlpconv layer.
"""

"""
4. Difference Bw epochs and batch size
The batch size is a hyperparameter of gradient descent that controls the number of training samples to work through
before the model’s internal parameters are updated.

The number of epochs is a hyperparameter of gradient descent that controls the number of complete passes through 
the training dataset.
"""

# Sequential

'''
Note: For setting the hyperparamters of different layers of our CNN model, there is no fixed defined manner.
      Based on our observations and experiments (which architecture yields more accuracy) we build and train our model 
'''

model = tensorflow.keras.Sequential(
        [
            # we have grayscale images with one channel and there are total 28*28 images
            # Unlike other layers in our Architecture we don't have many options to experiment with here.
            # As we have grayscale images with 1 channel only and dimensions of img is 28*28
            # Hence we have to use that ony in the input layer
            Input(shape=(28,28,1)),

            # we have first 2D convolutional as input layer with 32 filters each of size 3*3 and relu activation layer
            Conv2D(32,(3,3),activation="relu"),

            # second convolutional layer
            Conv2D(64,(3,3),activation="relu"),

            # Perform max pooling
            MaxPool2D(),

            # BATCH NORMALIZATION is a layer that allows every layer of the network to do learning more independently.
            # It is used to normalize the output of the previous layers.
            # The layer is added to the sequential model to standardize the input or the outputs.
            # using Batch Normalization helps the GD to converge faster towards the global minima of cost function
            BatchNormalization(),

            Conv2D(128,(3,3),activation="relu"),
            MaxPool2D(),
            BatchNormalization(),

            # GLOBAL AVERAGE POOLING is a pooling operation designed to replace fully connected layers in classical CNNs
            # It takes the output from the BatchNormalization() layer and then computes the avg of those values acc to
            # some axes
            GlobalAvgPool2D(),

            # Dense layer is same as FC(Fully Connected) layer
            # Values from GlobalAvgPool2D() are fed to Dense Layer.This layer doesn't contain any filters and hence no
            # convolution operation is performed
            Dense(64,activation="relu"),

            # This Dense Layer is the Output layer of our CNN
            # We choose 10 = number of possible output classes acc to our MNIST dataset
            # Using Softmax Activation fn we get probability of an image belonging to a particular class,
            # hence our output will be the class having the maximum probability
            # Unlike other initial layers or the previous Dense() layer we don't have much options to experiment with in
            # the last layer as we have to classify the img out of 0-9 only.Hence we have to use
            # Softmax Activation function for probabilistic estimation of our image
            Dense(10,activation="softmax")
        ]
    )

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

    print("x_train.shape = ", x_train.shape)
    print("y_train.shape = ", y_train.shape)
    print("x_test.shape = ", x_test.shape)
    print("y_test.shape = ", y_test.shape)

    # if False:
    display_some_examples(x_train, y_train)

    # Normalising the Training and Testing data bw 0 and 1 so that GD converges faster to the global minima of Cost function
    # We need to convert the unsigned data to float
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Making training and testing data compatible with input dimensions that it can accept
    # The following is equivalent to x[:, np.newaxis] i.e. we are adding a new dimension at the end
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Modifying y_train and y_test to one hot encoding if we want to use categorical_crossentropy loss fucntion
    y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
    y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

    # # model = functional_model()
    # model = MyCustomModel()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

    # model training
    # batch size represents how many images our model will see each time
    # So batch_size=64 means our model will see 64 images (64 training examples) each time
    # One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE
    # validation_split=0.2 means 20% of training data will be used for cross validation
    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2)

    # Evaluation on test set
    model.evaluate(x_test, y_test, batch_size=64)