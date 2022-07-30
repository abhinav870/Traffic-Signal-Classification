import os
import glob
from sklearn.model_selection import train_test_split
import shutil
import tensorflow as tf
from my_utils import splitdata,order_test_set,create_generators
from MNIST_functional_and_class import streetsigns_model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

if __name__=="__main__":

    path_to_train = "C:\\Users\\GARG\\PycharmProjects\\Python Codes\\CNN Project\\GSTRB Dataset\\training_data\\train"
    path_to_val = "C:\\Users\\GARG\\PycharmProjects\\Python Codes\\CNN Project\\GSTRB Dataset\\training_data\\val"
    path_to_test = "C:\\Users\\GARG\\PycharmProjects\\Python Codes\\CNN Project\\GSTRB Dataset\\Test"
    batch_size = 64
    epochs = 2
    learning_rate = 0.0001

    train_generator,val_generator,test_generator = create_generators(batch_size,path_to_train,path_to_val,path_to_test)
    no_of_classes = train_generator.num_classes

    TRAIN = False
    TEST = True

    if TRAIN:
        # ModelCheckPoint is a callback method that saves the best model during training
        # It will save the current model if its validation accuracy is more than the previous model's validation accuracy.
        # For this purpose we have set monitor = "val_accuracy" and mode="max"
        # We can also set monitor = "val_loss", mode = "min" i.e. saving the model based on least validation loss
        path_to_save_model = "./Best_Model"
        check_point_saver = ModelCheckpoint(
                                            path_to_save_model,
                                            monitor = "val_accuracy",
                                            mode = "max",
                                            save_best_only = True,
                                            # this saves only the best model and not all those models whose accuracy is more tahn the previous models
                                            save_freq = "epoch",
                                            # we will be checking the validation accuracy at the end of epoch only
                                            verbose = 1
                                            # Used for debugging purposes,to check if our model has been saved or not
                                            )
        # suppose we are training our model for 100 epochs and after 15 epochs our accuracy doesn't improve
        # example: bw 16th and 35th epochs our accuracy is same, then we stop the training
        early_stop = EarlyStopping(monitor="val_accuracy",patience=20)


        model = streetsigns_model(no_of_classes)
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate,amsgrad = True)

        # Using categorical_loss_entropy because we have set class_mode="categorical" in train, test and val generator
        model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=['accuracy'])

        # we can pass full generators as they contain the images with correct labels
        # flow_from_directory() has done that for us
        model.fit(train_generator,
                  epochs = epochs,
                  batch_size = batch_size,
                  validation_data = val_generator,
                  callbacks=[check_point_saver]
                  # setting the 2 callback methods to save the best model during the training
                  )
        # in our earlier implementation, we had to explicitly mention the training data and its labels, but here that task
        # is performed by train_generator

    # saver = tf.train.Saver(tf.model_variables())
    if TEST:

        # Our model is saved in this script. So we load it from its correct path
        model = tf.keras.models.load_model('./Best_Model')

        summary = model.summary()
        print("Printing Model Summary:")
        print(summary)

        print("Evaluating Validation Set:")
        model.evaluate(val_generator)

        print("Evaluating Test Set:")
        model.evaluate(test_generator)
