import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
from sklearn.model_selection import train_test_split
import glob
import csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def display_some_examples(examples,labels):

    # create an inch-by-inch image, which would be 10-by-10 pixels
    plt.figure(figsize=(10,10))

    for i in range(25):

        # np.random.randint() generates a random integer in the range [low,high), low inclusive and high exclusive
        idx = np.random.randint(0,examples.shape[0]-1)
        img = examples[idx]
        label_of_img = labels[idx]

        # we want to plot 25 images hence creating a grid of 5*5, i+1 is the img number
        plt.subplot(5,5,i+1)
        plt.title(str(label_of_img))

        # creates more space bw 5*5 grid of 25 images
        plt.tight_layout()
        plt.imshow(img,cmap="gray")
        # changes the background of each img to gray colour

    # finally display the grid of images
    plt.show()

# creating a function for splitting the data
# 90% of the data will be put in the training folder and remaining 10% data will be put in the validation folder
def splitdata(path_to_data,path_to_save_train,path_to_save_val,split_size = 0.1):

    # getting all the folders that exist inside the directory
    folders = os.listdir(path_to_data)

    # iterating over folders
    for folder in folders:

        # We concatenate the name of the folder with the path to get the full path of an img
        full_path = os.path.join(path_to_data,folder)

        # getting a list of all the paths to the images inside that folder
        # glob() allows us to look inside a folder and load all the files existing inside that folder
        # so glob() gives us a list of all the files having extension as .png
        images_paths = glob.glob(os.path.join(full_path,"*.png"))
        # now we have a list of all the images inside that folder

        # splitting training and validation set, test_size = 0.1
        x_train,x_val = train_test_split(images_paths, test_size=split_size)

        for x in x_train:

            # getting the full name of a particular image in training dataset
            # basename = os.path.basename(x)

            # we are creating a new folder inside our directory
            # in that folder we will be saving our training and cross-validation dataset images
            # just like our original training dataset we had 43 folders (each representing diff class)
            # we will create such folders from our training dataset for training and cross-validation purpose
            path_to_folder = os.path.join(path_to_save_train,folder)

            # if given folder doesn't exist we create a new one
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            # copy the given image and put it inside the folder
            shutil.copy(x,path_to_folder)

        # repeat the above steps for the cross-validation dataset in the exact same manner
        for x in x_val:

            path_to_folder = os.path.join(path_to_save_val, folder)

            # if given folder doesn't exist we create a new one
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            # copy the given image and put it inside the folder
            shutil.copy(x, path_to_folder)

def order_test_set(path_to_images,path_to_csv):

    try:
        # we are reading our CSV file
        with open(path_to_csv,'r') as csvfile:

            # we are reading our csv file and our delimiter is ","
            # because in our csv file the elements are separated using "," only
            # reader will contain all the data in row-wise manner
            reader = csv.reader(csvfile,delimiter=',')

            for i,row in enumerate(reader):

                # we need to skip the contents of the first row
                if i==0:
                    continue

                else:

                    # we get the img name in the last element of each row.
                    # replace 'Test/' with ''
                    img_name = row[-1].replace('Test/','')

                    #  image label is the second last entry in each row
                    label = row[-2]

                    path_to_folder = os.path.join(path_to_images,label)

                    # check if this path exists or not. If not we create it
                    if not os.path.isdir(path_to_folder):
                        os.makedirs(path_to_folder)

                    img_full_path = os.path.join(path_to_images,img_name)
                    shutil.move(img_full_path,path_to_folder)

    except:
        print("Error Reading CSV file")

# This function will take images from our data sets folder (i.e. training , cross-validation and testing)
# This function will then perform data augmentation and later assign correct labels to each image
def create_generators(batch_size, train_data_path, val_data_path, test_data_path):

    # PERFORMING DATA AUGMENTATION
    # we are preprocessing our training images before they are fed to the CNN
    # rescale the images by a factor of 255
    # rotate the image by 10 degrees i.e bw [-10,10] degrees
    # adjust the width range by 0.1 i.e bw [10% to left,10% to right]
    train_preprocessor = ImageDataGenerator(
        rescale = 1 / 255.,
        rotation_range=10,
        width_shift_range=0.1
    )

    # rescale the test images by a factor of 255
    test_preprocessor = ImageDataGenerator(
        rescale = 1 / 255.,
    )

    # ASSIGNING CORRECT LABELS TO EACH IMAGE IN THE TRAINING AND VALIDATION DATASET
    # flow_from_directory() takes a path from the folder (here train inside the training_data)
    # and then look at all the folders that exist within that directory. It then supposes that each folder contains
    # images that belong to one class
    train_generator = train_preprocessor.flow_from_directory(
        train_data_path,
        class_mode="categorical", # if we use categorical class_mode we will have to use categorical class entropy as loss function
        target_size=(60,60), # resizing all the images to 60*60
        color_mode='rgb', # color_mode = rgb, we are using 3 channels for coloured images
        shuffle=True,
        # images will be shuffled bw successive epochs i.e. the order of the images will not be the same
        # randomness bw images allows our model to learn better
        batch_size=batch_size
    )
    # flow_from_directory() takes a path from the folder (here val inside the training_data)
    # and then look at all the folders that exist within that directory. It then supposes that each folder contains
    # images that belong to one class
    val_generator = test_preprocessor.flow_from_directory(
        val_data_path,
        class_mode="categorical",
        target_size=(60,60),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size,
    )

    # similarly for the test set
    test_generator = test_preprocessor.flow_from_directory(
        test_data_path,
        class_mode="categorical",
        target_size=(60,60),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size,
    )

    return train_generator, val_generator, test_generator

