import tensorflow as tf
import numpy as np

def predict_with_model(model, imgpath):

    # load image using tensorflow utility function
    image = tf.io.read_file(imgpath)

    # decoding the image, for RGB channel=3
    image = tf.image.decode_png(image, channels=3)

    # changing datatype of img pixels from unsigned to float32
    # this will also scale them bw 0 and 1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # resizing my image to target size
    # in the generators we have target size = 60*60 hence all of our images were resized to 60*60
    # hence we need to maintain that size here too
    image = tf.image.resize(image, [60,60]) # (60,60,3)

    # introducing a new dimension
    # the input layer of CNN accepts images of shape (None,60,60,3)
    # hence we were required to add a dimension in extreme LHS
    image = tf.expand_dims(image, axis=0) # (1,60,60,3)

    # contains the probabilities as to the different classes to which our image belongs
    predictions = model.predict(image) # [0.005, 0.00003, 0.99, 0.00 ....]

    # test, val and train generators it sorts the directories as strings instead of integers, so we create a dictionary
    prediction_dict = {
        0: 0,
        1: 1,
        2: 10,
        3: 11,
        4: 12,
        5: 13,
        6: 14,
        7: 15,
        8: 16,
        9: 17,
        10: 18,
        11: 19,
        12: 2,
        13: 20,
        14: 21,
        15: 22,
        16: 23,
        17: 24,
        18: 25,
        19: 26,
        20: 27,
        21: 28,
        22: 29,
        23: 3,
        24: 30,
        25: 31,
        26: 32,
        27: 33,
        28: 34,
        29: 35,
        30: 36,
        31: 37,
        32: 38,
        33: 39,
        34: 4,
        35: 40,
        36: 41,
        37: 42,
        38: 5,
        39: 6,
        40: 7,
        41: 8,
        42: 9,
    }
    # Take the maximum value from the predictions list. Using np.argmax(), we get the index of maximum value in the list
    # And hence get the class number to which the image belongs
    predictions = prediction_dict[np.argmax(predictions)]

    return predictions

if __name__=="__main__":

    img_path_1 = "C:\\Users\\GARG\\PycharmProjects\\Python Codes\\CNN Project\\GSTRB Dataset\\Test\\12\\01003.png"
    img_path_2 = "C:\\Users\\GARG\\PycharmProjects\\Python Codes\\CNN Project\\GSTRB Dataset\\Test\22\\11950.png"

    model = tf.keras.models.load_model('./Best_Model')
    prediction1 = predict_with_model(model, img_path_1)
    prediction2 = predict_with_model(model, img_path_2)

    print(f"prediction 1= {prediction1}")
    print(f"prediction 2= {prediction2}")