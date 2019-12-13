"""
Authors: Griffin Noe '21,
         Utkrist P. Thapa '21
This program manipulates the data set in order to get it ready for our model
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from scipy.misc import imresize

filepath = "img_align_celeba/"

def load_dataset(file_path, image_size):
    # getting the image ids 
    img_ids = np.sort(os.listdir(file_path))

    # training data
    train_img_ids = img_ids[: 200000]
    # test data
    test_img_ids = img_ids[200000 : 200000 + 2000]

    # get the image data
    train_img = []
    counter = 0
    for img_id in train_img_ids:
        counter += 1
        image = load_img(filepath + "/" + img_id)
        image = img_to_array(image)
        image = ((image-image.min())/(255-image.min()))
        image = image*2 - 1
        r, c = image.shape[:2]
        cr, cc = 150,150
        sr, sc = (r-cr)//2, (c-cc)//2
        er, ec = r-sr, c-sr
        image = image[sr:er,sc:ec,:]
        image = imresize(image,image_size)
        train_img.append(image)

    test_img = []
    counter = 0
    for img_id in test_img_ids:
        counter += 1
        image = load_img(filepath + "/" + img_id)
        image = img_to_array(image)
        image = ((image-image.min())/(255-image.min()))
        image = image*2 - 1
        r, c = image.shape[:2]
        cr, cc = 150,150
        sr, sc = (r-cr)//2, (c-cc)//2
        er, ec = r-sr, c-sr
        image = image[sr:er,sc:ec,:]
        image = imresize(image,image_size)
        test_img.append(image)
        
    train_img = np.array(train_img)   # converting into np array type
    test_img = np.array(test_img)
    return (train_img, test_img)

def load_image(filename, image_size=(64,64):
    image = plt.imread(filename)
    r, c = image.shape[:2]
    cr, cc = 150,150
    sr, sc = (r-cr)//2, (c-cc)//2
    er, ec = r-sr, c-sr
    image = image[sr:er,sc:ec,:]
    image = imresize(image,image_size)
    return image

train, test = load_dataset(filepath, (64,64))
plt.imshow(train[0])
plt.show()
plt.imshow(test[0])
plt.show()
