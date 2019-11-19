"""
Authors: Griffin Noe '21,
         Utkrist P. Thapa '21
This program manipulates the data set in order to get it ready for our model
"""
import matplotlib.pyplot as plt
import os, time
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

filepath = "data/img_align_celeba/"

# getting the image ids 
img_ids = np.sort(os.listdir(filepath))

# training data
train_img_ids = img_ids[: 200000]
# test data
test_img_ids = img_ids[200000 : 200000 + 2000]

# resize shape
img_shape = (64, 64, 3)

# get the image data
train_img = []
counter = 0
for img_id in train_img_ids:
    counter += 1
    image = load_img(filepath + "/" + img_id)
    image = img_to_array(image) / 255  # converting image to array and normalize
    train_img.append(image)
train_img = np.array(train_img)   # converting into np array type
