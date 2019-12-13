"""
Authors: Griffin Noe '21,
         Utkrist P. Thapa '21
This program is composed of all the parts
"""

from keras.models import Sequential
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.layers import LeakyReLU, BatchNormalization
from keras.layers import Conv2DTranspose, Reshape, Activation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from scipy.misc import imresize
import numpy as np
from scipy.io import loadmat
from scipy.misc import imresize
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import keras
import keras.backend as K
from keras.initializers import RandomNormal
from keras.layers import Dense, Activation, LeakyReLU, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_image(filename, image_size=(64,64)):
    image = plt.imread(filename)
    r, c = image.shape[:2]
    cr, cc = 150,150
    sr, sc = (r-cr)//2, (c-cc)//2
    er, ec = r-sr, c-sr
    image = image[sr:er,sc:ec,:]
    image = imresize(image,image_size)
    return image

#~~~~~~~~~~~~~~~~~~~~~~~~~~Models~~~~~~~~~~~~~~~~~~~~~~~~

def create_discriminator(shape,channels):
    strides = 2
    kernel_size = 3
    padding = 'same'
    momentum = 0.9
    alpha = 0.2
    filters = 64

    d = Sequential()
    d.name="Discriminator"

    d.add(Conv2D(filters,
                 kernel_size = kernel_size,
                 strides = strides,
                 padding = padding))

    d.add(LeakyReLU(alpha=alpha))

    d.add(Conv2D(filters*2,
                 kernel_size = kernel_size,
                 strides = strides,
                 padding = padding))

    d.add(BatchNormalization(momentum=momentum))

    d.add(LeakyReLU(alpha=alpha))

    d.add(Conv2D(filters*4,
                 kernel_size = kernel_size,
                 strides = strides,
                 padding = padding))

    d.add(BatchNormalization(momentum=momentum))
    
    d.add(LeakyReLU(alpha=alpha))    

    d.add(Conv2D(filters*8,
                 kernel_size = kernel_size,
                 strides = strides,
                 padding = padding))

    d.add(BatchNormalization(momentum=momentum))

    d.add(LeakyReLU(alpha=alpha))

    d.add(Conv2D(filters*16,
                 kernel_size = kernel_size,
                 strides = strides,
                 padding = padding))

    d.add(BatchNormalization(momentum=momentum))

    d.add(LeakyReLU(alpha=alpha))

    d.add(Flatten())
    
    d.add(Dense(1,activation='sigmoid'))

    d.build((None,shape,shape,channels))

    leaky_alpha = 0.2
    init_stddev = 0.02
    return Sequential([        
        Conv2D(64, kernel_size=5, strides=2, padding='same', 
               kernel_initializer=RandomNormal(stddev=init_stddev),    # 16x16
               input_shape=(32, 32, 3)),
        LeakyReLU(alpha=leaky_alpha),
        Conv2D(128, kernel_size=5, strides=2, padding='same', 
               kernel_initializer=RandomNormal(stddev=init_stddev)),   # 8x8
        BatchNormalization(),
        LeakyReLU(alpha=leaky_alpha),
        Conv2D(256, kernel_size=5, strides=2, padding='same', 
               kernel_initializer=RandomNormal(stddev=init_stddev)),   # 4x4
        BatchNormalization(),
        LeakyReLU(alpha=leaky_alpha),
        Flatten(),
        Dense(1, kernel_initializer=RandomNormal(stddev=init_stddev)),
        Activation('sigmoid')        
    ])

dis = create_discriminator(64,3)
print(dis.summary())

def create_generator(shape,channels,dims):
    filters = 64
    kernel_size = (5,5)
    strides = (2,2)
    rf = 16
    padding = 'same'
    momentum = .9

    g = Sequential()
    g.name="Generator"

    g.add(Dense(rf*filters*shape // rf*shape // rf,
                activation = 'relu'))

    g.add(Reshape((shape//rf,shape//rf,filters*rf)))

    g.add(BatchNormalization())

    g.add(Activation('relu'))

    g.add(Conv2DTranspose(filters*16,
                          kernel_size = kernel_size,
                          strides = strides,
                          padding = padding))

    g.add(BatchNormalization(momentum=momentum))

    g.add(Activation('relu'))

    g.add(Conv2DTranspose(filters*8,
                          kernel_size = kernel_size,
                          strides = strides,
                          padding = padding))

    g.add(BatchNormalization(momentum=momentum))

    g.add(Activation('relu'))

    g.add(Conv2DTranspose(filters*4,
                          kernel_size = kernel_size,
                          strides = strides,
                          padding = padding))

    g.add(BatchNormalization(momentum=momentum))

    g.add(Activation('relu'))

    g.add(Conv2DTranspose(filters*2,
                          kernel_size = kernel_size,
                          strides = strides,
                          padding = padding))

    g.add(BatchNormalization(momentum=momentum))

    g.add(Activation('relu'))

    g.add(Conv2DTranspose(filters,
                          kernel_size = kernel_size,
                          strides = strides,
                          padding = padding))

    g.add(BatchNormalization(momentum=momentum))

    g.add(Activation('relu'))

    g.add(Conv2DTranspose(filters=channels,
                          kernel_size = (3,3),
                          strides = (1,1)))

    g.add(Activation('tanh'))

    g.build((None,dims))

    input_size = 100
    leaky_alpha = 0.2
    init_stddev = 0.02
    return Sequential([
        Dense(4*4*512, input_shape=(input_size,), 
              kernel_initializer=RandomNormal(stddev=init_stddev)),
        Reshape(target_shape=(4, 4, 512)),
        BatchNormalization(),
        LeakyReLU(alpha=leaky_alpha),
        Conv2DTranspose(512, kernel_size=5, strides=2, padding='same',
                        kernel_initializer=RandomNormal(stddev=init_stddev)),
        BatchNormalization(),
        LeakyReLU(alpha=leaky_alpha),
        Conv2DTranspose(256, kernel_size=5, strides=2, padding='same', 
                        kernel_initializer=RandomNormal(stddev=init_stddev)), # 8x8
        BatchNormalization(),
        LeakyReLU(alpha=leaky_alpha),
        Conv2DTranspose(128, kernel_size=5, strides=2, padding='same', 
                        kernel_initializer=RandomNormal(stddev=init_stddev)), # 16x16
        BatchNormalization(),
        LeakyReLU(alpha=leaky_alpha),
        Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', 
                        kernel_initializer=RandomNormal(stddev=init_stddev)), # 32x32
        Activation('tanh')
    ])

gen = create_generator(64,3,100)
print(gen.summary())

def create_gan(shape, channels, dims,
               gen_learning_rate, gen_beta,
               dis_learning_rate, dis_beta):
    
    g = create_generator(shape, channels, dims)
    d = create_discriminator(shape, channels)
    d.compile(optimizer=Adam(lr=dis_learning_rate,
                             beta_1=dis_beta),
              loss='binary_crossentropy')

    gan = Sequential([g,d])
    gan.compile(optimizer=Adam(lr=gen_learning_rate,
                               beta_1=gen_beta),
                loss='binary_crossentropy')

    return gan, g, d

#~~~~~~~~~~~~~~~~~~~~~~~~~~~Training~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_latent_samples(n_samples, sample_size):
    return np.random.normal(loc=0, scale=1, size=(n_samples, sample_size))

def make_trainable(model, trainable):
    for layer in model.layers:
        layer.trainable = trainable
        
def make_labels(size):
    return np.ones([size, 1]), np.zeros([size, 1])

def show_losses(losses):
    losses = np.array(losses)
    
    fig, ax = plt.subplots()
    plt.plot(losses.T[0], label='Discriminator')
    plt.plot(losses.T[1], label='Generator')
    plt.title("Validation Losses")
    plt.legend()
    plt.show()
    
def show_images(generated_images):
    n_images = len(generated_images)
    cols = 10
    rows = n_images//cols
    
    plt.figure(figsize=(10, 8))
    for i in range(n_images):
        img = deprocess(generated_images[i])
        ax = plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()


def pp(x):
    return (x/255)*2-1

def dp(x):
    return np.uint8((x+1)/2*255)

def train(epochs, batch_size, eval_size,
          shape, channels, dims,
          gen_learning_rate, gen_beta,
          dis_learning_rate, dis_beta,
          smooth, x_train, x_test):

        gan, g, d = create_gan(shape, channels, dims,
                               gen_learning_rate, gen_beta,
                               dis_learning_rate, dis_beta)

        l = []

        ytr, ytf = make_labels(batch_size)
        yer, yef = make_labels(eval_size)

        for e in range(epochs):
            for i in tqdm(range(len(x_train)//batch_size)):
                xb = x_train[i*batch_size:(i+1)*batch_size]
                xbr= np.array([pp(load_image(filename)) for filename in xb])

                ls = make_latent_samples(batch_size, dims)
                xbf= g.predict_on_batch(ls)
                print(batch_size)
                print(dims)
                print(xbf.shape)

                for l in d.layers:
                    l.trainable = True

                d.train_on_batch(xbr, ytr * (1-smooth))
                d.train_on_batch(xbf, ytf)

                for l in d.layers:
                    l.trainable = False

                gan.train_on_batch(ls, ytr)
                
            xe = x_test[np.random.choice(len(x_test), eval_size, replace=False)]
            xer= np.array([pp(load_image(filename)) for filename in xe])

            ls = make_latent_samples(eval_size, sample_size)
            xef= g.predict_on_batch(ls)

            d_loss = d.test_on_batch(xer, yer)
            d_loss+= d.test_on_batch(xef, yef)
            g_loss = gan.test_on_batch(ls,yer)

            l.append((d_loss,g_loss))

            print("Epoch: {:>3}/{} Discriminator Loss: {:>6.4f} Generator Loss: {:>6.4f}".format(
                    e+1, epochs, d_loss, g_loss))    
            show_images(xef[:10])

        show_losses(losses)
        show_images(g.predict(make_latent_samples(80, sample_size)))

        return g

filenames = np.array(glob('img_align_celeba/*.jpg'))

x_train, x_test = train_test_split(filenames, test_size=1000)

train(epochs = 3,
      batch_size = 128,
      eval_size = 16,
      shape = 64,
      channels = 3,
      dims = 100,
      gen_learning_rate = 0.0001,
      gen_beta = 0.5,
      dis_learning_rate = 0.001,
      dis_beta = 0.5,
      smooth = 0.1,
      x_train=x_train,
      x_test=x_test)
