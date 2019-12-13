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

filenames = np.array(glob('img_align_celeba/*.jpg'))

x_train, x_test = train_test_split(filenames, test_size=1000)

def load_image(filename, image_size=(64,64)):
    image = plt.imread(filename)
    r, c = image.shape[:2]
    cr, cc = 150,150
    sr, sc = (r-cr)//2, (c-cc)//2
    er, ec = r-sr, c-sr
    image = image[sr:er,sc:ec,:]
    image = imresize(image,image_size)
    return image

def preprocess(x):
    return (x/255)*2-1

def deprocess(x):
    return np.uint8((x+1)/2*255)

def create_generator(input_size, alpha, stdev, kernel_size, strides, padding):
    g = Sequential()

    filters = 512
    
    g.add(Dense(4*4*filters,
                input_shape=(input_size,), 
                kernel_initializer=RandomNormal(stddev=stdev)))

    g.add(Reshape(target_shape=(4,4,filters)))

    g.add(BatchNormalization())

    g.add(Conv2DTranspose(filters=filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding='same', 
                          kernel_initializer=RandomNormal(stddev=stdev)))

    g.add(LeakyReLU(alpha=alpha))

    g.add(BatchNormalization())

    g.add(LeakyReLU(alpha=alpha))

    g.add(Conv2DTranspose(filters=filters/2,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding='same', 
                          kernel_initializer=RandomNormal(stddev=stdev)))

    g.add(BatchNormalization())

    g.add(LeakyReLU(alpha=alpha))

    g.add(Conv2DTranspose(filters=filters/4,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding='same', 
                          kernel_initializer=RandomNormal(stddev=stdev)))

    g.add(BatchNormalization())

    g.add(LeakyReLU(alpha=alpha))

    g.add(Conv2DTranspose(filters=3,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding='same', 
                          kernel_initializer=RandomNormal(stddev=stdev)))

    g.add(Activation('tanh'))

    return g


def create_discriminator(alpha, stdev, input_shape, kernel_size, strides, padding):
    d = Sequential()

    filters = 64

    d.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 strides=strides,
                 padding='same', 
                 kernel_initializer=RandomNormal(stddev=stdev),
                 input_shape=(input_shape)))

    d.add(LeakyReLU(alpha=alpha))

    d.add(Conv2D(filters=filters*2,
                 kernel_size=kernel_size,
                 strides=strides,
                 padding='same', 
                 kernel_initializer=RandomNormal(stddev=stdev)))

    d.add(BatchNormalization())

    d.add(LeakyReLU(alpha=alpha))

    d.add(Conv2D(filters=filters*4,
                 kernel_size=kernel_size,
                 strides=strides,
                 padding='same', 
                 kernel_initializer=RandomNormal(stddev=stdev)))

    d.add(BatchNormalization())

    d.add(LeakyReLU(alpha=alpha))

    d.add(Flatten())

    d.add(Dense(1, kernel_initializer=RandomNormal(stddev=stdev)))

    d.add(Activation('sigmoid'))

    return d 

def make_DCGAN(sample_size, 
               g_learning_rate, 
               g_beta,
               d_learning_rate,
               d_beta,
               alpha,
               init_std,
               generated_image_size,
               g_kernel_size, g_strides, g_padding,
               d_kernel_size, d_strides, d_padding):
    
    g = create_generator(sample_size, alpha, init_std,
                         g_kernel_size, g_strides, g_padding)

    d = create_discriminator(alpha, init_std, generated_image_size,
                             d_kernel_size, d_strides, d_padding)
    d.compile(optimizer=Adam(lr=d_learning_rate, beta_1=d_beta),
                          loss='binary_crossentropy')
    
    gan = Sequential([g, d])
    gan.compile(optimizer=Adam(lr=g_learning_rate, beta_1=g_beta),
                loss='binary_crossentropy')
    
    return gan, g, d

def make_latent_samples(n_samples, sample_size):
    return np.random.normal(loc=0, scale=1, size=(n_samples, sample_size))

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

def train(
    g_learning_rate, # learning rate for the generator
    g_beta,        # the exponential decay rate for the 1st moment estimates in Adam optimizer
    d_learning_rate, # learning rate for the discriminator
    d_beta,        # the exponential decay rate for the 1st moment estimates in Adam optimizer
    alpha,
    init_std,
    smooth=0.1,
    sample_size=100, # latent sample size (i.e. 100 random numbers)
    epochs=3,
    batch_size=128,  # train batch size
    eval_size=16,    # evaluate size
    show_images=True,
    generated_image_size=(64,64,3),
    g_kernel_size=5,
    g_strides=2,
    g_padding='same',
    d_kernel_size=5,
    d_strides=2,
    d_padding='same'):

    # labels for the batch size and the test size
    ytr, ytf = np.ones([batch_size,1]), np.zeros([batch_size,1])
    yer,  yef  = np.ones([eval_size,1]), np.zeros([eval_size,1])

    # create a GAN, a generator and a discriminator
    gan, g, d = make_DCGAN(
        sample_size, 
        g_learning_rate, 
        g_beta,
        d_learning_rate,
        d_beta,
        alpha,
        init_std,
        generated_image_size,
        g_kernel_size, g_strides, g_padding,
        d_kernel_size, d_strides, d_padding)

    l = []
    for e in range(epochs):
        for i in tqdm(range(len(x_train)//batch_size)):
            # real CelebA images
            xb = x_train[i*batch_size:(i+1)*batch_size]
            xbr = np.array([preprocess(load_image(fn,generated_image_size)) for fn in xb])

            # latent samples and the generated digit images
            ls = make_latent_samples(batch_size, sample_size)
            xbf = g.predict_on_batch(ls)

            # train the discriminator to detect real and fake images
            for l in d.layers:
                l.trainable = True
                
            d.train_on_batch(xbr, ytr * (1 - smooth))
            d.train_on_batch(xbf, ytf)

            # train the generator via GAN
            for l in d.layers:
                l.trainable = False
                
            gan.train_on_batch(ls, ytr)

        # evaluate
        X_eval = x_test[np.random.choice(len(x_test), eval_size, replace=False)]
        X_eval_real = np.array([preprocess(load_image(filename, generated_image_size[:2])) for filename in X_eval])

        ls = make_latent_samples(eval_size, sample_size)
        X_eval_fake = g.predict_on_batch(ls)

        d_loss  = d.test_on_batch(X_eval_real, yer)
        d_loss += d.test_on_batch(X_eval_fake, yef)
        g_loss  = gan.test_on_batch(ls, yer) # we want the fake to be realistic!

        l.append((d_loss, g_loss))

        print("Epoch: {:>3}/{} Discriminator Loss: {:>6.4f} Generator Loss: {:>6.4f}".format(
            e+1, epochs, d_loss, g_loss))
        if show_images:
            show_images(X_eval_fake[:10])
    
    if show_images:
        show_losses(l)
        show_images(g.predict(make_latent_samples(80, sample_size)))
        
    return gan, d, g

g_learning_rate = 0.0001
g_beta = 0.5
d_learning_rate = 0.001


gan, d, g = train(g_learning_rate=0.0001, 
                   g_beta=0.5, 
                   d_learning_rate=0.001, 
                   d_beta=0.5, 
                   alpha=0.2, 
                   init_std=0.02);

gan_json = gan.to_json()
d_json = d.to_json()
g_json = g.to_json()

with open("gan.json", "w") as jf:
    jf.write(gan_json)

with open("discriminator.json", "w") as jf:
    jf.write(d_json)

with open("generator.json", "w") as jf:
    jf.write(g_json)


