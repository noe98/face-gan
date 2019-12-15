import numpy as np
from glob import glob
from scipy.misc import imresize
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, BatchNormalization, Conv2D
from keras.layers import Conv2DTranspose, Dense, Flatten, LeakyReLU, Reshape
from keras.optimizers import Adam
from keras.initializers import RandomNormal
import os

filenames = np.array(glob('img_align_celeba/*.jpg'))

x_train, x_test = train_test_split(filenames, test_size=1000)

def create_generator(filters, input_size, alpha, stdev, kernel_size, strides, padding, momentum):
    g = Sequential()
    
    g.add(Dense(4*4*filters,
                input_shape=(input_size,), 
                kernel_initializer=RandomNormal(stddev=stdev)))

    g.add(Reshape(target_shape=(4,4,filters)))

    g.add(BatchNormalization())

    g.add(Conv2DTranspose(filters=filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding=padding, 
                          kernel_initializer=RandomNormal(stddev=stdev)))

    g.add(LeakyReLU(alpha=alpha))

    g.add(BatchNormalization(momentum=momentum))

    g.add(Conv2DTranspose(filters=filters//2,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding=padding, 
                          kernel_initializer=RandomNormal(stddev=stdev)))

    g.add(LeakyReLU(alpha=alpha))

    g.add(BatchNormalization(momentum=momentum))

    g.add(LeakyReLU(alpha=alpha))

    g.add(Conv2DTranspose(filters=filters//4,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding=padding, 
                          kernel_initializer=RandomNormal(stddev=stdev)))

    g.add(BatchNormalization(momentum=momentum))

    g.add(LeakyReLU(alpha=alpha))

    g.add(Conv2DTranspose(filters=filters//8,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding=padding, 
                          kernel_initializer=RandomNormal(stddev=stdev)))

    g.add(BatchNormalization(momentum=momentum))

    g.add(LeakyReLU(alpha=alpha))

    g.add(Conv2DTranspose(filters=3,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding=padding, 
                          kernel_initializer=RandomNormal(stddev=stdev)))

    g.add(Activation('tanh'))

    return g


def create_discriminator(filters, alpha, stdev, input_shape, kernel_size, strides, padding, momentum):
    d = Sequential()

    d.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 strides=strides,
                 padding=padding, 
                 kernel_initializer=RandomNormal(stddev=stdev),
                 input_shape=(input_shape)))

    d.add(LeakyReLU(alpha=alpha))

    d.add(Conv2D(filters=filters*2,
                 kernel_size=kernel_size,
                 strides=strides,
                 padding=padding, 
                 kernel_initializer=RandomNormal(stddev=stdev)))

    d.add(BatchNormalization(momentum=momentum))

    d.add(LeakyReLU(alpha=alpha))

    d.add(Conv2D(filters=filters*4,
                 kernel_size=kernel_size,
                 strides=strides,
                 padding=padding, 
                 kernel_initializer=RandomNormal(stddev=stdev)))

    d.add(BatchNormalization(momentum=momentum))

    d.add(LeakyReLU(alpha=alpha))

    d.add(Conv2D(filters=filters*8,
                 kernel_size=kernel_size,
                 strides=strides,
                 padding=padding,
                 kernel_initializer=RandomNormal(stddev=stdev)))

    d.add(Flatten())

    d.add(Dense(1, kernel_initializer=RandomNormal(stddev=stdev)))

    d.add(Activation('sigmoid'))

    return d 

def create_DCGAN(latent_dims, 
               g_learning_rate, 
               g_beta,
               d_learning_rate,
               d_beta,
               alpha,
               init_std,
               generated_image_size,
               g_kernel_size, g_strides, g_padding,
               d_kernel_size, d_strides, d_padding,
               g_filters, d_filters, momentum):
    
    g = create_generator(g_filters, latent_dims, alpha, init_std,
                         g_kernel_size, g_strides, g_padding, momentum)

    d = create_discriminator(d_filters, alpha, init_std, generated_image_size,
                             d_kernel_size, d_strides, d_padding, momentum)
    
    d.compile(optimizer=Adam(lr=d_learning_rate, beta_1=d_beta),
                          loss='binary_crossentropy')
    
    gan = Sequential([g, d])
    
    gan.compile(optimizer=Adam(lr=g_learning_rate, beta_1=g_beta),
                loss='binary_crossentropy')
    
    return gan, g, d

def load_image(filename, image_size):
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

def show_losses(losses, filename):
    losses = np.array(losses)
    
    fig, ax = plt.subplots()
    plt.plot(losses.T[0], label='Discriminator')
    plt.plot(losses.T[1], label='Generator')
    plt.title("Validation Losses")
    plt.legend()
    plt.savefig(filename)
    #plt.show()
    
def show_images(generated_images, filename):
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

    plt.savefig(filename)
    #plt.show()

def train(
    g_learning_rate, 
    g_beta,        
    d_learning_rate, 
    d_beta,        
    alpha,
    init_std,
    smooth,
    latent_dims, 
    epochs,
    batch_size,  
    eval_size,    
    images,
    generated_image_size,
    g_kernel_size,
    g_strides,
    g_padding,
    d_kernel_size,
    d_strides,
    d_padding,
    g_filters,
    d_filters,
    momentum):

    # labels for the batch size and the test size
    ytr, ytf = np.ones([batch_size,1]), np.zeros([batch_size,1])
    yer,  yef  = np.ones([eval_size,1]), np.zeros([eval_size,1])

    # create a GAN, a generator and a discriminator
    gan, g, d = create_DCGAN(
        latent_dims, 
        g_learning_rate, 
        g_beta,
        d_learning_rate,
        d_beta,
        alpha,
        init_std,
        generated_image_size,
        g_kernel_size, g_strides, g_padding,
        d_kernel_size, d_strides, d_padding,
        g_filters, d_filters, momentum)

    loss = []
    for e in range(epochs):
        for i in tqdm(range(len(x_train)//batch_size)):
            xb = x_train[i*batch_size:(i+1)*batch_size]
            xbr = np.array([preprocess(load_image(fn,generated_image_size)) for fn in xb])

            ls = np.random.normal(loc=0,scale=1,size=(batch_size,latent_dims))
            xbf = g.predict_on_batch(ls)

            for l in d.layers:
                l.trainable = True
                
            d.train_on_batch(xbr, ytr * (1 - smooth))
            d.train_on_batch(xbf, ytf)

            for l in d.layers:
                l.trainable = False
                
            gan.train_on_batch(ls, ytr)

            if(i%500==0):
                gen_file_name="generator_" + str(e) + "_" + str(i) + ".json"
                dis_file_name="discriminator_"+str(e)+"_"+str(i)+".json"
                gan_file_name="gan_"+str(e)+"_"+str(i)+".json"
                d_json = d.to_json()
                g_json = g.to_json()
                gan_json = gan.to_json()
                with open(gen_file_name,"w") as jf:
                    jf.write(g_json)
                with open(gan_file_name,"w") as jf:
                    jf.write(gan_json)
                with open(dis_file_name,"w") as jf:
                    jf.write(d_json)

        # evaluate
        xe = x_test[np.random.choice(len(x_test), eval_size, replace=False)]
        xer = np.array([preprocess(load_image(filename, generated_image_size[:2])) for filename in xe])

        ls = np.random.normal(loc=0,scale=1,size=(eval_size,latent_dims))
        xef = g.predict_on_batch(ls)

        d_loss  = d.test_on_batch(xer, yer)
        d_loss += d.test_on_batch(xef, yef)
        g_loss  = gan.test_on_batch(ls, yer) 

        loss.append((d_loss, g_loss))

        gan_json = gan.to_json()
        d_json = d.to_json()
        g_json = g.to_json()

        gan_filename = "gan_" + str(e)+ "_final.json"
        with open(gan_filename, "w") as jf:
            jf.write(gan_json)

        dis_filename="dis_"+str(e)+"_final.json"
        with open(dis_filename, "w") as jf:
            jf.write(d_json)

        gen_filename = "generator_" + str(e) + "_final.json"
        with open(gen_filename, "w") as jf:
            jf.write(g_json)

        print("Epoch: {:>3}/{} Discriminator Loss: {:>6.4f} Generator Loss: {:>6.4f}".format(e+1, epochs, d_loss, g_loss))

        if images:
            plot_name= "epoch_" + str(e) + ".png"
            show_images(xef[:10],plot_name)
    
    if images:
        plot_name = "final.png"
        loss_name = "loss.png"
        show_losses(loss, loss_name)
        show_images(g.predict(np.random.normal(loc=0,scale=1,size=(80, latent_dims))),plot_name)
        
    return gan, d, g


gan, d, g = train(g_learning_rate=0.0001,               #Generator learning rate
                   g_beta=0.5,                          #Generator decay rate for ADAM
                   d_learning_rate=0.0005,              #Discriminator learning rate
                   d_beta=0.5,                          #Discriminator decay rate for ADAM
                   alpha=0.2,                           #Alpha value for leaky ReLU
                   init_std=0.02,                       #Initial standard deviation for Kernel Initializer
                   smooth=0.1,                          #Label smoothing variable
                   latent_dims=100,                     #Dimension of the latent space
                   epochs=3,                            #Number of training iterations
                   batch_size=64,                       #Batch size for training
                   eval_size=16,                        #Size for evaluation
                   images=True,                         #if true, images are displayed between epochs
                   generated_image_size=(128,128,3),    #Dimensions of generated images
                   g_kernel_size=3,                     #Generator's kernel size
                   g_strides=2,                         #Generator's strides
                   g_padding='same',                    #Generator's padding
                   d_kernel_size=5,                     #Discriminator's kernel size 
                   d_strides=2,                         #Discriminator's strides
                   d_padding='same',                    #Discriminator's padding
                   g_filters=512,                       #Generator's first layer filters
                   d_filters=64,                        #Discriminator's first layer filters
                   momentum=0.9)                        #Momentum for batch normalization

gan_json = gan.to_json()
d_json = d.to_json()
g_json = g.to_json()

with open("gan.json", "w") as jf:
    jf.write(gan_json)

with open("discriminator.json", "w") as jf:
    jf.write(d_json)

with open("generator_final.json", "w") as jf:
    jf.write(g_json)


