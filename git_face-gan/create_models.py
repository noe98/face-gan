"""
Authors: Griffin Noe '21,
         Utkrist P. Thapa '21
This program generates the generator and discriminator
"""
from keras.models import Sequential
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.layers import LeakyReLU, BatchNormalization
from keras.layers import Conv2DTranspose, Reshape, Activation


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
    
    return d

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

    return g

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
    
