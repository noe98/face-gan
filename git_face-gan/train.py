"""
Authors: Griffin Noe '21,
         Utkrist P. Thapa '21
This program trains the generator and discriminator
"""
from tqdm import tqdm

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
          x_train, x_test, smooth):

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

                for l in d.layers:
                    l.trainable = True

                d.train_on_batch(xbr, ytr * (1-smooth))
                d.train_on_batch(xbf, trf)

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
