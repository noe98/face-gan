# face_gan

Authors: Griffin Noe and Utkrist P. Thapa

Washington and Lee University

This is a Python implementation of a GAN (Generative Adversarial Network) that generates faces. We use CelebA dataset found here: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html  

# Preparing the Data 
First, we import the dataset into the python file. We extract the image ids first and then we iterate through all the image ids in order to retrieve the image. The shape of each of the images in the original dataset is (218, 178, 3). Since this is very large, it might make it slower to do computations, hence slowing down the training time. Hence, we downsize the images to (64, 64, 3). Then, we attempt to normalize the dataset by dividing each face image pixel value by 255 (the range of pixel values for each image is from 0 to 255). We use 200,000 images for training. All these normalized image data is appended to a training list. 

# Models
As this is a generative adversarial network, we had to construct both a generator and a discriminator. 
