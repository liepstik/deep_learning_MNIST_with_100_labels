
import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import mnist
from keras.layers import (Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import Adam

class Dataset:
    def __init__(self):
        self.num_labeled = 100

        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

    def training_set(self):
        x_train = self.x_train[range(self.num_labeled)]
        y_train = self.y_train[range(self.num_labeled)]
        return x_train, y_train


    def test_set(self):
        return self.x_test, self.y_test

class DCGAN:
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.z_dim = 100
        

        

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256 * 7 * 7, input_dim=self.z_dim))
        model.add(Reshape((7, 7, 256)))
        model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))
        model.add(Activation('tanh'))
        return model

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(32,kernel_size=3, strides=2,input_shape=self.img_shape,padding='same'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2D(64,kernel_size=3,strides=2, input_shape=self.img_shape,padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2D(128,kernel_size=3,strides=2,input_shape=self.img_shape,padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def build_gan(self,generator,discriminator):
        model = Sequential()
        model.add(generator)
        model.add(discriminator)
        return model


    def train(self,iterations, batch_size, sample_interval):
        train_hist={}
        train_hist['D_losses']=[]
        train_hist['G_losses']=[]
        def save_model(step):
            f1= 'models/dcgan_discriminator_weight_%04d.h5' % (step+1)
            f2= 'models/dcgan_generator_weight_%04d.h5' % (step+1)
            discriminator.save(f1)
            generator.save(f2)
            #print("save model")

        losses = []
        accuracies = []
        iteration_checkpoints = []
        d_losses_real=[]
        d_losses_fake=[]
    
        (X_train, _), (_,_) = mnist.load_data()

        def sample_images(epoch,image_grid_rows=4, image_grid_columns=4):

            z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, self.z_dim))
            gen_imgs = generator.predict(z)
            gen_imgs = 0.5 * gen_imgs + 0.5
            fig, axs = plt.subplots(image_grid_rows,
                                    image_grid_columns,
                                    figsize=(10, 4),
                                    sharey=True,
                                    sharex=True)

            cnt = 0
            for i in range(image_grid_rows):
                for j in range(image_grid_columns):
                    axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                    axs[i, j].axis('off')
                    cnt += 1
            plt.tight_layout()
            plt.show()
            #fig.savefig("images/generator/%d.png" % epoch)

        
        discriminator = self.build_discriminator()
        discriminator.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])

        generator = self.build_generator()
        discriminator.trainable = False

        gan = self.build_gan(generator, discriminator)
        gan.compile(loss='binary_crossentropy', optimizer=Adam())
        

        X_train = X_train / 127.5 - 1.0
        X_train = np.expand_dims(X_train, axis=3)

        real = np.ones((batch_size, 1))

        fake = np.zeros((batch_size, 1))

        for iteration in range(iterations):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            z = np.random.normal(0, 1, (batch_size, 100))
            gen_imgs = generator.predict(z)

            d_loss_real = discriminator.train_on_batch(imgs, real)
            d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
            d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)


            z = np.random.normal(0, 1, (batch_size, 100))
            gen_imgs = generator.predict(z)

            g_loss = gan.train_on_batch(z, real)

            if (iteration + 1) % sample_interval == 0:
                train_hist['D_losses'].append(d_loss)
                train_hist['G_losses'].append(g_loss)

                #losses.append((d_loss, g_loss))
                d_losses_real.append(d_loss_real)
                d_losses_fake.append(d_loss_fake)
                accuracies.append(100.0 * accuracy)
                iteration_checkpoints.append(iteration + 1)

                print("%d [D loss: %f] [G loss: %f]" % (iteration + 1, d_loss, g_loss))

                sample_images(iteration)
                save_model(iteration)
        return train_hist, accuracies, iteration_checkpoints, d_losses_real,d_losses_fake