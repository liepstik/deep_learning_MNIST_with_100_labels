import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import mnist
from keras.layers import (Activation, BatchNormalization, Concatenate, Dense,Embedding, Flatten, Input, Multiply, Reshape)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras.optimizers import Adam


class Dataset:
    def __init__(self):
        
        # nombre de labels pour entrainer
        self.num_labeled = 100

        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

    def training_set(self):
        x_train = self.x_train[range(self.num_labeled)]
        y_train = self.y_train[range(self.num_labeled)]
        return x_train, y_train

    ## Les données test 10000 
    def test_set(self):
        return self.x_test, self.y_test

class CGAN:
    def __init__(self):

        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.z_dim = 100
        self.num_classes = 10

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


    def build_cgan_generator(self):

        z = Input(shape=(self.z_dim, ))
        label = Input(shape=(1, ), dtype='int32')
        label_embedding = Embedding(self.num_classes, self.z_dim, input_length=1)(label)
        label_embedding = Flatten()(label_embedding)
        joined_representation = Multiply()([z, label_embedding])
        generator = self.build_generator()
        conditioned_img = generator(joined_representation)
        return Model([z, label], conditioned_img)

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(64,kernel_size=3, strides=2,
                input_shape=(self.img_shape[0], self.img_shape[1], self.img_shape[2] + 1),
                padding='same'))

        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))

        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(
            Conv2D(128,
                kernel_size=3,
                strides=2,
                input_shape=self.img_shape,
                padding='same'))

        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        return model


    def build_cgan_discriminator(self):
        img = Input(shape=self.img_shape)
        label = Input(shape=(1, ), dtype='int32')
        label_embedding = Embedding(self.num_classes, np.prod(self.img_shape), input_length=1)(label)

        label_embedding = Flatten()(label_embedding)
        label_embedding = Reshape(self.img_shape)(label_embedding)
        concatenated = Concatenate(axis=-1)([img, label_embedding])
        discriminator = self.build_discriminator()
        classification = discriminator(concatenated)

        return Model([img, label], classification)

    def build_cgan(self,generator,discriminator):

        z = Input(shape=(self.z_dim, ))
        label = Input(shape=(1, ))
        img = generator([z, label])
        classification = discriminator([img, label])
        model = Model([z, label], classification)
        return model


    def train(self,iterations, batch_size, sample_interval):
        train_hist={}
        train_hist['D_losses']=[]
        train_hist['G_losses']=[]
        accuracies = []
        losses = []
        dataset = Dataset()
        (X_train, y_train)= dataset.training_set()

        def save_model(step):
            f1= 'models/cgan_discriminator_weight_%04d.h5' % (step+1)
            f2= 'models/cgan_generator_weight_%04d.h5' % (step+1)
            f3= 'models/cgan_weight_%04d.h5' % (step+1)
            discriminator.save(f1)
            generator.save(f2)
            cgan.save(f3)
            #print("save model")

        def sample_images(epoch,image_grid_rows=2, image_grid_columns=5):

            z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, self.z_dim))

            labels = np.arange(0, 10).reshape(-1, 1)

            gen_imgs = generator.predict([z, labels])

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
                    axs[i, j].set_title("Digit: %d" % labels[cnt])
                    cnt += 1
            plt.tight_layout()
            plt.show()
            fig.savefig("images/%d.png" % epoch)




        discriminator = self.build_cgan_discriminator()
        discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        generator = self.build_cgan_generator()

        discriminator.trainable = False

        cgan = self.build_cgan(generator, discriminator)
        cgan.compile(loss='binary_crossentropy', optimizer=Adam())
    

        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for iteration in range(iterations):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            z = np.random.normal(0, 1, (batch_size, self.z_dim))
            gen_imgs = generator.predict([z, labels])

            d_loss_real = discriminator.train_on_batch([imgs, labels], real)
            d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            z = np.random.normal(0, 1, (batch_size, self.z_dim))

            labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)

            g_loss = cgan.train_on_batch([z, labels], real)

            if (iteration + 1) % sample_interval == 0:

                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                    (iteration + 1, d_loss[0], 100 * d_loss[1], g_loss))

                losses.append((d_loss[0], g_loss))
                accuracies.append(100 * d_loss[1])
                train_hist['D_losses'].append(d_loss)
                train_hist['G_losses'].append(g_loss)

                sample_images(iteration)
                save_model(iteration)
        return train_hist, accuracies






    
