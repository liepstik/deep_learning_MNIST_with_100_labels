#--------------------------------------

# ABIDAR Bouchra && LIEPCHITZ Laura
# MLDS 2019/2020

#--------------------------------------
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
import matplotlib.image as mpimg


class Dataset:
    def __init__(self):
        
        # nombre de labels pour entrainer
        self.num_labeled = 100

        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        # preprocessing pour les images
        def preprocess_imgs(x):
            x = (x.astype(np.float32) - 127.5) / 127.5
            x = np.expand_dims(x, axis=3)
            return x
  
        def preprocess_labels(y):
            return y.reshape(-1, 1)
     
        # Training data
        self.x_train = preprocess_imgs(self.x_train)
        self.y_train = preprocess_labels(self.y_train)
        
        # Testing data
        self.x_test = preprocess_imgs(self.x_test)
        self.y_test = preprocess_labels(self.y_test)

    # Obtenez un random batch d'images étiquetées et leurs étiquettes
    def batch_labeled(self, batch_size):
        idx = np.random.randint(0, self.num_labeled, batch_size)
        imgs = self.x_train[idx]
        labels = self.y_train[idx]
        return imgs, labels
    
    # Obtenez un random batch d'images sans étiquetées
    def batch_unlabeled(self, batch_size):

        idx = np.random.randint(self.num_labeled, self.x_train.shape[0], batch_size)
        imgs = self.x_train[idx]
        return imgs
    # Fonction pour renvoyer les données training (seulement 100 labels)
    def training_set(self):
        x_train = self.x_train[range(self.num_labeled)]
        y_train = self.y_train[range(self.num_labeled)]
        return x_train, y_train
    ## Les données test 10000 
    def test_set(self):
        return self.x_test, self.y_test


class SGAN:
    def __init__(self,n_epochs,batch_size):
        self.rows = 28
        self.cols = 28
        self.channels = 1
        self.img_shape = (self.rows, self.cols, self.channels)
        self.num_classes = 10 
        self.z_dim = 100     # la taille du vecteur de bruit z 
        self.n_epochs=n_epochs 
        self.batch_size = batch_size
        self.generator = self.build_generator()


    def build_generator(self):
        model = Sequential()
        # Reshape l'input 7x7x256 avec une couche connecté
        model.add(Dense(256 * 7 * 7, input_dim=self.z_dim))
        model.add(Reshape((7, 7, 256)))
    
        # Couche Transposed convolution : de 7x7x256 vers14x14x128 
        model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    
        # Batch normalization
        model.add(BatchNormalization())
    
        # ReLU
        model.add(LeakyReLU(alpha=0.01))
        
        # Couche Transposed convolution : de 14x14x128 vers 14x14x64 
        model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
    
        # Batch normalization
        model.add(BatchNormalization())
        # ReLU
        model.add(LeakyReLU(alpha=0.01))
        # Couche Transposed convolution : de 14x14x64 vers 28x28x1 
        model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))
    
        # Tanh activation
        model.add(Activation('tanh'))
        #model.summary()
        # Le generator prend le bruit en entrée et génere des images
        z = Input(shape=(self.z_dim,))
        img = model(z)
        model = Model(z, img)
    
        return model
    

    def build_discriminator_net(self):
        model = Sequential()
        # la couche Convolutional : 28x28x1 vers 14x14x32 
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))
        # Leaky ReLU
        model.add(LeakyReLU(alpha=0.01))
        # Couhce Convolutional layer :14x14x32 vers 7x7x64 
        model.add(Conv2D(64, kernel_size=3, strides=2,input_shape=self.img_shape, padding='same'))
        # Batch normalization
        model.add(BatchNormalization())
        # Leaky ReLU
        model.add(LeakyReLU(alpha=0.01))
        # Couche Convolutional : 7x7x64 tensor vers 3x3x128 
        model.add(Conv2D(128, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))
        # Batch normalization
        model.add(BatchNormalization())
        # Leaky ReLU
        model.add(LeakyReLU(alpha=0.01))
        # Droupout
        model.add(Dropout(0.5)) # Pour éviter le sur-apprentissage 
        # Flatten 
        model.add(Flatten())
        # Couche connecter avec num_classes neurones
        model.add(Dense(self.num_classes))
        return model

    def build_discriminator_supervised(self,discriminator_net): 
        model = Sequential () 
        model.add (discriminator_net) 
        # Softmax donnant la distribution de probabilité sur les classes réelles 
        model.add(Activation('softmax')) 
        return model
    
    def build_discriminator_unsupervised(self,discriminator_net):
        model = Sequential()
        model.add(discriminator_net)
        
        def predict(x):
            # Transformez la distribution sur des classes réelles en une probabilité binaire 
            prediction = 1.0 - (1.0 / (K.sum(K.exp(x), axis=-1, keepdims=True) + 1.0))
            return prediction
        
        # Neurone de sortie : réel ou faux
        model.add(Lambda(predict))
        
        return model



    def train(self,sample_interval=80):
        d_accuracies = []
        d_losses = []
        g_losses = []
        iteration_checkpoints =[]
        dataset = Dataset()
        # X, y = dataset.training_set()
        # y = to_categorical(y, num_classes=self.num_classes)

        # Discriminator de base
        
        discriminator_net = self.build_discriminator_net()
        
        # Compiler le Discriminator pour l'entrainement supervisé
        discriminator_supervised = self.build_discriminator_supervised(discriminator_net)
        discriminator_supervised.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())
        
        # Compiler le Discriminator pour l'entrainement non supervisé
        discriminator_unsupervised = self.build_discriminator_unsupervised(discriminator_net)
        discriminator_unsupervised.compile(loss='binary_crossentropy', optimizer=Adam())
        
        # --------------------- Generator ---------------------------------#        
        generator = self.build_generator()
        
        # les paramétres de Discriminator sont constant pour l'entrainement de generator
        discriminator_unsupervised.trainable = False
    
       # Model GAN combiné : 
        def combined(generator, discriminator):
            model = Sequential()
            model.add(generator)
            model.add(discriminator)
            return model

        combined = combined(generator, discriminator_unsupervised)
        combined.compile(loss='binary_crossentropy', optimizer=Adam())
        
        # Des labels pour des exemples réels et faux
        real = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        half_batch = int(self.batch_size / 2)
        print('n_epochs=%d, n_batch=%d, 1/2=%d' % (self.n_epochs, self.batch_size, half_batch))
        # une fonction pour l'affichage des images
        def sample_images(n_epoch):
            r, c = 5, 5
            noise = np.random.normal(0, 1, (r * c, self.z_dim))
            gen_imgs = generator.predict(noise)

            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 1

            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                    axs[i,j].axis('off')
                    cnt += 1
            fig.show()
            fig.savefig("images/%d.png" % n_epoch)
            plt.close()
        def save_model(step):
            f1= 'models/dcgan_discriminator_weight_%04d.h5' % (step+1)
            discriminator_supervised.save(f1)
            #print("save model")
    
        for iteration in range(self.n_epochs):
            
            # ------------------------- Train Discriminator-----------------------------#
            
            #Exemples étiquetés
            imgs, labels = dataset.batch_labeled(self.batch_size)
            
            # One-hot encode labels
            labels = to_categorical(labels, num_classes=self.num_classes)
    
            # Exemples non étiquetés
            imgs_unlabeled = dataset.batch_unlabeled(self.batch_size)
            #  Générez un random  des images fausses 
            z = np.random.normal(0, 1, (self.batch_size, self.z_dim))
            gen_imgs = generator.predict(z)
            
            # S'entraîner sur des exemples réels labellisés
            d_loss_supervised, accuracy = discriminator_supervised.train_on_batch(imgs, labels)
            
            # S'entraîner sur des exemples réels non labellisés
            d_loss_real = discriminator_unsupervised.train_on_batch(imgs_unlabeled, real)
            
            # S'entraîner sur de faux exemples
            d_loss_fake = discriminator_unsupervised.train_on_batch(gen_imgs, fake)
            
            d_loss_unsupervised = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # --------------------- Train Generator--------------------------------------#
            
            # Générer un batch de fausses images
            z = np.random.normal(0, 1, (self.batch_size, self.z_dim))
            gen_imgs = generator.predict(z)
    
            # entrainement de generator
            g_loss = combined.train_on_batch(z, np.ones((self.batch_size, 1)))

            

            # g_losses.append(g_loss)
            # d_losses.append(d_loss_supervised)
            d_accuracies.append(accuracy)
            if (iteration + 1) % sample_interval == 0:
                # on le sauvgarde pour la visualisation
                d_losses.append(d_loss_supervised)
                iteration_checkpoints.append(iteration + 1)
                sample_images(iteration)
                save_model(iteration)
        return iteration_checkpoints, d_losses,d_accuracies
 
    
