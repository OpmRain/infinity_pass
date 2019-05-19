from __future__ import print_function
import  tensorflow as tf
import  matplotlib.pyplot as plt
import  numpy as np
import  pandas as pd
import os, signal
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Model,load_model
from keras import layers
from keras import backend as K
import  numpy as np
from keras.callbacks import CSVLogger
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img,array_to_img
y=np.arange(start=1,stop=321)
from numpy import *
import math
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class CNN:


    def __init__(self):
        # Parameters for our graph; we'll output images in a 4x4 configuration
        #os.kill(os.getpid(), signal.pthread_kill())
        self.nrows = 4
        self.ncols = 4
        self.class_dict=None
        self.train_ben_fnames=None
        self.train_bless_fnames=None
        self.train_strato_fnames=None


        # Index for iterating over images
        self.pic_index = 0

        #For touch imageset
        self.base_touch_dir = 'touch_dataset'
        self.train_touch_dir = os.path.join(self.base_touch_dir, 'train')
        self.validation_touch_dir = os.path.join(self.base_touch_dir, 'validation')

        # Directory with our training cat pictures
        self.train_ben_touch_dir = os.path.join(self.train_touch_dir, 'ben')

        # Directory with our training dog pictures
        self.train_bless_touch_dir = os.path.join(self.train_touch_dir, 'bless')

        self.train_strato_touch_dir = os.path.join(self.train_touch_dir, 'strato')

        # Directory with our validation cat pictures
        self.validation_ben_touch_dir = os.path.join(self.validation_touch_dir, 'ben')

        # Directory with our validation dog pictures
        self.validation_bless_touch_dir = os.path.join(self.validation_touch_dir, 'bless')
        self.validation_strato_touch_dir = os.path.join(self.validation_touch_dir, 'strato')

        # For Signature imageset
        self.base_signature_dir = 'signature_dataset'
        self.train_signature_dir = os.path.join(self.base_signature_dir, 'train')
        self.validation_signature_dir = os.path.join(self.base_signature_dir, 'validation')

        # Directory with our training cat pictures
        self.train_ben_signature_dir = os.path.join(self.train_signature_dir, 'ben')

        # Directory with our training dog pictures
        self.train_bless_signature_dir = os.path.join(self.train_signature_dir, 'bless')

        self.train_strato_signature_dir = os.path.join(self.train_signature_dir, 'strato')

        # Directory with our validation cat pictures
        self.validation_ben_signature_dir = os.path.join(self.validation_signature_dir, 'ben')

        # Directory with our validation dog pictures
        self.validation_bless_signature_dir = os.path.join(self.validation_signature_dir, 'bless')
        self.validation_strato_signature_dir = os.path.join(self.validation_signature_dir, 'strato')



    def statistics(self):

        self.train_ben_fnames = os.listdir(self.train_ben_touch_dir)
        print(self.train_ben_fnames[:10],end="\n")

        self.train_bless_fnames = os.listdir(self.train_bless_touch_dir)
        self.train_bless_fnames.sort()

        self.train_strato_fnames = os.listdir(self.train_strato_touch_dir)
        self.train_strato_fnames.sort()
        print(self.train_bless_fnames[:10],end="\n")

        print('total training ben images:', len(os.listdir(self.train_ben_touch_dir)), end="\n")
        print('total training bless images:', len(os.listdir(self.train_bless_touch_dir)), end="\n")
        print('total training strato images:', len(os.listdir(self.train_strato_touch_dir)), end="\n")
        print('total validation ben images:', len(os.listdir(self.validation_ben_touch_dir)), end="\n")
        print('total validation bless images:', len(os.listdir(self.validation_bless_touch_dir)), end="\n")
        print('total validation strato images:', len(os.listdir(self.validation_strato_touch_dir)), end="\n")

    def show_image(self):
        # Set up matplotlib fig, and size it to fit 4x4 pics
        self.statistics()
        fig = plt.gcf()
        fig.set_size_inches(self.ncols * 4, self.nrows * 4)

        self.pic_index += 8
        next_cat_pix = [os.path.join(self.train_ben_touch_dir, fname)
                        for fname in self.train_ben_fnames[self.pic_index - 8:self.pic_index]]
        next_dog_pix = [os.path.join(self.train_bless_touch_dir, fname)
                        for fname in self.train_bless_fnames[self.pic_index - 8:self.pic_index]]

        for i, img_path in enumerate(next_cat_pix + next_dog_pix):
            # Set up subplot; subplot indices start at 1
            sp = plt.subplot(self.nrows, self.ncols, i + 1)
            sp.axis('Off')  # Don't show axes (or gridlines)

            img = mpimg.imread(img_path)
            plt.imshow(img)

        plt.show()

    def generate_generator_multiple(self,cust_gen, dir1, dir2, batch_size):
        genX1 = cust_gen.flow_from_directory(dir1,
                                              target_size=(250,235),
                                              class_mode='categorical',
                                              batch_size=batch_size,
                                              shuffle=False
                                              )

        genX2 = cust_gen.flow_from_directory(dir2,
                                              target_size=(250,235),
                                              class_mode='categorical',
                                              batch_size=batch_size,
                                              shuffle=False
                                              )

        self.class_dict=genX2.class_indices
        while True:
            X1i = genX1.next()
            X2i = genX2.next()
            #print(X1i[1], X2i[1])
            yield [X1i[0], X2i[0]], X2i[1]  # Yield both images and their mutual label

    def  cnn_layer(self):

        # Configure the TF backend session
        tf_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True))
        K.set_session(tf.Session(config=tf_config))

        # Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
        # the three color channels: R, G, and B

        touch_input = layers.Input(shape=(250, 235, 3))
        signature_input = layers.Input(shape=(250, 235, 3))

        self.statistics()


        # Convolution Touch
        x = layers.Conv2D(16, 3, activation='relu')(touch_input)
        x = layers.MaxPooling2D(2)(x)

        # Second convolution extracts 32 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = layers.Conv2D(32, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)

        # Third convolution extracts 64 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = layers.Convolution2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)

        # Flatten feature map to a 1-dim tensor
        x = layers.Flatten()(x)


        #Convolution Signature

        y = layers.Conv2D(16, 3, activation='relu')(signature_input)
        y = layers.MaxPooling2D(2)(y)

        # Second convolution extracts 32 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        y = layers.Conv2D(32, 3, activation='relu')(y)
        y = layers.MaxPooling2D(2)(y)

        # Third convolution extracts 64 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        y = layers.Convolution2D(64, 3, activation='relu')(y)
        y = layers.MaxPooling2D(2)(y)

        y = layers.Flatten()(y)

        combine_architecture = layers.concatenate([x, y], axis=-1)

        # Create a fully connected layer with ReLU activation and 512 hidden units
        final_encode = layers.Dense(512, activation='relu')(combine_architecture)

        # Add a dropout rate of 0.5
        final_encode = layers.Dropout(0.7)(final_encode)

        # Create output layer with a single node and sigmoid activation
        output = layers.Dense(3, activation='softmax')(final_encode)

               # Configure and compile the model
        model = Model([touch_input, signature_input], output)

        summarize = model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=0.0001),
                      metrics=['acc'])



        # All images will be rescaled by 1./255
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range = 30,
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            )
        #All images will be rescaled by 1./255
        #train_datagen = ImageDataGenerator(rescale=1. / 255)

        # Note that the validation data should not be augmented!
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        # Flow training images in batches of 32 using train_datagen generator
        train_generator =self.generate_generator_multiple(train_datagen,self.train_touch_dir,self.train_signature_dir,5)

        # Flow validation images in batches of 32 using test_datagen generator
        validation_generator = self.generate_generator_multiple(test_datagen,self.validation_touch_dir,self.validation_signature_dir,5)

        #Training
        csv_logger = CSVLogger('combine_log.csv', append=True)
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=54,  # 2000 images = batch_size * steps
            epochs=150,
            validation_data=validation_generator,
            validation_steps=6,  # 1000 images = batch_size * steps
            verbose=2,callbacks = [csv_logger])

        model.save("infinity_model.h5")

        # Let's define a new Model that will take an image as input, and will output
        # intermediate representations for all layers in the previous model.
        # successive_outputs = [layer.output for layer in model.layers]
        # visualization_model = Model(touch_input, successive_outputs)
        #
        # Let's prepare a random input image of a cat or dog from the training set.

        #cat_img_files = [os.path.join(self.train_ben_touch_dir, f) for f in self.train_ben_fnames]
        #dog_img_files = [os.path.join(self.train_bless_touch_dir, f) for f in self.train_bless_fnames]

        touch_path ="touch_dataset\\validation\\strato\\touchSequenceGraph_MR STRATO_session94 (Copy).png"
        signature_path = "signature_dataset\\validation\\strato\\SignatureCapture_MR STRATO_session91 (Copy).png"
        #random.choice(cat_img_files + dog_img_files)

        #print(img_path)

        img_touch = load_img(touch_path, target_size=(250, 235))  # this is a PIL image
        x = img_to_array(img_touch)  # Numpy array with shape (150, 150, 3)
        x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)=

        # Rescale by 1/255
        x /= 255

        img_signature = load_img(signature_path, target_size=(250, 235))  # this is a PIL image
        y = img_to_array(img_signature)  # Numpy array with shape (150, 150, 3)
        y = y.reshape((1,) + y.shape)  # Numpy array with shape (1, 150, 150, 3)=

        # Rescale by 1/255
        y /= 255

        available_classes= self.class_dict
        print(available_classes)
        prop=model.predict([x,y])
        print(available_classes)
        print(prop)

        y_classes =prop.argmax(axis=-1)


        for res in y_classes :
            for class_keys in available_classes.items():
                if class_keys[1]==res:
                  predicted_class=class_keys[0]
                  print("Predicted Individual ",predicted_class, "with propability of ",prop[0][res])
                  break






        # Let's run our image through our network, thus obtaining all
        # intermediate representations for this image.
        #successive_feature_maps = visualization_model.predict(x)

        #exit(1)
        # These are the names of the layers, so can have them as part of our plot
        #layer_names = [layer.name for layer in model.layers]



        # Now let's display our representations
        # for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        #     if len(feature_map.shape) == 4:
        #         # Just do this for the conv / maxpool layers, not the fully-connected layers
        #         n_features = feature_map.shape[-1]  # number of features in feature map
        #         # The feature map has shape (1, size, size, n_features)
        #         size = feature_map.shape[1]
        #         # We will tile our images in this matrix
        #         display_grid = np.zeros((size, size * n_features))
        #         for i in range(n_features):
        #             # Postprocess the feature to make it visually palatable
        #             x = feature_map[0, :, :, i]
        #             x -= x.mean()
        #             x /= x.std()
        #             x *= 64
        #             x += 128
        #             x = np.clip(x, 0, 255).astype('uint8')
        #             # We'll tile each filter into this big horizontal grid
        #             display_grid[:, i * size: (i + 1) * size] = x
        #         # Display the grid
        #         scale = 20. / n_features
        #         plt.figure(figsize=(scale * n_features, scale))
        #         plt.title(layer_name)
        #         plt.grid(False)
        #         plt.imshow(display_grid, aspect='auto', cmap='viridis')
        #
        #         # Retrieve a list of accuracy results on training and test data
        #         # sets for each training epoch
        #         acc = history.history['acc']
        #         val_acc = history.history['val_acc']
        #
        #         # Retrieve a list of list results on training and test data
        #         # sets for each training epoch
        #         loss = history.history['loss']
        #         val_loss = history.history['val_loss']
        #
        #         # Get number of epochs
        #         epochs = range(len(acc))
        #
        #         # Plot training and validation accuracy per epoch
        #         plt.plot(epochs, acc)
        #         plt.plot(epochs, val_acc)
        #         plt.title('Training and validation accuracy')
        #
        #         plt.figure()
        #
        #         # Plot training and validation loss per epoch
        #         plt.plot(epochs, loss)
        #         plt.plot(epochs, val_loss)
        #         plt.title('Training and validation loss')
        #
        #         plt.show()


    def typical_data_augmentation(self):

        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        self.statistics()
        img_path = os.path.join(self.train_ben_touch_dir, self.train_ben_fnames[2])
        img = load_img(img_path, target_size=(250, 235))  # this is a PIL image
        x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
        x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

        # The .flow() command below generates batches of randomly transformed images
        # It will loop indefinitely, so we need to `break` the loop at some point!
        i = 0
        for batch in datagen.flow(x, batch_size=1):
            plt.figure(i)
            imgplot = plt.imshow(array_to_img(batch[0]))
            plt.show()
            i += 1
            if i % 5 == 0:
                break

if __name__=="__main__":



     convolve=CNN()
     print([]*6)
     #convolve.extract()
     #convolve.show_image()
     convolve.cnn_layer()
     #convolve.typical_data_augmentation()


