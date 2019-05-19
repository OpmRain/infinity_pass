from __future__ import print_function
import  tensorflow as tf
import  matplotlib.pyplot as plt
import  numpy as np
import  pandas as pd
import os, signal
from tensorflow.python.keras.callbacks import CSVLogger
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img,array_to_img
import random
import  keras
import  nipy
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
class CNN:

    def __init__(self):
        # Parameters for our graph; we'll output images in a 4x4 configuration
        #os.kill(os.getpid(), signal.pthread_kill())
        self.nrows = 4
        self.ncols = 4

        self.train_ben_fnames=None
        self.train_bless_fnames=None

        # Index for iterating over images
        self.pic_index = 0
        self.base_dir = 'touch_dataset'
        self.train_dir = os.path.join(self.base_dir, 'train')
        self.validation_dir = os.path.join(self.base_dir, 'validation')

        # Directory with our training cat pictures
        self.train_ben_dir = os.path.join(self.train_dir, 'ben')

        # Directory with our training dog pictures
        self.train_bless_dir = os.path.join(self.train_dir, 'bless')

        self.train_strato_dir = os.path.join(self.train_dir, 'strato')

        # Directory with our validation cat pictures
        self.validation_ben_dir = os.path.join(self.validation_dir, 'ben')

        # Directory with our validation dog pictures
        self.validation_bless_dir = os.path.join(self.validation_dir, 'bless')
        self.validation_strato_dir = os.path.join(self.validation_dir, 'strato')

    def extract(self):
        local_zip = os.path.join(os.path.dirname(__file__), 'ben_and_bless_filtered.zip')
        zip_ref = zipfile.ZipFile(local_zip, 'r')
        zip_ref.extractall(os.path.join(os.path.dirname(__file__)))
        zip_ref.close()
        print("Extraction complete",end="\n")

    def statistics(self):

        self.train_ben_fnames = os.listdir(self.train_ben_dir)
        print(self.train_ben_fnames[:10],end="\n")

        self.train_bless_fnames = os.listdir(self.train_bless_dir)
        self.train_bless_fnames.sort()

        self.train_strato_fnames = os.listdir(self.train_strato_dir)
        self.train_strato_fnames.sort()
        print(self.train_bless_fnames[:10],end="\n")
        print('total training ben images:', len(os.listdir(self.train_ben_dir)),end="\n")
        print('total training bless images:', len(os.listdir(self.train_bless_dir)),end="\n")
        print('total training strato images:', len(os.listdir(self.train_strato_dir)), end="\n")
        print('total validation ben images:', len(os.listdir(self.validation_ben_dir)),end="\n")
        print('total validation bless images:', len(os.listdir(self.validation_bless_dir)),end="\n")
        print('total validation strato images:', len(os.listdir(self.validation_strato_dir)), end="\n")

    def show_image(self):
        # Set up matplotlib fig, and size it to fit 4x4 pics
        self.statistics()
        fig = plt.gcf()
        fig.set_size_inches(self.ncols * 4, self.nrows * 4)

        self.pic_index += 8
        next_cat_pix = [os.path.join(self.train_ben_dir, fname)
                        for fname in self.train_ben_fnames[self.pic_index - 8:self.pic_index]]
        next_dog_pix = [os.path.join(self.train_bless_dir, fname)
                        for fname in self.train_bless_fnames[self.pic_index - 8:self.pic_index]]

        for i, img_path in enumerate(next_cat_pix + next_dog_pix):
            # Set up subplot; subplot indices start at 1
            sp = plt.subplot(self.nrows, self.ncols, i + 1)
            sp.axis('Off')  # Don't show axes (or gridlines)

            img = mpimg.imread(img_path)
            plt.imshow(img)

        plt.show()


    def  cnn_layer(self):

        # Configure the TF backend session


        # Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
        # the three color channels: R, G, and B
        img_input = layers.Input(shape=(250, 235, 3))

        self.statistics()

        # First convolution extracts 16 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = layers.Conv2D(16, 3, activation='relu')(img_input)
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

        # Create a fully connected layer with ReLU activation and 512 hidden units
        x = layers.Dense(512, activation='relu')(x)

        # Add a dropout rate of 0.5
        x = layers.Dropout(0.5)(x)

        # Create output layer with a single node and sigmoid activation
        output = layers.Dense(3, activation='sigmoid')(x)

        # Configure and compile the model
        model = Model(img_input, output)

        summarize = model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=0.0001),
                      metrics=['acc'])




        # All images will be rescaled by 1./255
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            )
        #All images will be rescaled by 1./255
        #train_datagen = ImageDataGenerator(rescale=1. / 255)

        # Note that the validation data should not be augmented!
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        # Flow training images in batches of 32 using train_datagen generator
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,  # This is the source directory for training images
            target_size=(250, 235),  # All images will be resized to 150x150
            batch_size=10,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode="categorical")

        # Flow validation images in batches of 32 using test_datagen generator
        validation_generator = test_datagen.flow_from_directory(
            self.validation_dir,
            target_size=(250, 235),
            batch_size=20,
            class_mode="categorical")


        #Training
        csv_logger = CSVLogger('touch_log.csv', append=True)
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=100,  # 2000 images = batch_size * steps
            epochs=150,
            validation_data=validation_generator,
            validation_steps=50,  # 1000 images = batch_size * steps
            verbose=2,
            callbacks = [csv_logger])

        # Let's define a new Model that will take an image as input, and will output
        # intermediate representations for all layers in the previous model.
        # successive_outputs = [layer.output for layer in model.layers]
        # visualization_model = Model(img_input, successive_outputs)
        #
        # Let's prepare a random input image of a cat or dog from the training set.

        cat_img_files = [os.path.join(self.train_ben_dir, f) for f in self.train_ben_fnames]
        dog_img_files = [os.path.join(self.train_bless_dir, f) for f in self.train_bless_fnames]
        img_path ="touch_dataset\\validation\\strato\\touchSequenceGraph_MR STRATO_session94 (Copy).png"
        #random.choice(cat_img_files + dog_img_files)

        print(img_path)

        img = load_img(img_path, target_size=(250, 235))  # this is a PIL image
        x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
        x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)=

        # Rescale by 1/255
        x /= 255

        available_classes=train_generator.class_indices

        prop=model.predict(x)
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
        img_path = os.path.join(self.train_ben_dir, self.train_ben_fnames[2])
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


