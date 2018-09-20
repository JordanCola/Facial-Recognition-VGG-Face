#
#WIP
#

#A script to train a pretrained model on new data using the transfer
#learning method.

import keras.utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras import optimizers
from PIL import Image
import numpy as np
import h5py
import cv2
from matplotlib import pyplot as plt

from keras import backend as K
K.set_image_data_format('channels_last')

#Load in the pretrained model without the top layer
model_location ="./Other Files/VGG_Face_pretrained_model_no_top.h5"

#Set training and validation directory locations
train_dir="./Dataset/train"
val_dir="./Dataset/validation"

#Number of training and validation images. Shoule be 5:1 ratio
nTrain = 20
nValidation = 5
img_width, image_height = 224, 224
batch_size = 20

def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1./255)

    pretrained_model = keras.models.load_model(model_location)
    #pretrained_model.summary()
    generator = datagen.flow_from_directory(
        train_dir,
        target_size = (img_width, image_height),
        batch_size = batch_size,
        class_mode = None,
        shuffle = False)
    bottleneck_features_train = pretrained_model.predict_generator(
        generator, nTrain // batch_size)

    print(str(bottleneck_features_train))

    np.save(open('bottleneck_features_train.npy', 'wb'),
        bottleneck_features_train)

    
    generator = datagen.flow_from_directory(
        val_dir,
        target_size = (img_width, image_height),
        batch_size = batch_size,
        class_mode = None,
        shuffle = False)
    bottleneck_features_validation = pretrained_model.predict_generator(
        generator, nValidation // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'wb'),
        bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_labels = np.array(
        [0] * (nTrain // 2) + [1] * (nTrain // 2))

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array(
        [0] * (nValidation // 2) + [1] * (nValidation // 2))

    newModel = keras.models.Sequential()
    newModel.add(Flatten(input_shape=train_data.shape[1:]))
    newModel.add(keras.layers.Dense(256, activation = 'relu'))
    newModel.add(keras.layers.Dropout(0.5))
    newModel.add(keras.layers.Dense(1, activation = 'softmax'))

    newModel.compile(optimizer = optimizers.RMSprop(lr=2e-4),
                loss = 'binary_crossentropy',
                metrics =['accuracy'])

    #ValueError: Error when checking input: expected flatten_1_input to have 4 dimensions, but got array with shape (0, 1)
    #Not sure what is causing this
    newModel.fit(train_data,
                train_labels,
                epochs = 20,
                batch_size = batch_size,
                validation_data = (validation_data, validation_labels))
    
save_bottleneck_features()
train_top_model()