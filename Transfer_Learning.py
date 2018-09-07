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
from PIL import Image
import numpy as np
import h5py

from keras import backend as K
K.set_image_data_format('channels_last')

#Load in the pretrained model without the top layer
model_location ="./Other Files/VGG_Face_pretrained_model_no_top.h5"
pretrained_model = keras.models.load_model(model_location)

#Set training and validation directory locations
traindir="./Dataset/train"
valdir="./Dataset/validation"

#Want to keep about an 80:20 ratio for Training images to Validation images
nTrain=80
nVal=20

#Generate batches of tensor image data
datagen = ImageDataGenerator(rescale = 1./255)
batch_size = 20

train_features = np.zeros(shape=(nTrain, 7, 7, 512))
train_labels = np.zeros(shape=(nTrain,3))

#Get training data from traindir
train_generator = datagen.flow_from_directory(
    traindir,
    target_size=(224,224),
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = True)


#Pass image through pretrained network to get 7 x 7 x 512 tensor
i = 0
for inputs_batch, labels_batch in train_generator:
    features_bathc = pretrained_model.predict(inputs_batch)
    train_features[i * batch_size: (i + 1) * batch_size] = features_batch
    train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i*batch_size >= nImages:
        break


#Reshape the Tensor into a vector
train_features = np.reshape(train_features, (nTrain, 7 * 7 * 512))