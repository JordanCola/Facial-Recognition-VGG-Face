#A script to fine tune the pretrained VGG Face Model that was
#created in "Trained Model Creation.py"
import keras.utils
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from PIL import Image
import numpy as np
import h5py

from keras import backend as K
K.set_image_data_format('channels_last')

#Load in the pretrained model without it's top layer
modelLocation = "C:/Users/Jordan Svoboda/Documents/VGG Face Model Project/Other Files/VGG_Face_pretrained_model_no_top.h5"
pretrainedModel = keras.models.load_model(modelLocation)

#Freeze all layers except last 4
for layer in pretrainedModel.layers[:-4]:
    layer.trainable = False

#Check trainable layers
#for layer in pretrainedModel.layers:
#    print(layer, layer.trainable)

#Create new model
newModel = keras.models.Sequential()

#Add Pretrained base
newModel.add(pretrainedModel)

#Add new layers
newModel.add(Flatten())
newModel.add(Dense(1024, activation='relu'))
newModel.add(Dropout(0.5))
newModel.add(Dense(3, activation='softmax'))

#Print a summary of the new model
newModel.summary()