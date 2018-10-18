#
#WIP
#

#A script to train a pretrained model on new data using the transfer
#learning method.
import os
import keras.utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras import optimizers
from PIL import Image
import numpy as np
import h5py

from keras import backend as K
K.set_image_data_format('channels_last')

#Load in the pretrained model without the top layer
model_location ="./Other Files/VGG_Face_pretrained_model_no_top.h5"

#Set training and validation directory locations
train_dir="./Dataset/train"
val_dir="./Dataset/validation"

labels = os.listdir(train_dir)
np.save(open('labels.npy', 'wb'),labels)
total_classes = len(labels)
#Number of training and validation images. Should be 5:1 ratio
#Use an even number for nTrain and nLabel, otherwise there will be a mismatch
#In number of labels and number of images

import os
nTrain = sum([len(files) for r,d, files in os.walk("./Dataset/train")])
nValidation = sum([len(files) for r,d, files in os.walk("./Dataset/validation")])

img_width, image_height = 224, 224
batch_size = 2

def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1./255)

    #Load pretrained model
    pretrained_model = keras.models.load_model(model_location)
    
    #Generate training data from ./Dataset/train directory
    generator = datagen.flow_from_directory(
        train_dir,
        target_size = (img_width, image_height),
        batch_size = batch_size,
        class_mode = None,
        shuffle = True)
    bottleneck_features_train = pretrained_model.predict_generator(
        generator, nTrain // batch_size)
    
    #Save training data
    np.save(open('bottleneck_features_train.npy', 'wb'),
        bottleneck_features_train)

    #Generate validation data from ./Dataset/validation directory
    generator = datagen.flow_from_directory(
        val_dir,
        target_size = (img_width, image_height),
        batch_size = batch_size,
        class_mode = None,
        shuffle = True)
    bottleneck_features_validation = pretrained_model.predict_generator(
        generator, nValidation // batch_size)
    
    #Save validation data
    np.save(open('bottleneck_features_validation.npy', 'wb'),
        bottleneck_features_validation)


#A method to add the layers for the retrained model to an existing model. Used to create the model for transfer learning
#and used to add layers to the pretrained model to create a new model for facial recognition. Takes in a pre-exisitng model as input,
#adds the layers, then returns the model with the layers added.

def addSmallModel(inputModel):
    if len(inputModel.layers) == 0:
        inputModel.add(Flatten(input_shape=(7, 7, 512, total_classes)))
    else:
        inputModel.add(Flatten())
    
    inputModel.add(Dense(256, activation='relu'))
    inputModel.add(Dropout(0.5))
    inputModel.add(Dense(total_classes, activation='sigmoid'))    
    #inputModel.summary()
    return inputModel

def train_top_model():
    #Load in the train data and create label array
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_labels = np.array(
        [0] * (nTrain // 2) + [1] * (nTrain // 2))

    #print(str(train_labels))

    #Load in the train data and create label array
    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array(
        [0] * (nValidation // 2) + [1] * (nValidation // 2))
    
    #print(str(validation_labels))
    #Build new, small model to train on
    newModel = keras.models.Sequential()
    newModel = addSmallModel(newModel)
    #newModel.summary()

    #Convert integer class vector to binary class matrix
    #Allows for multiple probability vector outputs fo proper face identification
    from keras.utils.np_utils import to_categorical
    cat_train_data = to_categorical(train_data, num_classes= total_classes)
    cat_validation_data = to_categorical(validation_data, num_classes= total_classes)
    cat_train_labels = to_categorical(train_labels, num_classes= total_classes)
    cat_validation_labels = to_categorical(validation_labels, num_classes= total_classes)
    #Compile new model
    newModel.compile(optimizer = optimizers.RMSprop(lr=2e-4),
                loss = 'categorical_crossentropy',
                metrics =['accuracy'])
 
    #Train the model on the new data
    newModel.fit(cat_train_data,
                cat_train_labels,
                epochs = 40,
                batch_size = batch_size,
                validation_data = (cat_validation_data, cat_validation_labels))
    newModel.save_weights("./Other Files/Transfer Weights.h5")


def createTransferModel():
    pretrained_model = keras.models.load_model(model_location)

    pretrained_model = addSmallModel(pretrained_model)

    #Compile new model
    pretrained_model.compile(optimizer = optimizers.RMSprop(lr=2e-4),
                loss = 'categorical_crossentropy',
                metrics =['accuracy'])

    #pretrained_model.summary()

    #Set weights for small top model
    pretrained_model.load_weights("./Other Files/Transfer Weights.h5", by_name = True)

    #Recompile model with new weights
    pretrained_model.compile(optimizer = optimizers.RMSprop(lr=2e-4),
                loss = 'categorical_crossentropy',
                metrics =['accuracy'])
    #Save new model
    pretrained_model.save("./Other Files/Transfer_Model.h5")  

save_bottleneck_features()
train_top_model()
createTransferModel()