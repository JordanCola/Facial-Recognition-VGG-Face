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

from keras import backend as K
K.set_image_data_format('channels_last')

#Load in the pretrained model without the top layer
model_location ="./Other Files/VGG_Face_pretrained_model_no_top.h5"
pretrained_model = keras.models.load_model(model_location)

#Set training and validation directory locations
train_dir="./Dataset/train"
val_dir="./Dataset/validation"

#Number of training and validation images. Shoule be 5:1 ratio
nTrain = 20
nVal = 5

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

#Load training images and generate batches of images and labels
train_features = np.zeros(shape=(nTrain, 7, 7, 512))
train_labels = np.zeros(shape=(nTrain,3))

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size = (224,224),
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = True
)

#Load validation images and generate batches of images and labels
val_features = np.zeros(shape=(nVal, 7, 7, 512))
val_labels = np.zeros(shape=(nVal,3))

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size = (224,224),
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = True
)


i = 0
for inputs_batch, labels_batch in train_generator:
    features_batch = pretrained_model.predict(inputs_batch)
    train_features[i * batch_size : (i + 1) * batch_size] = features_batch
    train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nTrain:
        break 

train_features = np.reshape(train_features, (nTrain, 7 * 7 * 512))

i = 0
for inputs_batch, labels_batch in val_generator:
    features_batch = pretrained_model.predict(inputs_batch)
    val_features[i * batch_size : (i + 1) * batch_size] = features_batch
    val_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nVal:
        break 
        
val_features = np.reshape(val_features, (nVal, 7 * 7 * 512))

newModel = keras.models.Sequential()
newModel.add(keras.layers.Dense(256, activation = 'relu', input_dim = 7 * 7 * 512))
newModel.add(keras.layers.Dropout(0.5))
newModel.add(keras.layers.Dense(3, activation = 'softmax'))

newModel.compile(optimizer = optimizers.RMSprop(lr=2e-4),
                 loss = 'categorical_crossentropy',
                 metrics =['acc'])

history = newModel.fit(train_features,
                       train_labels,
                       epochs = 20,
                       batch_size = batch_size,
                       validation_data = (val_features, val_labels))

fnames = val_generator.filenames

ground_truth = val_generator.classes

label2index = val_generator.class_indices

index2label = dict((v,k) for k,v in label2index.items())

predictions = newModel.predict_classes(val_features)
prob = newModel.predict(val_features)

errors = np.where(predictions != ground_truth)[0]
print('Number of errors = {}/{}'.format(len(errors), nVal))

for i in range (len(errors)):
    pred_class = np.argmax(prob[errors[i]])
    pred_label = index2label[pred_class]

    print("Original label: {}, Prediction: {}, Confidence: {:.3f}".format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        prob[errors[i]][pred_class]))

    original = load_img('{}/{}'.format(val_dir, fnames[errors[i]]))
    plt.imshow(original)
    plt.show()
