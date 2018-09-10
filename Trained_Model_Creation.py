#A script that creates a keras Sequential model to identify faces. This script creates the model
#architecture and then loads the weights from the VGG Face Project. The file with the weights can
#be downloads

#Based on instructions and code found at 
#https://aboveintelligent.com/face-recognition-with-keras-and-opencv-2baf2a83b799
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from PIL import Image
import numpy as np
import h5py

from keras import backend as K
K.set_image_data_format('channels_last')

#Building block of convolution layers
def convblock(cdim, nb, bits=3):
    L=[]

    for k in range(1,bits+1):
        convname = 'conv'+str(nb)+'_'+str(k)
        L.append(Convolution2D(cdim, kernel_size=(3, 3),padding='same',activation='relu',name=convname) )
    
    L.append( MaxPooling2D((2, 2), strides=(2, 2)) )
    
    return L

 #A function to Initialize the Model. Allows the removal of the top layer of the model,
 #which consists of fully connected layers.
def blankModel(topLayer):

    #Using Sequential Model
    newModel = Sequential()

    #Trick :
    #dummy-permutation = identity to specify input shape
    #index starts at 1 as 0 is the sample dimension
    newModel.add( Permute((1,2,3), input_shape=(224,224,3)) )

    #Create Convolution Layers
    for l in convblock(64, 1, bits=2):
        newModel.add(l)
    
    for l in convblock(128, 2, bits=2):
        newModel.add(l)

    for l in convblock(256, 3, bits=3):
        newModel.add(l)
    
    for l in convblock(512, 4, bits=3):
        newModel.add(l)
    
    for l in convblock(512, 5, bits=3):
        newModel.add(l)


    #This will be used to fine tune the model. We will not need a top layer when we
    #fine-tune the model, as we will be training the other layers further.
    if topLayer:
        newModel.add( Convolution2D(4096, kernel_size=(7, 7), activation='relu', name='fc6') )
        newModel.add( Dropout(0.5) )
        newModel.add( Convolution2D(4096, kernel_size=(1, 1), activation='relu', name='fc7') )
        newModel.add( Dropout(0.5))
        newModel.add( Convolution2D(2622, kernel_size=(1, 1), activation='relu', name='fc8') )
        newModel.add( Flatten() )
        newModel.add( Activation('softmax') )

    #Print a summary of created layers
    #newModel.summary()
    
    return newModel

#Create our model
model = blankModel(True)

#Loading in the matlab file containing the VGG model weights
from scipy.io import loadmat

#Should get changed depending on where file is
filename = './Other Files/vgg-face.mat'

data = loadmat(filename,matlab_compatible=False, struct_as_record = False)

layers = data['layers']
description = data['meta'][0,0].classes[0,0].description

#BEGIN LOADING WEIGHTS
kerasnames = [layer.name for layer in model.layers]

#Needed to ensure MatLab data is in correct format.
#I believe the data is in the correct format already, so I am not using this
prmt = (0,1,2,3)

#Set the weights for the model
for i in range(layers.shape[1]):
    matname = layers[0,i][0,0].name[0]
    if matname in kerasnames:
        kindex = kerasnames.index(matname)
        #print(matname)
        
        l_weights = layers[0,i][0,0].weights[0,0]
        l_bias = layers[0,i][0,0].weights[0,1]
        flip_l_weights = l_weights.transpose(prmt)
        
        #Check to make sure data in weights and bias matches the
        #MatLab data
        assert (l_weights.shape == model.layers[kindex].get_weights()[0].shape)
        assert (l_bias.shape[1] == 1)
        assert (l_bias[:,0].shape == model.layers[kindex].get_weights()[1].shape)
        assert (len(model.layers[kindex].get_weights()) == 2)
        model.layers[kindex].set_weights([flip_l_weights, l_bias[:,0]])

#Compile the Model
model.compile(optimizer = 'adam', loss = "categorical_crossentropy")

#Save the model
model.save("./Other Files/VGG_Face_pretrained_model.h5")