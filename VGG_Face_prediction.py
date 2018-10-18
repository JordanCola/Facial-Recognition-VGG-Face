#Based on instructions and code found at 
#https://aboveintelligent.com/face-recognition-with-keras-and-opencv-2baf2a83b799


from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
import keras.utils
from PIL import Image
import numpy as np
import h5py
import cv2
from keras import backend as K
K.set_image_data_format('channels_last')

#Load in the trained model
model = keras.models.load_model("./Other Files/VGG_Face_pretrained_model.h5")
#Run Transfer_Learning.py to get the Transfer_Model.h5 file
#model = keras.models.load_model("./Other Files/Transfer_Model.h5")

#Loading in the matlab file containing the VGG model weights
from scipy.io import loadmat

#Should get changed depending on where file is
#Use this for the pretrained model
filename = "names.txt"
file = open(filename)
for line in file:
    description=line.split(',')

#Make sure this is alphabetized
#Use this description for Transfer_Model. Need to add subjects in alphabetical order
#description = np.load(open('labels.npy', 'rb'))
#print(str(description))

#The prediction function
def prediction(kmodel, img):
    imarr = np.array(img).astype(np.float32)

    #Turn image into a 1-element batch
    imarr = np.expand_dims(imarr, axis=0)

    #Prediction Probability vector
    out= kmodel.predict(imarr)
    #print(str(out))

    #Most Probable item
    best_index = np.argmax(out, axis=1)[0]

    best_name = description[best_index]
    print('\nPrediction:')
    print(best_index, best_name, out[0,best_index], [np.min(out), np.max(out)])

#BEGIN DETECTION IN UNCROPPED IMAGES

#Based on instructions from 
# https://realpython.com/face-recognition-with-python/

#Set image and cascade file paths
imagePath = "./Test Images/Mark_Hamill.jpg"
cascadePath = "./Other Files/haarcascade_frontalface_default.xml"

#Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascadePath)

#Read image
image = cv2.imread(imagePath)

#Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Detect faces in image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor = 1.3,
    minNeighbors = 5,
    minSize = (30, 30),
)

print('Found {0} faces!'.format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#Displays the image with the face highlighted.
#Just to check the detection
#cv2.imshow("Faces found", image)
#cv2.waitKey(0)

#Crop image and scale to 224x224

image = Image.open(imagePath)

(x, y, w, h) = faces[0]

center_x = (x + w) / 2
center_y = (y + h) / 2

box = (x, y, x + w, y + h)

#Crop Image
cropImage = image.crop(box).resize((224,224))

#Perform the prediction on the cropped image
prediction(model, cropImage)