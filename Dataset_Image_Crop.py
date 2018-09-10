
#
#WIP
#

import cv2
from PIL import Image
import numpy as np

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
    #image,
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
cv2.imshow("Faces found", image)
cv2.waitKey(0)

#Crop image and scale to 224x224

image = Image.open(imagePath)

(x, y, w, h) = faces[0]

center_x = (x + w) / 2
center_y = (y + h) / 2

box = (x, y, x + w, y + h)

#Crop Image
cropImage = image.crop(box).resize((224,224))
cropImage_cv = np.array(cropImage)
cropImage_cv = cropImage_cv[:, :, ::-1].copy()
cv2.imshow("Cropped Image", cropImage_cv)
cv2.waitKey(0)