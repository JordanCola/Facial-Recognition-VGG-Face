#Iterates through the given directory and identifies all faces in images,
#then crops them and scales them to 244x244 and saves them so that they 
#may be used for training our model.

import os
import cv2
from PIL import Image
import numpy as np

#Set image and cascade file paths
directory = "./Webcam Captures"
cascadePath = "./Other Files/haarcascade_frontalface_default.xml"

#The desired save location and name of subject for training
saveDirectory = "./Dataset/train/Jordan Svoboda"
name = "Jordan_Svoboda"

#Initialize i to be the number of files in the directory to avoid 
#overwriting any pieces of the dataset
i = len(os.listdir(saveDirectory)) + 1

for filename in os.listdir(directory):
    #Only use .jpg and .png files
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".JPG"):
        #Set image path to be the path to the image file
        imagePath = directory +'/' + filename
    else:
        continue    

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
        scaleFactor = 1.15,
        minNeighbors = 5,
        minSize = (30, 30),
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #Displays the image with the face highlighted.
    #Just to check the detection
    #cv2.imshow("Faces found", image)
    #cv2.waitKey(0)

    #Set face boundries for cropping
    (x, y, w, h) = faces[0]

    center_x = (x + w) / 2
    center_y = (y + h) / 2

    box = (x, y, x + w, y + h)

    #Crop Image and resize to 224x224
    image = Image.open(imagePath)
    cropImage = image.crop(box).resize((224,224))

    #Convert cropped image to an array so it can be read by imshow
    cropImage_cv = np.array(cropImage)

    #Convert color to RGB
    cropImage_cv = cv2.cvtColor(cropImage_cv, cv2.COLOR_BGR2RGB)
    
    #Display the cropped image until a user presses a key in
    #the image window
    #Should only be used to confirm script is running properly
    cv2.imshow("Cropped Image", cropImage_cv)
    cv2.waitKey(0)

    #Save the cropped image
    saveName = saveDirectory + "/" + name + str(i) + ".jpg"
    cv2.imwrite(saveName, cropImage_cv)
    i += 1