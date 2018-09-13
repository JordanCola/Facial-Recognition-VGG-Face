#A Script to allow the usage of the webcam to capture frames
#for use in dataset creation. Run the script and press picKey
#to capture a frame and 'q' to exit the script

import cv2
import sys

picKey = ' ' #Currently set to space

#Set cascade file path and create the face cascade
cascPath = "./Other Files/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

#Set video source to default webcam
video_capture = cv2.VideoCapture(0)

#Initialize the count variable, used in naming frame images
count = 0

while True:
    #Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.15,
        minNeighbors = 5,
        minSize = (30,30),
    )

    cv2.imshow('Video', frame)

    #Take a picture if picKey is pressed
    if cv2.waitKey(1) & 0xFF == ord(picKey):
        cv2.imwrite("./Webcam Captures/frame%d.jpg" % count, frame)
        count += 1

    #Exit script if quitKey is pressed
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break
    


#When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()