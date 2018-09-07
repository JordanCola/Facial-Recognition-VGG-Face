# Facial-Recognition-VGG-Face
Code for facial recognition using the VGG Face Model using Keras with a TensorFlow backend.

## Before using, ensure that you have the vgg-face.mat file, Keras and TensorFlow installed. 
The vgg-face.mat file can be found [here.](http://www.vlfeat.org/matconvnet/pretrained/#face-recognition)
This code assumes it is located in the Other Files directory, but the file path in the code should be updated based on where you choose to save the file.

## Getting Started
Once the vgg-face.mat file is downloaded and the paths are updated, run the Trained_Model_Creation.py script to create the pretrained model. Then the VGG_Face_prediction.py script can be run to confirm that evrything is properly set up.

If you wish to remove the fully connected top layer for transfer learning or fine-tuning, simply replace
```
model = blankModel(True)

```
with 
```

model = blankModel(False)

```
