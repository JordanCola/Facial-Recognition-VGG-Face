#Loading in the matlab file containing the VGG model weights
from scipy.io import loadmat

#Should get changed depending on where file is
filename = "./Other Files/vgg-face.mat"

data = loadmat(filename,matlab_compatible=False, struct_as_record = False)

#Sets the descriptions we will need access for prediction output
description = data['meta'][0,0].classes[0,0].description


file=open("names.txt","w")

for i in description:
    name = str(i[0])[2:-2]
    file.write(name +",")
