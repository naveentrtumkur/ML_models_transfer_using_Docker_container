# This is a simple python program to find if the face exists in an image or not.
# It prints out "Face Exists", if a face is found in an image, else prints "No face found"

#Input would be a folder name and it classifies the images as "Face Present" or "Face not Present"
#Source: Used StackOverflow to learn this process and wrote the below code based on my learning.

import dlib
from skimage import io, img_as_float
import os
import pandas as pd
import matplotlib.image as mpimg

# Now try to read a series of images from the folder and perform Face Detection.
images = []
for filename in os.listdir("/Users/naveentr/Desktop/VMM_Docker/First_baseline_transfer/Face_recognition/Detection/Images"):
    image = mpimg.imread(os.path.join("/Users/naveentr/Desktop/VMM_Docker/First_baseline_transfer/Face_recognition/Detection/Images", filename))

    #image = io.imread('naveen_id.jpg') #Pass an image to detect if a face exists.

    #Get an object for the detector.
    face_detector = dlib.get_frontal_face_detector()

    #Detect for faces and upsample the image 1x

    detection = face_detector(image,1)

    if(len(detection) ==0):
        print("No face found, Image = ",filename)
    else:
        print("There were ",len(detection),"face(s) found, Image Name=",filename)



