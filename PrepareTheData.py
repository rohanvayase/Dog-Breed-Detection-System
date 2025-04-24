import numpy as np
import cv2

IMAGE_SIZE = (331,331)
IMAGE_FULL_SIZE = (331,331,3)

trainMyImageFolder = "D:/PROJECTS/Dog Breed Identification/train"

import pandas as pd

df = pd.read_csv('D:/PROJECTS/Dog Breed Identification/labels.csv')
print("head of lables :")
print("=================")

print(df.head())
print(df.describe())

print("Group by labels: ")
grouplables = df.groupby("breed")["id"].count()
print(grouplables.head(10))


imgPath = "D:/PROJECTS/Dog Breed Identification/train/00ba244566e36e0af3d979320fd3017f.jpg"
img = cv2.imread(imgPath)
#cv2.imshow("img", img)
#cv2.waitKey(0)


# prepare all the images and labels as Numpy arrays

allImages = []
allLables = []
import os

for ix , (image_name, breed) in enumerate(df[['id' , 'breed']].values):
    img_dir = os.path.join(trainMyImageFolder, image_name + '.jpg')
    print(img_dir)
    
    img = cv2.imread(img_dir)
    resized = cv2.resize(img,IMAGE_SIZE, interpolation= cv2.INTER_AREA)
    allImages.append(resized)
    allLables.append(breed)
    
print(len(allImages))
print(len(allLables))


print("save the data: ")
np.save("D:/PROJECTS/Dog Breed Identification/allImages.npy", allImages)
np.save("D:/PROJECTS/Dog Breed Identification/alllabels.npy", allLables)

print("finished saving the data.....")