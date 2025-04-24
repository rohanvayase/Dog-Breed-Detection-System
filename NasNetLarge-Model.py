import numpy as np

IMAGE_SIZE = (331,331)
IMAGE_FULL_SIZE = (331,331,3)
batchSize = 8

allImages = np.load("D:/PROJECTS/Dog Breed Identification/allImages.npy")
allLabels = np.load("D:/PROJECTS/Dog Breed Identification/alllabels.npy")

print(allImages.shape)
print(allLabels.shape)

#convert the lables text to integers
print(allLabels)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
integerLables = le.fit_transform(allLabels)
print(integerLables)

numOfCategories = len(np.unique(integerLables))
print(numOfCategories)

from tensorflow.keras.utils import to_categorical

allLabelsForModel = to_categorical(integerLables, num_classes = numOfCategories)
print(allLabelsForModel)


allImagesForModel = allImages / 255.0


from sklearn.model_selection import train_test_split

print("Before split train and test: ")

X_train, X_test, y_train, y_test = train_test_split(allLabelsForModel, allImagesForModel, test_size=0.3, random_state=42)

print("X_train, X_test, y_train, y_test -------> shapes")
print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)


#free some memory
del allImages
del allLabels
del integerLables
del allImagesForModel

