# Image Processing
# 1. Resize
# (attempt to isolate faces at another time)

# Help From
# https://github.com/faizankshaikh/JuliaKaggle/blob/master/resizeData.py

import os
from skimage.io import imread, imsave
from skimage.transform import resize

# Directory set up
path = 'D:/dog breed data/CU_Dogs/'
if not os.path.exists(path + '/trainProcessed'):
    os.makedirs(path + '/trainProcessed')
if not os.path.exists(path + '/testProcessed'):
    os.makedirs(path + '/testProcessed')

# Process training images
trainpaths = ([path + 'dogImages/' + line.rstrip('\n')
              for line in open(path + '/training.txt')])


for i, nameFile in enumerate(trainpaths):
    image = imread(nameFile)
    imageResized = resize(image, (64, 64))
    split_filename = nameFile.split("/")
    newName = (path + 'trainProcessed/' +
               split_filename[-2][0:3] + split_filename[-1])
    imsave(newName, imageResized)

# Process testing images
testpaths = ([path + 'dogImages/' + line.rstrip('\n')
              for line in open(path + '/testing.txt')])

for i, nameFile in enumerate(testpaths):
    image = imread(nameFile)
    imageResized = resize(image, (64, 64))
    split_filename = nameFile.split("/")
    newName = (path + 'testProcessed/' +
               split_filename[-2][0:3] + split_filename[-1])
    imsave(newName, imageResized)
