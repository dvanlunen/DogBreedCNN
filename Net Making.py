from lasagne import layers
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet, BatchIterator
from lasagne.nonlinearities import softmax
import os
from PIL import Image, ImageOps
import numpy as np

path = 'D:/dog breed data/CU_Dogs/'
imgs = os.listdir(path + 'trainProcessed/')
numImgs = len(imgs)
PIXELS = 64
X = np.zeros((numImgs, 3, PIXELS, PIXELS), dtype='float32')
y = np.zeros(numImgs)

# get images read
# (1) set correct breed value
# (2) normalize image data
# (3) reshape data

for i in range(0, numImgs):
    # breed is the first 3 characters of the file name
    y[i] = int(imgs[0][0:3])
    img = Image.open(path + 'trainProcessed/' + imgs[i])
    img = ImageOps.fit(img, (PIXELS, PIXELS), Image.ANTIALIAS)
    img = np.asarray(img, dtype='float32') / 255.
    img = img.transpose(2, 0, 1).reshape(3, PIXELS, PIXELS)
    X[i] = img

net1 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('conv3', layers.Conv2DLayer),
        ('hidden4', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, 64, 64),
    conv1_num_filters=32, conv1_filter_size=(5, 5),
    pool1_pool_size=(2, 2),
    dropout1_p=0.2,
    conv2_num_filters=64, conv2_filter_size=(5, 5),
    pool2_pool_size=(2, 2),
    dropout2_p=0.2,
    conv3_num_filters=128, conv3_filter_size=(5, 5),
    hidden4_num_units=500,
    output_num_units=133,
    output_nonlinearity=softmax,
    update_learning_rate=0.001,
    update_momentum=0.9,
    batch_iterator_train=BatchIterator(batch_size=100),
    batch_iterator_test=BatchIterator(batch_size=100),
    use_label_encoder=True,
    regression=False,
    max_epochs=10,
    verbose=1,
    )

X = X.astype(np.float32)
y = y.astype(np.int32)
net1.fit(X, y)
