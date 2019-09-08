import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils, to_categorical
from sklearn.model_selection import train_test_split

img_rows, img_cols = 28, 28
nb_pool = 2
nb_conv = 3

# loading data from images.npy and labels.npy
imgData = np.load('images.npy')
labelData = np.load('labels.npy')

# split into 20% test set, 20% validation set, and 60% training set
X_train, X_test, Y_train, Y_test = train_test_split(imgData, labelData, test_size=0.2, random_state=1)

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
input_shape = (img_rows, img_cols, 1)
X_train = X_train.astype('flaot32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)