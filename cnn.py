import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils, to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

img_rows, img_cols = 28, 28
nb_pool = 2
nb_conv = 3

# loading data from images.npy and labels.npy
imgData = np.load('images.npy')
labelData = np.load('labels.npy')

# split into 20% test set, 20% validation set, and 60% training set
X_train, X_test, Y_train, Y_test = train_test_split(imgData, labelData, test_size=0.2, random_state=1)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation='softmax'))

    return model

def train_evaluate(model, X_train, Y_train, X_test, Y_test):
    # Stochastic gradient descent with 0.001 learning rate
    sgd = optimizers.SGD(lr=0.01)
    ada = optimizers.Adadelta()
    # Categorical cross-entropy loss function
    model.compile(
        optimizer=sgd,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # fit our training model with 512 batch size and 500 epochs
    # labelData is structured as a "one-hot" vector
    history = model.fit(
        X_train,
        Y_train,
        epochs=15,
        batch_size=256,
        validation_split=0.1
    )

    # evaluate our final model with Test set that we split
    evaluation = model.evaluate(
        X_test,
        Y_test
    )
    # predict our final model with Test set that we split
    prediction = model.predict(
        X_test,
        batch_size=256
    )
    return history, evaluation, prediction

model = create_model()
history, evaluation, prediction = train_evaluate(model, X_train, Y_train, X_test, Y_test)
print(evaluation)
# plot the accuracy of training set and validation set over epochs
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()