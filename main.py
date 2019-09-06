import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras import optimizers
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# loading data from images.npy and labels.npy
imgData = np.load('images.npy')
labelData = np.load('labels.npy')

# flatten the data matrix into a vector
imgData = imgData.reshape((-1, 784))

# split into 20% test set, 20% validation set, and 60% training set
X_train, X_test, Y_train, Y_test = train_test_split(imgData, labelData, test_size=0.2, random_state=1)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

# build the model
# 1 input layer with 784 inputs
# 10 hidden layers with 50 nodes each
# 1 output layer with 1 node
model = Sequential([
    Dense(50, activation='relu', input_shape=(784,)), # hidden layer 1 with input 
    Dense(50, activation='relu'), # hidden layer 2
    Dense(50, activation='relu'), # hidden layer 3
    Dense(50, activation='relu'), # hidden layer 4
    Dense(50, activation='relu'), # hidden layer 5
    Dense(50, activation='relu'), # hidden layer 6
    Dense(50, activation='relu'), # hidden layer 7
    Dense(50, activation='relu'), # hidden layer 8
    Dense(50, activation='relu'), # hidden layer 9
    Dense(50, activation='relu'), # hidden layer 10
    Dense(10, activation='softmax'), # output layer <== still not sure about the softmax function
])

sgd = optimizers.SGD(lr=0.001)

# Stochastic gradient descent with 0.001 learning rate
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
    to_categorical(Y_train, num_classes=10),
    epochs=500,
    batch_size=512,
    validation_split=0.25
)

# evaluate our final model with Test set that we split
# model.evaluate(
#     X_test,
#     to_categorical(Y_test, num_classes=10),
#     batch_size=512
# )
model.predict(
    X_test,
    batch_size=512
)

# plot the accuracy of training set and validation set over epochs
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()