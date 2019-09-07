import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt

# loading data from images.npy and labels.npy
imgData = np.load('images.npy')
labelData = np.load('labels.npy')

# flatten the data matrix into a vector
imgData = imgData.reshape((-1, 784))

# split into 20% test set, 20% validation set, and 60% training set
X_train, X_test, Y_train, Y_test = train_test_split(imgData, labelData, test_size=0.2, random_state=1)


# build the model
# @param hiddenLayer int, number of hidden layers
# @param dropOut boolean, True for dropOut
# @param dropSize int, number of size to drop out on the first layer
# example from task2:
# 1 input layer with 784 inputs
# 10 hidden layers with 50 nodes each
# 1 output layer with 1 node
# New model with dropout 20%
def create_model(hiddenLayer, dropOut, dropSize):
    # instantiate a new model
    model = Sequential()
    if(dropOut): # if dropOut, add a drop-out to input layer
        model.add(Dropout(dropSize, input_shape=(784,), seed=1))
    else: # else, just add an input layer
        model.add(Dense(50, activation='relu', input_shape=(784,)))
        hiddenLayer = hiddenLayer - 1
    
    # depending on the numbers of given hidden layers, this loop will create hidden layers
    while(hiddenLayer > 0):
        model.add(Dense(50, activation='relu'))
        hiddenLayer = hiddenLayer - 1

    # add a output layer
    model.add(Dense(10, activation='softmax'))
    
    #return this model
    return model

# Stochastic gradient descent with 0.001 learning rate
sgd = optimizers.SGD(lr=0.001)

# Categorical cross-entropy loss function
model = create_model(10, True, 0.2)
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
prediction = model.predict(
    X_test,
    batch_size=512
)

confusion = confusion_matrix(Y_test, np.argmax(prediction, axis=1))
print(confusion)


def cross_validation(model, imgData, labelData):
    n_fold = 3
    folds = StratifiedKFold(labelData, n_folds=n_fold, shuffle=True)

    for i, (train,test) in enumerate(skf):
        model = None
        




# plot the accuracy of training set and validation set over epochs
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()