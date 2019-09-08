import numpy as np
from copy import deepcopy
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

# train and evaluate the model
# this function has optimization, activation functions
# @param model, the given or pre-built model
# @param X_train, the training image set
# @param Y_train, the training label set corresponding to X_train
# @param X_test, the testing image set
# @param Y_test, the testing label set corresponding to X_test
# @return history, scores of training model
# @return evaluation, scores of final model
# @return prediction, predictions of final model with test set
def train_evaluate(model, X_train, Y_train, X_test, Y_test, validSplit):
    # Stochastic gradient descent with 0.001 learning rate
    sgd = optimizers.SGD(lr=0.001)

    # Categorical cross-entropy loss function
    model.compile(
        optimizer=sgd,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    # fit our training model with 512 batch size and 500 epochs
    # labelData is structured as a "one-hot" vector
    history = None
    if(validSplit):
        print("here splited")
        history = model.fit(
            X_train,
            to_categorical(Y_train, num_classes=10),
            epochs=500,
            batch_size=512,
            validation_split=0.25
        )
    else:
        print("not splited")
        history = model.fit(
            X_train,
            to_categorical(Y_train, num_classes=10),
            epochs=500,
            batch_size=512
        )
    # evaluate our final model with Test set that we split
    evaluation = model.evaluate(
        X_test,
        to_categorical(Y_test, num_classes=10),
        batch_size=512
    )
    # predict our final model with Test set that we split
    prediction = model.predict(
        X_test,
        batch_size=512
    )
    return history, evaluation, prediction    

"""
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
"""
# print the confusion matrix
# @param prediction, prediction from model.predict
def print_confusion_matrix(prediction):
    confusion = confusion_matrix(Y_test, np.argmax(prediction, axis=1))
    print(confusion)

# k-fold cross validation
# number of folds is already defined as 3 inside the function
# @param model, prebuilt model to used in cross validation
# @param imgData, dataset of Mnist
# @param labelData, dataset of labels corresponding to imgData
# @return accuracies, scores for each iteration of evaluation
def cross_validation(model, imgData, labelData):
    n_fold = 3 # num of folds
    # here we use stratified k-fold for randomness when folding the dataset
    folds = StratifiedKFold(n_splits=n_fold, random_state=1, shuffle=True)
    # the given build model is copy for iterations 
    tempModel = deepcopy(model)
    accuracies = list()
    evaluation = None
    # for each iteration, use k-fold and evaluate the model
    for train_index, test_index in folds.split(imgData, labelData):
        if(evaluation!=None):
            model = None    # model is discarded after each iteration
            model = deepcopy(tempModel) # model is reinitiated with the given model
            evaluation = None
        # split into dataset into corresponing folds
        X_train, X_test = imgData[train_index], imgData[test_index]
        Y_train, Y_test = labelData[train_index], labelData[test_index]
        # evaluate the model and receive the score
        evaluation = train_evaluate(model, X_train, Y_train, X_test, Y_test, False)[1]
        # each score is stored in a array
        accuracies.append(evaluation[1])
    return accuracies


model = create_model(10, False, 0)
history, evaluation, prediction = train_evaluate(model, X_train, Y_train, X_test, Y_test, True)
print(evaluation)
print_confusion_matrix(prediction)
# plot the accuracy of training set and validation set over epochs
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# scores = cross_validation(model, imgData, labelData)
# print(scores)

