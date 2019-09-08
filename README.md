# Artificial Neural Network

We have built a Simple Neural Network using 1 input layer, 10 hidden layers and 1 output layer on mnist dataset.
Mnist dataset that we used here are much smaller dataset than you can acquire from the database. We also implemented convolutional version which can be found inside cnn.py

Author: Myo Min Thant, Robert Dutile

## Prerequisites

There are several things you need to install before using this source code.

* Python >3.4 and <3.7
* Keras
* Tensorflow
* matplotlib
* scikit-learn

## How to build and run

We have two source code which you can test and run.
* main.py provides a simple feed forward neural network
* cnn.py provides a convolutional neural network on the same dataset

Inside main.py
* build_train() will build the model with 20% validation, 20% test and 20% training sets. It will then train
the neural network using the SGD and relu.
* batch_size_difference() is used to test how our NN performs with different batch size. You will need to configure hyperparameters by yourself.
* ANNvsKFold() is used to compare how 3-fold cross validation performs across 1, 2, and 10 hidden layers

Choose one function from above and run main.py

To run cnn.py, simple run "python3 main.py".

You can also work inside python3 shell on terminal. Navigate to the project root folder, import the source files and you can start testing it. 


