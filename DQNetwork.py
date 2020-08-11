import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.nn import Module
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
import numpy as np
from torch.optim import SGD
from sklearn.metrics import accuracy_score

"""
Code based on the tutorial by:
https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
"""
"""
layer perceptron
define the model that will be used
The class implementation is for a simple layer, not for a multilayer
Derives from Module class in torch
"""
class MLP(Module):
    def __init__(self, inputNumber):
        #call super constructor for Module
        super(MLP, self).__init__()

        #take the linear approximation of the layer
        self.layer = Linear(inputNumber, 1)

        #sigmaoid of the layer data array
        #example https://pytorch.org/docs/stable/generated/torch.sigmoid.html
        self.activation = Sigmoid()

    #function to forward propagate input
    def forward(self, prop):

        #insert into the layer and activate
        layer = self.layer(prop)
        layer = self.activation(layer) #could also be prop

        #return the layer
        return layer

"""
DQ Network has its necessary functions for learning. 
It will interact with MLP and agent in order to make them work
"""
class DQNetwork(object):

    def __init__(self, gamma, epsilon, n_actions, input_dims,
                 out_dims, batch_size, hidden_dimension):

        #discount factor
        self.gamma = gamma

        #probability
        self.epsilon = epsilon
        #number of actions
        self.actions = n_actions
        #input size
        self.inputs = input_dims
        # input size
        self.outputs = out_dims

        #data batch input size
        self.batch = batch_size

        #hidden dimension
        self.hidden = hidden_dimension

    """
    Load the data into x and y 
    (data sets as shown by professor shelton on office hours)
    """
    def loadData(self):
        #create tensors from the data file
        #for the moment, we will create temporaries from random
        x =  T.randn(self.actions, self.inputs)
        y =  T.randn(self.actions, self.outputs)
        return x,y

    """
    alternate way to create the model without defining an MLP
    """
    def Model(self):
        model = T.nn.Sequential(T.nn.Linear(self.inputs, self.hidden),
                                T.nn.ReLU(),
                                T.nn.Linear(self.hidden, self.outputs),)
        return model

    """
    Loss determines how good the weights are
    MSELoss measures the mean squared error
    """
    def LossFunction(self):
        loss_fn = T.nn.MSELoss()
        return loss_fn

    """
    To train the model we need to optamize compute, get loss, and
    update
    """
    def TrainModel(self, trainer, model):

        #optamization algorithm
        criterion = self.LossFunction()

        #give SDG learning rate and momentum
        #SDG: Implements stochastic gradient descent
        optimizer = SGD(model.parameters(), lr = 0.01)

        #train 50 iterations
        for epoch in range(50):
            #give minibatches umertaions
            for i, (inputs, targets) in enumerate(trainer):

                #zero_grad clears old gradients from the last step
                optimizer.zero_grad()

                #get the model output
                model_output = model(inputs)

                #obtain loss from model and targets
                loss = criterion(model_output, targets)

                #pass data back
                loss.backward()

                #updates the parameters for gradietns (loss.backward)
                optimizer.step()

    """
    This function evaluates the model and returns accuracy
    """
    def evaluate(self, testData, model):

        #containers for the predictions and actual data
        predictions = list()
        actuals     = list()

        for i, (input, target) in enumerate(testData):

            #evaluate data set model
            modelEval = model(input)

            #detach(): constructs a new view on a tensor
            #assigns it as a numpy array
            modelEval = modelEval.detach().numpy()

            #get numpy array of targets
            actualData = target.numpy()

            #tensor with the same data and number of elements as
            #input, but with the specified shape
            actualData = actualData.reshape((len(actualData), 1))

            #round class values
            modelEval = modelEval.round()

            predictions.append(modelEval)
            actuals.append(actualData)

            #Stack arrays in sequence vertically (row wise)
            predictions = np.vstack(predictions)
            actuals     = np.vstack(actuals)

            #get the accuracy (will improve as iterations occur)
            accuracy = accuracy_score(actuals, predictions)

        return accuracy

    """
    Predicts for a row of data. May have to be more specific because
    of how pacman is structured
    """
    def predict(self, rowData, model):
        #convert row to data
        rowData = Tensor([rowData])

        #create a prediction
        modelData = model(rowData)

        #get the numpy Data
        modelData = modelData.detach().numpy()

        return modelData



