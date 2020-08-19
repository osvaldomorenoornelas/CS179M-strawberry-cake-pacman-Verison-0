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
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import tensorflow as tf
import pickle
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
def readTrainingInputs():
    """
    Will return the Counter of Q(s,a) from qValueFile
    """
    with open('./trainingInput.pickle', 'rb') as handle:
        trainingInputs = pickle.load(handle)
    return trainingInputs


def readTrainingOutputs():
    """
    Return the list of weights from LinearApproxFile
    """
    with open('./trainingOutput.pickle', 'rb') as handle:
        trainingOutputs = pickle.load(handle)
    return trainingOutputs
class MLP(Module):
    def __init__(self, inputNumber):
        #call super constructor for Module
        super(MLP, self).__init__()

        #take the linear approximation of the layer
        #self.layer = Linear(inputNumber, 2)
        # input to first hidden layer
        self.hidden1 = Linear(inputNumber, 1)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()

        #sigmaoid of the layer data array
        #example https://pytorch.org/docs/stable/generated/torch.sigmoid.html
        #self.activation = Sigmoid()
        # third hidden layer and output
        self.hidden2 = Linear(2, 2)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()

    #function to forward propagate input
    def forward(self, prop):

        #insert into the layer and activate
        #prop = self.layer(prop)
        #prop = self.activation(prop) #could also be prop
        # input to first hidden layer
        prop = self.hidden1(prop)
        prop = self.act1(prop)

        #second hidden layer and output
        prop = self.hidden2(prop)
        prop = self.act2(prop)

        #return the layer
        return prop

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

        #self.model = self.Model()
        #define the model
        self.model = T.nn.Sequential(T.nn.Linear(self.inputs, self.hidden),
                     T.nn.ReLU(), T.nn.Linear(self.hidden, self.outputs), )

    def setModel(self, model):
        self.model = model

    """
    Load the data into x and y 
    (data sets as shown by professor shelton on office hours)
    
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)  
    
    torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    """
    def loadData(self):
        #create tensors from the data file
        #for the moment, we will create temporaries from random

        a = np.asarray(readTrainingInputs(), dtype=np.float32)
        b = np.asarray(readTrainingOutputs(), dtype=np.float32)
        

        x = T.tensor(a)
        y = T.tensor(b)

        print(x)
        print(y)

        return x, y

    """
    return the current model being used
    """
    def Model(self):
        return self.model

    """
    Loss determines how good the weights are
    MSELoss measures the mean squared error
    """
    def LossFunction(self):
        loss_fn = T.nn.MSELoss(reduction='mean')
        return loss_fn

    """
    Train model using a diff method
    using tutorial code from: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    """
    def Train(self, x, y):

        learning_rate = 0.001

        for t in range(4000):
            # Forward pass: compute predicted y by passing x to the model. Module objects
            # override the __call__ operator so you can call them like functions. When
            # doing so you pass a Tensor of input data to the Module and it produces
            # a Tensor of output data.
            y_pred = self.model(x)
            # Compute and print loss. We pass Tensors containing the predicted and true
            # values of y, and the loss function returns a Tensor containing the
            # loss.
            lossF = self.LossFunction()
            loss  = lossF(y_pred, y)
            if t % 100 == 99:
                print(t, loss.item())

            # Zero the gradients before running the backward pass.
            self.model.zero_grad()

            # Backward pass: compute gradient of the loss with respect to all the learnable
            # parameters of the model. Internally, the parameters of each Module are stored
            # in Tensors with requires_grad=True, so this call will compute gradients for
            # all learnable parameters in the model.
            loss.backward()

            # Update the weights using gradient descent. Each parameter is a Tensor, so
            # we can access its gradients like we did before.
            with T.no_grad():
                for param in self.model.parameters():
                    param -= learning_rate * param.grad

    """
    To train the model we need to optamize compute, get loss, and
    update
    """
    def TrainModel(self, trainData):

        #optamization algorithm
        criterion = self.LossFunction()

        #give SDG learning rate and momentum
        #SDG: Implements stochastic gradient descent
        optimizer = SGD(self.model.parameters(), lr = 0.01)

        #train 50 iterations
        for epoch in range(50):
            #give minibatches umertaions
            for i, (inputs, targets) in enumerate(trainData):

                #zero_grad clears old gradients from the last step
                optimizer.zero_grad()

                #get the model output
                model_output = self.model(inputs)

                #obtain loss from model and targets
                loss = criterion(model_output, targets)

                #pass data back
                loss.backward()

                #updates the parameters for gradietns (loss.backward)
                optimizer.step()

    """
    This function evaluates the model and returns accuracy
    """
    def evaluate(self, testData):

        #containers for the predictions and actual data
        predictions = list()
        actuals     = list()

        for i, (input, target) in enumerate(testData):

            #evaluate data set model
            modelEval = self.model(input)

            #detach(): constructs a new view on a tensor
            #assigns it as a numpy array
            modelEval = modelEval.detach().numpy()

            #get numpy array of targets
            actualData = target.numpy()

            #tensor with the same data and number of elements as
            #input, but with the specified shape
            #actualData = actualData.reshape((len(actualData), 1))

            #round class values
            modelEval = modelEval
            # modelEval = modelEval.round()

            predictions.append(modelEval)
            actuals.append(actualData)

        #Stack arrays in sequence vertically (row wise)
        predictions = np.vstack(predictions)
        actuals     = np.vstack(actuals)

        #get the accuracy (will improve as iterations occur)
        # accuracy = accuracy_score(actuals.round(), predictions, normalize = False)
        accuracy = accuracy_score(actuals, predictions, normalize = True)

        return accuracy

    """
    Predicts for a row of data. May have to be more specific because
    of how pacman is structured
    """
    def predict(self, features):
        #convert row to data
        features = Tensor([features])
        #create a prediction
        modelData = self.model(features)

        #get the numpy Data
        modelData = modelData.detach().numpy()
        return modelData



