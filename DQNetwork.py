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

https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/

https://www.youtube.com/watch?v=wc-FxNENg9U
"""

"""
This module reads the training data from a pickle file for the inputs
"""
def readTrainingInputs():
    """
    Will return the Counter of Q(s,a) from qValueFile
    """
    with open('./trainingInput.pickle', 'rb') as handle:
        trainingInputs = pickle.load(handle)
    return trainingInputs

"""
This module reads the training data from a pickle file for the outputs
"""
def readTrainingOutputs():
    """
    Return the list of weights from LinearApproxFile
    """
    with open('./trainingOutput.pickle', 'rb') as handle:
        trainingOutputs = pickle.load(handle)
    return trainingOutputs

"""
This module reads the training data from a pickle file for the inputs from an alternative file
"""
def readDQInputs():
    """
    Will return the Counter of Q(s,a) from qValueFile
    """
    with open('./DQInput.pickle', 'rb') as handle:
        trainingInputs = pickle.load(handle)
    return trainingInputs

"""
This module reads the training data from a pickle file for the outputs from an alternative file
"""
def readDQOutputs():
    """
    Return the list of weights from LinearApproxFile
    """
    with open('./DQOutput.pickle', 'rb') as handle:
        trainingOutputs = pickle.load(handle)
    return trainingOutputs

"""
This is the DQNetwok class. It gets the data, declares the model , trains, and make predictions
DQ Network has its necessary functions for learning.

Code based on the tutorial in:
https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/

https://www.youtube.com/watch?v=wc-FxNENg9U
"""
class DQNetwork(object):

    def __init__(self, input_dims, out_dims, hidden_dimension):

        #input size
        self.inputs = input_dims
        # input size
        self.outputs = out_dims

        #hidden dimension
        self.hidden = hidden_dimension

        #define the sequential model
        self.model = T.nn.Sequential(T.nn.Linear(self.inputs, self.hidden),
                     T.nn.ReLU(), T.nn.Linear(self.hidden, self.outputs), )
    
    """
    This function sets the model in case there is another trained model saved or ready to use
    """
    def setModel(self, model):
        self.model = model

    """
    Load the data into x and y 
    """
    def loadData(self):
        #create numpy arrays from the data file

        a = np.asarray(readTrainingInputs(), dtype=np.float32)
        b = np.asarray(readTrainingOutputs(), dtype=np.float32)

        #convert arrays to tensors
        x = T.tensor(a)
        y = T.tensor(b)

        #return tensors
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
    Train model using the nn module
    using example code from: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    """
    def Train(self, x, y):
        #declaring the learning rate
        learning_rate = 0.001

        #iterate 10000 in order to achieve a plateau in the loss function
        for t in range(10000):
            # Forward pass: compute predicted y by passing x to the model. When
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
            # parameters of the model. 
            loss.backward()

            # Update the weights using gradient descent. Each parameter is a Tensor, so
            # we can access its gradients like we did before.
            with T.no_grad():
                for param in self.model.parameters():
                    param -= learning_rate * param.grad

    """
    Predicts for a row of data. Pass in the set of features to the model
    The model will return a tensor with the output corresponding to the
    set of inputs.

    Based on examples from
    https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/

    https://www.youtube.com/watch?v=wc-FxNENg9U
    """
    def predict(self, features):
        #convert row to data
        features = Tensor([features])
        #create a prediction
        modelData = self.model(features)

        #get the numpy Data
        modelData = modelData.detach().numpy()

        #return the output back to best choice
        return modelData



