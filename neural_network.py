__author__ = 'asantha'

import numpy as np
from decimal import Decimal

class NeuralNetwork:

    def __init__(self):

        #Define the neural network layers
        self.inputLayerSize = 3
        self.outputLayerSize = 1
        self.hiddenLayerSize = 4

        #define weights
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        try:
            res = 1/(1+np.exp(-z))
        except OverflowError:
            res = 0.0
        return res

    def sigmoidPrime(self, z):
        #Gradient of sigmoid
        try:
            res = np.exp(-z)/((1+np.exp(-z))**2)
        except OverflowError:
            res = 0.0
        return res

    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J

    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    def getParams(self):
        #get w1 and w2 unrolled into vector
        params = np.concatenate((self.W1.ravel(),self.W2.ravel()))
        return params

    def setParams(self, params):
        #set W1 and W2 using single parameter vector
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, x, y):
        dJdW1, dJdW2 = self.costFunctionPrime(x, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))



