__author__ = 'asantha'

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self):

        #initialize the layers
        self.inputLayerSize = 3
        self.hiddenLayerSize = 4
        self.outputLayerSize = 1

        #initialize the weights of neural network
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W3 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, x):
        self.z2 = np.dot(x, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        self.a3 = self.sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.W3)
        yHat = self.sigmoid(self.z4)
        return yHat

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self, z):
        return np.exp(-z)/((1+np.exp(-z))**2)

    def cost(self, x, y):
        self.yHat = self.forward(x)
        j = 0.5 * sum((y-self.yHat)**2)
        return j

    def costprime(self, x, y):
        self.yhat = self.forward(x)
        delta4 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z4))
        dJdW3 = np.dot(self.a2.T, delta4)
        delta3 = np.dot(delta4, self.W3.T) * self.sigmoidPrime(self.z3)
        dJdW2 = np.dot(x.T, delta3)
        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(x.T, delta2)
        return dJdW1, dJdW2, dJdW3

    def trainnetowrk(self, x, y, numberofiterations):
        for iteration in range(numberofiterations):
            adjustment1, adjustment2, adjustment3 = self.costprime(x, y)
            self.W1 = adjustment1
            self.W2 = adjustment2
            self.W3 = adjustment3