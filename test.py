import sklearn

__author__ = 'asantha'
import numpy as np
from scipy import optimize
import  matplotlib.pyplot as plt
class NeuralNetwork:
    def __init__(self):

        #initialize the layers
        self.inputLayerSize = 2
        self.hiddenLayerSize = 3
        self.outPutLayerSize = 1

        #initialize the weights
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outPutLayerSize)

    def forward(self, x):
        self.z2 = np.dot(x, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.z2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self, z):
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costFuntion(self,x,y):
        self.yHat = self.forward(x)
        j = 0.5*sum((y-self.yHat)**2)
        return j

np.random.seed(0)
x, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(x[:, 0], x[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
plt.show()