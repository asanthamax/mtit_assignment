__author__ = 'asantha'
from scipy import optimize
import numpy as np
class TrainMe:

    def __init__(self, N):

        self.N = N

    #here we define wrapper function to send the cost and gradient to minimize function
    def costFunctionWrapper(self, params, x, y):
        self.N.setParams(params)
        cost = self.N.costFunctionPrime(x, y) #call to back propagtion of cost function in our neural network
        grad = self.N.computeGradients(x, y) #compute the gradient decent of out backpropargation network
        return cost, grad

    #here is the call back function wich iteratively call and append cost value to J array to plot the graph during training the network
    def callbackF(self, params):
        self.N.setParams(params)
        self.parse = self.parse + 1
        self.J.append(self.N.costFunction(self.X, self.y))
        self.iteration.append(self.parse)

    def trainnetwork(self, x, y):
        self.X = x
        self.y = y
        self.J = []
        self.parse = 0
        self.iteration = []
        params0 = self.N.getParams()
        options = {'maxiter': 200, 'disp': True}#here we use SLSQP algorithm included in scipy optimize library
        res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='SLSQP', args=(x, y), options=options, callback=self.callbackF)

        self.N.setParams(res.x)
        self.optimizationResults = res
