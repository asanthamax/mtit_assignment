from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import neural_network as nn
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import trainme as t
import numpy as np
__author__ = 'asantha'
import matplotlib.pyplot as plt
NN = nn.NeuralNetwork()
X = []
Y = []
#here we read the inputs from text file and store into X array
f = open("data.txt").readlines()
first_line = f.pop(0)
X = [line.split() for line in f] #after store into x array we get the last element from it and store it in y array to separate output
for i in range(len(X)):
    for j in range(len(X[i])):
        if j == len(X[i])-1:
            Y.append(float(X[i][j]))
        X[i][j] = float(X[i][j])
    del X[i][-1]
#and here we convert the x array into float array and copy to array c
c = np.array((X),dtype=float)
count = len(Y)
#here we convert the y array into float array and copy to array d
d = np.zeros((count, 1), dtype=float)
for index in range(len(Y)):
    d[index] = Y[index]
print(c)
print(d)
#he we normalize the array c and d
c = c/np.amax(c, axis=0)
d = d/100

#send the arrays to train network and graph the output
T = t.TrainMe(NN)
T.trainnetwork(c, d)


z=[]
xvals = []
yvals = []
print(len(NN.output))
print(NN.output[0])
for i in range(len(T.J)):
    xvals.append(T.iteration[i])
    yvals.append(T.J[i][0])
    z.append(NN.output[i][0][0])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(xvals, yvals, z)
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_zlabel('Score')


plt.title("Marks Analysis")
plt.show()

#now after we train our network we will test with a test inputs
R = np.array(([2, 3, 1]), dtype=float)
output = NN.forward(R)
print(output)




