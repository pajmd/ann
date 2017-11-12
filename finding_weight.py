import time
from neural_network import Neural_Network, X, y
import numpy as np

weightsToTry = np.linspace(-5, 5, 1000)
costs = np.zeros(1000)

NN = Neural_Network()
startTime = time.clock()
for i in range(1000):
    NN.W1[0, 0] = weightsToTry[i]
    yHat = NN.forward(X)
    costs[i] = 0.5 * sum((y - yHat) ** 2)

endTime = time.clock()
print(endTime)