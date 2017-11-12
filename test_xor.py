from neural_network import Neural_Network, train
import numpy as np

def test_xor():
    X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
    y = np.array(([75], [82], [93]), dtype=float)

    X = X / np.amax(X, axis=0)
    y = y / 100  # Max test score is 100

    X = np.array(([1, 1], [0, 1], [0, 0], [1,0]), dtype=float)
    y = np.array(([0], [1], [0], [1]), dtype=float)

    NN = Neural_Network()
    train(NN, X, y)

    X = np.array(([1,1]), dtype=float)
    yHat = NN.forward(X)
    print('estimate for {}: {}'.format(X,yHat))
    X = np.array(([0,1]), dtype=float)
    yHat = NN.forward(X)
    print('estimate for {}: {}'.format(X,yHat))
    X = np.array(([1,0]), dtype=float)
    yHat = NN.forward(X)
    print('estimate for {}: {}'.format(X,yHat))
    X = np.array(([0,0]), dtype=float)
    yHat = NN.forward(X)
    print('estimate for {}: {}'.format(X,yHat))
