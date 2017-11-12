from neural_network import Neural_Network, train
import numpy as np


def character(glyph):
    return list(map(lambda c: 1 if c == '#' else 0, glyph))

def alphabet():
    a = character(
        '.#####.' +
        '#.....#' +
        '#.....#' +
        '#######' +
        '#.....#' +
        '#.....#' +
        '#.....#'
    )

    b = character(
        '######.' +
        '#.....#' +
        '#.....#' +
        '######.' +
        '#.....#' +
        '#.....#' +
        '######.'
    )

    c = character(
        '#######' +
        '#......' +
        '#......' +
        '#......' +
        '#......' +
        '#......' +
        '#######'
    )

    altered_c = character(
      '#######' +
      '#......' +
      '#......' +
      '#......' +
      '#......' +
      '###....' +
      '#######'
    )
    return (a, [0.1]), (b, [0.3]), (c, [0.5]), altered_c

def test_train_ocr():
    X1 = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
    y1 = np.array(([75], [82], [93]), dtype=float)

    a, b ,c, c_to_recognized = alphabet()
    inputLayerSize = len(a[0])
    hiddenLayerSize = 3 * inputLayerSize
    outputLayerSize = 1

    NN = Neural_Network(inputLayerSize=inputLayerSize, hiddenLayerSize=hiddenLayerSize, outputLayerSize=outputLayerSize)
    X = np.array((a[0], b[0], c[0]), dtype=float)
    y = np.array((a[1], b[1], c[1]), dtype=float)

    train(NN, X, y)

    X = np.array((c[0]), dtype=float)
    yHat = NN.forward(X)
    print('estimate for good C: {}'.format( yHat))
    X = np.array((c_to_recognized), dtype=float)
    yHat = NN.forward(X)
    print('estimate for bad C: {}'.format( yHat))
