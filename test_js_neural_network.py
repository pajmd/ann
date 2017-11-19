import numpy as np
from js_neural_network import sigmoid, js_neural_network

def test_dot():
    i = np.array([0.1,0.2], dtype=float)
    m = np.array([[1,2,3],[4,5,6]], dtype=float)
    r = np.dot(i,m)
    # np.testing.assert_array_equal(r, np.array([0.9, 1.2, 1.5]))
    np.testing.assert_allclose(r, np.array([0.9, 1.2, 1.5]))
    sr = sigmoid(r)
    np.testing.assert_allclose(sr, np.array([sigmoid(0.9), sigmoid(1.2), sigmoid(1.5)]))
    pass


def test_dot_list_inputs():
    i = np.array([[1, 2], [2,4]], dtype=float)
    m = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    r = np.dot(i, m)
    np.testing.assert_array_equal(r, np.array([[9, 12, 15],[18, 24, 30]]))


def test_difference_output_result():
    output = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    res = np.array([[0.1, 0.2, 0.3], [0.4,0.5,0.6]])
    error = output - res
    np.testing.assert_allclose(error, np.array([[0.9, 1.8, 2.7], [3.6, 4.5, 5.4]]))


def test_multiply_element_wize():
    m1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    m2 = np.array([[0.1, 0.2, 0.3], [0.4,0.5,0.6]])
    m = m1 * m2
    np.testing.assert_allclose(m, np.array([[0.1, 0.4, 0.9], [1.6, 2.5, 3.6]]))


def test_apply_func():
    def func(x):
        return x + 1
    m1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    m2 = func(m1)
    np.testing.assert_allclose(m2, np.array([[2, 3, 4], [5, 6, 7]], dtype=float))


def test_transpose_multiply_by_scalar():
    m1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    m2 = m1.transpose()
    np.testing.assert_allclose(m2, np.array([[1,4],[2,5],[3,6]], dtype=float))
    m = m2 * 2
    np.testing.assert_allclose(m, np.array([[2,8],[4,10],[6,12]], dtype=float))


def test_add_delta():
    m1 = np.array([[2,8],[4,10],[6,12]], dtype=float)
    m2 = np.array([[0.1,0.8],[0.4,1],[0.6,1.2]], dtype=float)
    m1 = m1 + m2
    np.testing.assert_allclose(m1, np.array([[2.1,8.8],[4.4,11],[6.6,13.2]], dtype=float))


def test_normalize():
    nn = js_neural_network()
    # data = [{imput: [], output: []}]
    data = [
        { 'input': [1, 1], 'output': [0] },
        { 'input': [1, 0], 'output': [1] },
        { 'input': [0, 1], 'output': [1] },
        { 'input': [0, 0], 'output': [0] }
    ]
    examples = nn.normalize(data)
    assert len(examples) == 2
    assert examples['input'] is not None
    assert examples['output'] is not None
    np.testing.assert_array_equal(examples['input'], np.array([[1, 1], [1, 0], [0, 1], [0, 0]]))
    np.testing.assert_array_equal(examples['output'], np.array([[0], [1], [1], [0]]))


def test_setup_2_3_1():
    nn = js_neural_network()
    # data = [{imput: [], output: []}]
    data = [
        { 'input': [1, 1], 'output': [0] },
        { 'input': [1, 0], 'output': [1] },
        { 'input': [0, 1], 'output': [1] },
        { 'input': [0, 0], 'output': [0] }
    ]
    examples = nn.normalize(data)
    nn.setup(examples=examples, default_weight=0.1)
    assert len(nn.weights) == 2
    np.testing.assert_array_equal(nn.weights[0], np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]))
    np.testing.assert_array_equal(nn.weights[1], np.array([[0.1], [0.1], [0.1]]))
    assert nn.weights[0].shape == (2,3)
    assert nn.weights[1].shape == (3,1)


def test_setup_2_3_1_random_weight():
    nn = js_neural_network()
    # data = [{imput: [], output: []}]
    data = [
        { 'input': [1, 1], 'output': [0] },
        { 'input': [1, 0], 'output': [1] },
        { 'input': [0, 1], 'output': [1] },
        { 'input': [0, 0], 'output': [0] }
    ]
    examples = nn.normalize(data)
    nn.setup(examples=examples)
    assert len(nn.weights) == 2
    assert nn.weights[0].shape == (3,2)
    assert nn.weights[1].shape == (1,3)


def test_setup_2_3_2():
    nn = js_neural_network()
    # data = [{imput: [], output: []}]
    data = [
        { 'input': [1, 1], 'output': [0, 1] },
        { 'input': [1, 0], 'output': [1, 0] },
        { 'input': [0, 1], 'output': [1, 0] },
        { 'input': [0, 0], 'output': [0, 1] }
    ]
    examples = nn.normalize(data)
    nn.setup(examples=examples, default_weight=0.1)
    assert len(nn.weights) == 2
    np.testing.assert_array_equal(nn.weights[0], np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]))
    np.testing.assert_array_equal(nn.weights[1], np.array([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]))
    assert nn.weights[0].shape == (2,3)
    assert nn.weights[1].shape == (3,2)


def test_setup_2_3_2_random_weight():
    nn = js_neural_network()
    # data = [{imput: [], output: []}]
    data = [
        { 'input': [1, 1], 'output': [0, 1] },
        { 'input': [1, 0], 'output': [1, 0] },
        { 'input': [0, 1], 'output': [1, 0] },
        { 'input': [0, 0], 'output': [0, 1] }
    ]
    examples = nn.normalize(data)
    nn.setup(examples=examples)
    assert len(nn.weights) == 2
    assert nn.weights[0].shape == (3,2)
    assert nn.weights[1].shape == (2,3)


def test_setup_2_3_3_2():
    nn = js_neural_network(hidden_layers=2)
    # data = [{imput: [], output: []}]
    data = [
        { 'input': [1, 1], 'output': [0, 1] },
        { 'input': [1, 0], 'output': [1, 0] },
        { 'input': [0, 1], 'output': [1, 0] },
        { 'input': [0, 0], 'output': [0, 1] }
    ]
    examples = nn.normalize(data)
    nn.setup(examples=examples, default_weight=0.1)
    assert len(nn.weights) == 3
    np.testing.assert_array_equal(nn.weights[0], np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]))
    np.testing.assert_array_equal(nn.weights[1], np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]))
    np.testing.assert_array_equal(nn.weights[2], np.array([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]))
    assert nn.weights[0].shape == (2,3)
    assert nn.weights[1].shape == (3,3)
    assert nn.weights[2].shape == (3,2)


def test_setup_2_3_3_2_random_weight():
    nn = js_neural_network(hidden_layers=2)
    # data = [{imput: [], output: []}]
    data = [
        { 'input': [1, 1], 'output': [0, 1] },
        { 'input': [1, 0], 'output': [1, 0] },
        { 'input': [0, 1], 'output': [1, 0] },
        { 'input': [0, 0], 'output': [0, 1] }
    ]
    examples = nn.normalize(data)
    nn.setup(examples=examples)
    assert len(nn.weights) == 3
    assert nn.weights[0].shape == (3,2)
    assert nn.weights[1].shape == (3,3)
    assert nn.weights[2].shape == (2,3)


def test_xor_2_3_1():
    nn = js_neural_network() # (iterations=1)
    # data = [{imput: [], output: []}]
    data = [
        { 'input': [1, 1], 'output': [0] },
        { 'input': [1, 0], 'output': [1] },
        { 'input': [0, 1], 'output': [1] },
        { 'input': [0, 0], 'output': [0] }
    ]
    # examples = nn.normalize(data)
    # nn.setup(examples=examples, default_weight=0.1)
    nn.learn(data) # , default_weight=0.1)
    to_predict = [1, 0]
    res  = nn.predict(to_predict)
    print('Prediction for {} = {}'.format(to_predict, res))


def test_or_2_3_3():
    print('Testing ------- OR --------')
    nn = js_neural_network() # (iterations=1)
    # data = [{imput: [], output: []}]
    data = [
        { 'input': [1, 1, 1, 1], 'output': [1, 0, 0] },
        { 'input': [1, 1, 0, 1], 'output': [0, 1, 0] },
        { 'input': [1, 0, 0, 1], 'output': [0, 0, 1] },
        # { 'input': [0, 0], 'output': [0, 0, 0] }
    ]
    # examples = nn.normalize(data)
    # nn.setup(examples=examples, default_weight=0.1)
    nn.learn(data) # , default_weight=0.1)
    to_predict = [1, 1, 1, 1]
    res  = nn.predict(to_predict)
    print('Prediction for {} = {}'.format(to_predict, res))
    to_predict = [1, 1, 0, 1]
    res  = nn.predict(to_predict)
    print('Prediction for {} = {}'.format(to_predict, res))
    to_predict = [1, 0, 0, 1]
    res  = nn.predict(to_predict)
    print('Prediction for {} = {}'.format(to_predict, res))
    to_predict = [1, 0, 1, 1]
    res  = nn.predict(to_predict)
    print('Prediction for {} = {}'.format(to_predict, res))
