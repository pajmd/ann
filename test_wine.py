import pandas as pd
import gen_neural_network
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import deque


# https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/

def _test_y_to_list():
    for i in range(1,4):
        cultivators = deque([1, 0, 0])
        cultivators.rotate(i-1)
        print('{} {}'.format(i, cultivators))

    # for i in [1,1,3,2,3,3,2]:
    #     cultivators = deque([1, 0, 0])
    #     cultivators.rotate(-(i-1))
    #     print(cultivators)


def get_data():
    def transform_y(y_set):
        res = []
        for y in y_set:
            cultivator = deque([1, 0, 0])
            cultivator.rotate(y - 1)
            res.append(list(cultivator))
        return res

    wine = pd.read_csv('/home/pjmd/python_workspace/PycharmProjects/WineProjectNotebook/wine_data.csv',
                       names=["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium",
                              "Total_phenols", "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins",
                              "Color_intensity", "Hue", "OD280", "Proline"])
    X = wine.drop('Cultivator',axis=1)
    y = wine['Cultivator']
    # use random_state for repeatability
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
    # scale the data, better for NN processing
    scaler = StandardScaler()
    scaler.fit(X_train)
    StandardScaler(copy=True, with_mean=True, with_std=True)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = transform_y(y_train)
    y_test = transform_y(y_test)
    return X_train, y_train, X_test, y_test
#    return X_train, [[y] for y in y_train], X_test, [[y] for y in y_test]

def test_wine():
    X_train, y_train, X_test, y_test = get_data()
    nn = gen_neural_network.js_neural_network(iterations=5000,
                                              hidden_units=13,
                                              learning_rate=0.001,
                                              hidden_layers=1)
    wine_data = {
        'input': np.array(X_train),
        'output': np.array(y_train)
    }
    nn.learn(wine_data, default_weight=None, normalize_data=True)
    y_test = np.array(y_test)
    count = 0
    for i, x in enumerate(np.array(X_test)):
        prediction = nn.predict(x)
        if all(prediction == y_test[i]):
            count+=1
        else:
            print('x: {} prediction: {} expected: {}'.format('x', prediction, y_test[i]))
    print('correct {} out of {}'.format(count, i))