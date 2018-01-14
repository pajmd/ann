import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_prime(x):
    return np.exp(-x)/(1 + np.exp(-x))**2

def sample():
    return np.sqrt(-2 * np.log(np.random())) * np.cos(2 * np.PI * np.random());

class js_neural_network(object):

    def __init__(self, activator=sigmoid, learning_rate=0.7, hidden_layers=1, hidden_units=3, iterations=10000):
        self.activator = activator
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.iterations = iterations
        self.weights = []

    def learn(self, examples, default_weight=None, normalize_data=False): # examples [{imput: [], output: []}]
        if not normalize_data:
            examples = self.normalize(examples)
        if not self.weights:
            self.setup(examples, default_weight)
        for i in range(self.iterations):
            results = self.forward(examples)
            errors = self.back(examples, results)
            # self.emit('data', i, errors, results)

    def forward(self, examples):
        def sum(i, w):
            o = np.dot(i, w)
            so = self.activator(o)
            res = {
                'sum': o,
                'result': so
            }
            return res

        results = []
        # input -> hidden
        res = sum(examples['input'], self.weights[0])
        results.append(res)

        # hidden -> hidden
        for i in range(1, self.hidden_layers):
            res = sum(results[i]['result'], self.weights[i+1])
            results.append(res)

        #hidden -> output
        res = sum(results[len(results)-1]['result'], self.weights[len(self.weights)-1])
        results.append(res)
        return results

    def back(self, examples, results):
        # output -> hidden
        errors = examples['output'] - results[len(results) -1]['result']
        deltas = sigmoid_prime(results[len(results)-1]['sum']) * errors
        changes = np.dot( results[self.hidden_layers -1]['result'].transpose(), deltas) * self.learning_rate
        self.weights[len(self.weights) - 1] = self.weights[len(self.weights) - 1] + changes
        # hidden -> hidden
        for i in range(1, self.hidden_layers): # TO FIX
            deltas = np.dot(self.weights[len(self.weights) - i].trnspose(), deltas) \
                     * sigmoid_prime(results[len(results) - (i+1)]['sum'])
            changes = np.dot(deltas, results[self.hidden_layers -(i + 1)]['result'].tanspose()) * self.learning_rate
            self.weights[len(self.weights) - (i + 1)] = self.weights[len(self.weights - (i + 1))] + changes
        # hidden -> input
        deltas = np.dot(deltas , self.weights[1].transpose()) * sigmoid_prime(results[0]['sum'])
        changes = np.dot(examples['input'].transpose(), deltas) * self.learning_rate
        self.weights[0] = self.weights[0] + changes
        return errors


    def predict(self, input): # input []
        results = self.forward({'input': np.array(input)})
        return results[len(results) - 1]['result']

    def setup(self, examples, default_weight=None):
        if default_weight:
            if type(default_weight) == float:
                # input hidden weights matrix
                self.weights.append(np.array([[default_weight] * self.hidden_units] * len(examples['input'][0])))
                # hidden hidden weights matrix
                for _ in range(1, self.hidden_layers):
                    self.weights.append(np.array([[default_weight] * self.hidden_units] * self.hidden_units))
                # hidden output eights matrix
                self.weights.append(np.array([[default_weight] * len(examples['output'][0])] * self.hidden_units))
                idx = 1
                # for mtx in self.weights:
                #     for row in range(mtx.shape[0]):
                #         for col in range(mtx.shape[1]):
                #             mtx[row, col] = 0.001 * idx
                #             idx += 1
                pass
            else:
                raise ValueError('default weight must be a float')
        else:
            np.random.seed(2)
            # input hidden weights matrix
            self.weights.append(np.random.random_sample((len(examples['input'][0]), self.hidden_units)))

            # hidden hidden weights matrix
            for _ in range(1, self.hidden_layers):
                self.weights.append(np.random.random_sample((self.hidden_units, self.hidden_units)))

            # hidden output eights matrix
            self.weights.append(np.random.random_sample((self.hidden_units, len(examples['output'][0]))))

    @staticmethod
    def normalize(data):
        '''creates an input and an output matrice'''
        input_matrix = []
        output_matrix = []
        for entry in data:
            input_matrix.append(entry['input'])
            output_matrix.append(entry['output'])

        im = np.array(input_matrix)
        om = np.array(output_matrix)
        return {
            'input': im,
            'output': om
        }



