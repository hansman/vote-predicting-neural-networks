import json
import numpy as np
from dnn_app_utils import *

np.random.seed(1)

class VoteClassifier:
    def __init__(self, mepId):
        self.train_x, self.train_y, self.test_x, self.test_y = load_data(mepId)
        self.sample_amount = self.train_x.shape[0]
        self.test_amount = self.test_x.shape[0]
        self.parameters = {}
        self.is_trained = False

    def show_data_info(self):
        print ("Number of training examples: " + str(self.sample_amount))
        print ("Number of testing examples: " + str(self.test_amount))
        print ("train_x_orig shape: " + str(self.train_x.shape))
        print ("train_y shape: " + str(self.train_y.shape))
        print ("test_x_orig shape: " + str(self.test_x.shape))
        print ("test_y shape: " + str(self.test_y.shape))
        return self

    def flattern_x(self):
        self.train_x = self.train_x.reshape(self.sample_amount, -1).T
        self.test_x = self.test_x.reshape(self.test_amount, -1).T
        return self

    def L_layer_model(self, learning_rate=0.0075, num_iterations=300):  # lr was 0.009

        np.random.seed(1)

        nx = np.shape(self.train_x)[0]

        layers_dims = [nx, 32, 32, 16, 8, 1]

        K = np.shape(self.train_y)[0]

        for k in range(0, K):
            costs = []
            parameters = initialize_parameters_deep(layers_dims)
            for i in range(0, num_iterations):

                AL, caches = L_model_forward(self.train_x, parameters)

                cost = compute_cost(AL, self.train_y[k])

                grads = L_model_backward(AL, self.train_y[k], caches)

                self.parameters[k] = update_parameters(parameters, grads, learning_rate)

                if i % 100 == 0:
                    costs.append(cost)
                    # print ("Cost after iteration %i: %s" % (i, cost))

        self.is_trained = True

        return self

    def predict_standard(self):
        return predict(self.test_x, self.test_y, self.parameters, 3)


if __name__ == '__main__':

    predictions = []
    with open('./raw/ids.json') as data_file:
        ids = json.load(data_file)
        meps = ids['mepIds']
        for i in range(0, len(meps)):
            mep = meps[i]
            if mep['votes'] == 1:
                break
            print ("mep " + str(mep['mepId']) + " votes " + str(mep['votes']))
            p = VoteClassifier(mep['mepId']).flattern_x().L_layer_model(learning_rate=0.0075, num_iterations=4000).predict_standard()
            predictions.append(p)
            print ("total average accuracy", np.average(predictions))

