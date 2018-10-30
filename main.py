from dnn_app_utils import *
import h5py
import json
import numpy as np
import os.path

np.random.seed(1)

class VoteClassifier:


  def __init__(self, mep_id, iterations, number_of_layers):
    self.K = 3                                # number of classes
    self.mep_id = mep_id                      # parliamentarian id
    self.iterations = iterations              # training iterations
    self.number_of_layers = number_of_layers  # number of network hidden layers
    self.train_x, self.train_y, self.test_x, self.test_y = load_data(mep_id) # labeled samples
    self.m = self.train_x.shape[0]            # number of training samples
    self.test_amount = self.test_x.shape[0]   # number of testing samples
    self.parameters = {}                      # network paramters
    self.is_trained = False


  def show_data_info(self):
    print ('Number of training examples: ' + str(self.m))
    print ('Number of testing examples: ' + str(self.test_amount))
    print ('train_x_orig shape: ' + str(self.train_x.shape))
    print ('train_y shape: ' + str(self.train_y.shape))
    print ('test_x_orig shape: ' + str(self.test_x.shape))
    print ('test_y shape: ' + str(self.test_y.shape))
    return self


  def flattern_x(self):
    self.train_x = self.train_x.reshape(self.m, -1).T
    self.test_x = self.test_x.reshape(self.test_amount, -1).T
    return self


  def L_layer_model(self, learning_rate=0.03):

    self.load_model()
    if self.is_trained:
      return self

    nx = np.shape(self.train_x)[0]

    # hidden layers
    layers_dims = []
    for i in range(0, self.number_of_layers):
      layers_dims.append(nx)

    # output layer
    layers_dims.append(1)

    for k in range(0, self.K):
      costs = []
      parameters = initialize_parameters_deep(layers_dims)
      for i in range(0, self.iterations):

        AL, caches = L_model_forward(self.train_x, parameters)

        cost = compute_cost(AL, self.train_y[k])

        grads = L_model_backward(AL, self.train_y[k], caches)

        self.parameters[k] = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
          costs.append(cost)
          print ('iteration %i: %s' % (i, cost))

    self.is_trained = True

    return self


  def predict_standard(self):
    return predict(self.test_x, self.test_y, self.parameters, self.K)


  def get_model_filename(self):
    return 'models/' + str(self.mep_id) + '-' \
      + str(self.number_of_layers) + '-' \
      + str(self.iterations) + '.h5'


  def save_model(self):
    filename = self.get_model_filename()
    nx = np.shape(self.train_x)[0]
    if not os.path.isfile(filename):
      f = h5py.File(filename, 'w')
      f.create_dataset('layers', data=(nx*3))
      for k in range(0, self.K):
        for key, value in self.parameters[k].items():
            f.create_dataset(key + '-' + str(k), data=value)
    return self


  def load_model(self):
    filename = self.get_model_filename()
    if not os.path.isfile(filename):
      return self

    self.parameters = {}

    f = h5py.File(filename, 'r')
    for k in range(0, self.K):
      self.parameters[k] = {}
      for i in range(1, self.number_of_layers + 1):
        self.parameters[k]['W'+str(i)] = np.array(f['W'+str(i)+'-'+str(k)])
        self.parameters[k]['b'+str(i)] = np.array(f['b'+str(i)+'-'+str(k)])

    self.is_trained = True
    return self


def test_hyper_parameters():
  # train vote prediction models for meps
  vote_count = 33

  iterations = 200;
  while iterations < 400:
    number_of_layers = 1
    while number_of_layers <= 20:
      predictions = []
      with open('./raw/ids.json') as data_file:
        meps = json.load(data_file)['mepIds'] # array of parliamentarian objects
        for i in range(0, len(meps)):
          mep = meps[i]
          p = VoteClassifier(mep['mepId'], iterations, number_of_layers) \
            .flattern_x() \
            .L_layer_model(learning_rate=0.0075) \
            .save_model() \
            .predict_standard()

          predictions.append(p)
          if vote_count != mep['votes']:
            print ('accuracy', np.average(predictions), vote_count)
            vote_count = mep['votes']
          if mep['votes'] == 31:
            break
      number_of_layers += 1
    iterations += 200

def test_learning_rate():
  with open('./raw/ids.json') as data_file:
    meps = json.load(data_file)['mepIds']
    mep = meps[0]
    alpha = 0.01
    while alpha < 0.3:
      print('alpha ', alpha)
      p = VoteClassifier(mep['mepId'], 8000, 13) \
        .flattern_x() \
        .L_layer_model(learning_rate=alpha) \
        .save_model() \
        .predict_standard()
      alpha += 0.01


if __name__ == '__main__':
  test_learning_rate()
