"""
Implementations of functions for q1_a in hw1_notebook.ipynb
"""
import numpy as np

class Hyperparameters(object):
  def __init__(self, epochs=20, learning_rate=0.01, batch_size=100):
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    
    # update after the training
    self.train_loss = float("inf")
    self.test_loss = float("inf")]

class Dataset(object):
  pass

def get_loss(data, theta):
  pass


def train_step(batch_data, theta, hp_object):
  pass 

def q1_a(train_data, test_data, d, dset_id):
  """
  train_data: An (n_train,) numpy array of integers in {0, ..., d-1}
  test_data: An (n_test,) numpy array of integers in {0, .., d-1}
  d: The number of possible discrete values for random variable x
  dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

  Returns
  - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
  - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
  - a numpy array of size (d,) of model probabilities
  """
  
  """ YOUR CODE HERE """
  if dset_id == 1:
    d = 20
  elif dset_id == 2:
    d = 100
  else:
    raise Exception(f"Invalid dset_type {dset_id}.")
  
  theta = np.zeros(d)

  hp = Hyperparameters()
  train_loss = np.zeros(hp.epochs)
  test_loss = np.zeros(hp.epochs + 1)
  test_loss[0] = get_loss(test_data, theta)
  dataset = Dataset(train_data, shuffle=True)

  for epoch in range(hp.epochs):
    minibatch = next(dataset)
    theta = train_step(minibatch, theta, hp)
    train_loss[epoch] = get_loss(minibatch, theta)
    test_loss[epoch + 1] = get_loss(test_data, theta)
  prob_hat = np.exp(np.arange(d) * theta)
  
  return train_loss, test_loss, prob_hat / np.sum(prob_hat)
    
    

