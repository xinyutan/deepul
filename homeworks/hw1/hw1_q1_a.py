"""
Implementations of functions for q1_a in hw1_notebook.ipynb
"""
import tensorflow as tf
import numpy as np

hp = {"learning_rate": 1, "batch_size": 128, "epochs": 20}

class Histogram(tf.Module):
  def __init__(self, d):
    self.d = d
    self.logits = tf.Variable(np.zeros(d), dtype=tf.float32)
  
  def loss(self, x):
    logits = tf.repeat(self.logits[tf.newaxis, :], x.shape[0], axis=0)
    return tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=x, logits=logits))
  
  def distribution(self):
    return tf.nn.softmax(self.logits)

def train_step(model, x):
  with tf.GradientTape() as t:
    loss = model.loss(x)
  grad = t.gradient(loss, model.logits)
  model.logits.assign_sub(hp["learning_rate"] * grad)
  return loss.numpy()


def train(model, data):
  dataset = tf.data.Dataset.from_tensor_slices(data).shuffle().batch(hp["batch_size"])
  losses = []
  for x in dataset:
    l = train_step(model, x)
    losses.append(l)
  return losses


def eval_loss(model, data):
  dataset = tf.data.Dataset.from_tensor_slices(data).batch(hp["batch_size"])
  total_loss = 0.0
  for x in dataset:
    total_loss += model.loss(x).numpy() * len(x)
  
  return total_loss / len(data)


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
    
  model = Histogram(d)
  train_losses, test_losses = [], []

  test_losses.append(eval_loss(model, test_data))
  for _ in range(hp["epochs"]):
    train_losses.extend(train(model, train_data))
    test_losses.append(eval_loss(model, test_data))

  return train_losses, test_losses, model.distribution().numpy()

