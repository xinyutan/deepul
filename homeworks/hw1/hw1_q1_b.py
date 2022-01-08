import tensorflow as tf
import numpy as np


from tensorflow import keras

def train_step(model, x, optimizer):
    with tf.GradientTape() as t:
        loss = model.loss(x)
    grad = t.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    return loss.numpy()


def train(model, data, optimizer):
    dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(len(data)).batch(hp["batch_size"])
    losses = []
    for x in dataset:
        l = train_step(model, x, optimizer)
        losses.append(l)
    return losses


def eval_loss(model, data):
    dataset = tf.data.Dataset.from_tensor_slices(data).batch(hp["batch_size"])
    total_loss = 0.0
    for x in dataset:
        total_loss += model.loss(x).numpy() * len(x)
  
    return total_loss / len(data)


class MixtureOfLogistics(tf.Module):
    def __init__(self, d, n_mix=4):
        self.d = d
        self.n_mix = n_mix
        self.logits = tf.Variable(tf.zeros(
            shape=[n_mix],  dtype=tf.float32))
        self.mu = tf.Variable(tf.random.normal(
            shape=[n_mix], dtype=tf.float32)
        self.s = tf.Variable(tf.random.normal(
            shape=[n_mix],  dtype=tf.float32))

    def log_prob(self, x):
        # shape: batch * n_mix
        x = tf.cast(tf.tile(x[:, tf.newaxis], [1, self.n_mix]), dtype=tf.float32)
        mu = self.mu[tf.newaxis, :]  # 1 * n_mix
        s = self.s[tf.newaxis, :]  # 1 * n_mix
        logits = self.logits[tf.newaxis, :]
        
        bins = tf.constant(tf.ones_like(x))
        
        inv_scale = tf.math.exp(-s)
        plus_x = inv_scale * (x + bins*0.5 - mu)
        minus_x = inv_scale * (x - bins*0.5 - mu)

        cdf_plus_x = tf.math.sigmoid(plus_x)
        cdf_minus_x = tf.math.sigmoid(minus_x)

        prob_delta = tf.clip_by_value(cdf_plus_x - cdf_minus_x, 1e-12, 10)
        prob_plus = tf.clip_by_value(
            tf.math.sigmoid(inv_scale * (bins*0.5 - mu)), 1e-12, 10)
        prob_minus = tf.clip_by_value(
            1 - tf.math.sigmoid(inv_scale * (bins*(self.d - 1.5) - mu)), 1e-12, 10)
        
        log_prob = tf.where(
            x < 0.001, 
            tf.math.log(prob_plus), 
            tf.where(
                x > self.d - 1 - 0.001, 
                tf.math.log(prob_minus),
                tf.math.log(prob_delta)))
        log_logits = tf.nn.log_softmax(logits, axis=1)
        return tf.math.reduce_logsumexp(log_prob + log_logits, axis=1)
        
        return log_prob
    
    def loss(self, x):
        return -tf.reduce_mean(self.log_prob(x))
    
    def distribution(self):
        x = tf.constant(np.arange(self.d))
        return tf.math.exp(self.log_prob(x))


hp = {"learning_rate": 1, "batch_size": 128, "epochs": 30}


def q1_b(train_data, test_data, d, dset_id):
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
  
    if dset_id == 1:
        d = 20
    elif dset_id == 2:
        d = 100
    else:
        raise Exception(f"Invalid dset_type {dset_id}.")

    model = MixtureOfLogistics(d)
    train_losses, test_losses = [], []
    optimizer = keras.optimizers.SGD(learning_rate=hp['learning_rate'])  
    test_losses.append(eval_loss(model, test_data))
    for _ in range(hp["epochs"]):
        train_losses.extend(train(model, train_data, optimizer))
        test_losses.append(eval_loss(model, test_data))
    return train_losses, test_losses, model.distribution().numpy()
