import tensorflow as tf
import matplotlib.pyplot as plt

# Simple RBM class used for the Tensorflow version of RBM
# Designed to be modular, should attach the weights as well, will be helpful when dealing with DBM
class RBM:
    def __init__(self, n_visible, n_hidden, lr, epochs, batch_size=None):
        '''
        Initialize a model for an RBM with one layer of hidden units
        :param n_visible: Number of visible nodes
        :param n_hidden: Number of hidden nodes
        :param lr: Learning rate for the CD algorithm
        :param epochs: Number of iterations to run the algorithm for
        :param batch_size: Split the training data
        '''
        self.n_hidden = n_hidden
        self.n_visible = n_visible
        self.lr = lr

        self.epochs = epochs
        self.batch_size = batch_size


def get_probabilities(layer, weights, val, bias):
    '''
    Find the probabilities associated with layer specified
    :param layer: Hidden layer or visible layer, specified as string
    :param weights: Tensorflow placeholder for weight matrix
    :param val: Input units, hidden or visible as binary or float
    :param bias: Bias associated with the computation, opposite of the input
    :return: A tensor of probabilities associated with the layer specified
    '''
    if layer == 'hidden':
        with tf.name_scope("Hidden_Probabilities"):
            return tf.nn.sigmoid(tf.matmul(val, weights) + bias)
    elif layer == 'visible':
        with tf.name_scope("Visible_Probabilities"):
            return tf.nn.sigmoid(tf.matmul(val, tf.transpose(weights)) + bias)


def get_linear_probabilities(layer, weights, val, bias):
    '''
    Find the probabilities associated with layer specified
    :param layer: Hidden layer or visible layer, specified as string
    :param weights: Tensorflow placeholder for weight matrix
    :param val: Input units, hidden or visible as binary or float
    :param bias: Bias associated with the computation, opposite of the input
    :return: A tensor of probabilities associated with the layer specified
    '''
    if layer == 'hidden':
        with tf.name_scope("Hidden_Probabilities"):
            return tf.matmul(val, weights) + bias
    elif layer == 'visible':
        with tf.name_scope("Visible_Probabilities"):
            return tf.nn.sigmoid(tf.matmul(val, tf.transpose(weights)) + bias)


def gibbs(steps, v, hb, vb, W):
    '''
    Use the Gibbs sampler for a network of hidden and visible units.
    :param steps: Number of steps to run the algorithm
    :param v: Input data
    :param hb: Hidden Bias
    :param vb: Visible bias
    :param W: Weight matrix
    :return: Returns a sampled version of the input
    '''
    with tf.name_scope("Gibbs_sampling"):
        for i in range(steps):
            hidden_p = get_probabilities('hidden', W, v, hb)
            h = sample(hidden_p)

            visible_p = get_probabilities('visible', W, h, vb)
            v = sample(visible_p)
        return v


def gibbs_linear(steps, v, hb, vb, W):
    '''
    Use the Gibbs sampler for a network of hidden and visible units.
    :param steps: Number of steps to run the algorithm
    :param v: Input data
    :param hb: Hidden Bias
    :param vb: Visible bias
    :param W: Weight matrix
    :return: Returns a sampled version of the input
    '''
    with tf.name_scope("Gibbs_sampling"):
        for i in range(steps):
            hidden_p = get_linear_probabilities('hidden', W, v, hb)
            poshidstates = sample_linear(hidden_p)

            visible_p = get_linear_probabilities('visible', W, poshidstates, vb)
            v = sample_linear(visible_p)
        return v

def sample(probabilities):
    '''
    Sample a tensor based on the probabilities
    :param probabilities: A tensor of probabilities given by 'rbm.get_probabilities'
    :return: A sampled sampled tensor
    '''
    return tf.floor(probabilities + tf.random_uniform(tf.shape(probabilities), 0, 1))

def sample_linear(probabilities):
    '''
        Create a tensor based on the probabilities by adding gaussian noise from the input
        :param probabilities: A tensor of probabilities given by 'rbm.get_probabilities'
        :return: The addition of noise to the original probabilities
        '''
    return tf.add(probabilities, tf.random_uniform(tf.shape(probabilities)))


def plot_weight_update(x=None, y=None):
    plt.xlabel("Epochs")
    plt.ylabel("CD difference")
    plt.title('Weight increment change throughout learning')
    plt.plot(x, y, 'r--')
    plt.show()

