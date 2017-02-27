'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
from queue import dequeue_and_enqueue

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data



# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout, max_pool_factors):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, side, side, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=max_pool_factors[0])

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=max_pool_factors[1])

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# Network Parameters
side = 280
n_input = np.square(side) # MNIST data input (img shape: 28*28)
n_classes = 1 # MNIST total classes (0-9 digits)

# tf Graph input

class Net:

  def define(self, max_pool_factors=[2, 2, 4], conv_sizes=[5, 5, 5], learning_rate=0.001):

    self.x = x = tf.placeholder(tf.float32, [None, side, side])
    self.y = y = tf.placeholder(tf.float32, [None, n_classes])
    self.keep_prob = keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

    max_pooled_area = np.square(side / max_pool_factors[0] / max_pool_factors[1])
    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([conv_sizes[0], conv_sizes[0], 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([conv_sizes[0], conv_sizes[0], 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([max_pooled_area*64, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = conv_net(x, weights, biases, keep_prob, max_pool_factors)

    # Define loss and optimizer
    self.cost = cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    self.optimizer = optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    self.accuracy = accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    self.init = init = tf.global_variables_initializer()

  def train(self, images, labels, training_iters = 200000, batch_size = 128, display_step = 10, dropout = 0.75, test=False):

    # Launch the graph
    with tf.Session() as sess:
        sess.run(self.init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            images = dequeue_and_enqueue(images, batch_size)
            batch_x = images[:batch_size]
            labels = dequeue_and_enqueue(labels, batch_size)
            batch_y = labels[:batch_size]
            # Run optimization op (backprop)
            sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y,
                                           self.keep_prob: dropout})

            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([self.cost, self.accuracy], feed_dict={self.x: batch_x,
                                                                  self.y: batch_y,
                                                                  self.keep_prob: 1.})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")

        if test:
          # Calculate accuracy for 256 mnist test images
          print("Testing Accuracy:", \
              sess.run(self.accuracy, feed_dict={self.x: mnist.test.images[:256],
                                          self.y: mnist.test.labels[:256],
                                          self.keep_prob: 1.}))

