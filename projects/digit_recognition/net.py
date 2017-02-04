from __future__ import print_function
import tensorflow as tf
from preprocess import Preprocess
import numpy as np

side_length = 28 * 5 * 2
n_input = np.square(side_length)
# 1 length, 4 bbox, 5 * 11 one hot encoded
n_classes = 60

preprocess = Preprocess();

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=2):
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
    x = tf.reshape(x, shape=[-1, side_length, side_length, 1])
    print(x)
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=max_pool_factors[0])

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=max_pool_factors[1])

    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=max_pool_factors[2])

    # Fully connected layer
    # Reshape conv3 output to fit fully connected layer input
    print(conv3.get_shape().as_list(), weights['wd1'].get_shape().as_list())
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

class Net:

  def execute(self, data):
    # get image set you want
    self.define()
    self.train(np.array(data['sequences']), np.array(data['labels']))


  def define(self, max_pool_factors=[2, 2, 4], conv_sizes=[25, 10, 5], learning_rate = 0.001):
    self.graph = graph = tf.Graph()
    with graph.as_default():
      # tf Graph input
      self.x = x = tf.placeholder(tf.float32, [None, side_length, side_length])
      self.y = y = tf.placeholder(tf.float32, [None, n_classes])
      self.keep_prob = keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

      # Store layers weight & bias
      max_pooled_area = 147456 # np.square(side_length / max_pool_factors[0] / max_pool_factors[1] / max_pool_factors[2])
      weights = {
          # conv size, inputs, outputs
          'wc1': tf.Variable(tf.random_normal([conv_sizes[0], conv_sizes[0], 1, 32])),
          'wc2': tf.Variable(tf.random_normal([conv_sizes[1], conv_sizes[1], 32, 64])),
          'wc3': tf.Variable(tf.random_normal([conv_sizes[2], conv_sizes[2], 64, 128])),

          'wd1': tf.Variable(tf.random_normal([max_pooled_area * 128, 1024])),
          'out': tf.Variable(tf.random_normal([1024, n_classes]))
      }

      biases = {
          'bc1': tf.Variable(tf.random_normal([32])),
          'bc2': tf.Variable(tf.random_normal([64])),
          'bc3': tf.Variable(tf.random_normal([128])),
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
      self.init = tf.global_variables_initializer()


  def train(self, images, labels, training_iters = 10000,
    batch_size = 128, display_step = 10, dropout = 0.75, test = False):

    # Launch the graph
    with tf.Session(graph=self.graph) as sess:
      sess.run(self.init)
      step = 0
      # Keep training until reach max iterations
      while (step+1) * batch_size < training_iters:
        # just grabbing a random set of images here.
        indices = np.floor(images.shape[0] * np.random.rand(batch_size)).astype(int)
        batch_x = images[indices]
        batch_y = labels[indices]
        print(batch_x.shape)
        # Run optimization op (backprop)
        sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y,
                                       self.keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={self.x: batch_x,
                                                              self.y: batch_y,
                                                              self.keep_prob: 1.})
            print("Iter " + str(start) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
      print("Optimization Finished!")
      if test:
        self.test(test['images'], tests['labels'])
      print("Testing Accuracy:", \
          sess.run(self.accuracy, feed_dict={self.x: images,
                                        self.y: labels,
                                        keep_prob: 1.}))

