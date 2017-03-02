from __future__ import print_function
from preprocess import Preprocess
from queue import prepare, dequeue_and_enqueue
from net import Net
import numpy as np
import tensorflow as tf

class Bbox:

  def __init__(self, side=280, n_out=4, n_classes=280):
    self.side = side
    self.n_out = n_out
    self.n_classes = 280

  def define(self, n_samples, learning_rate=0.01):
    # tf Graph Input
    self.X = X = tf.placeholder(tf.float32, [None, np.square(self.side)])
    self.Yx = Yx = tf.placeholder(tf.int64, [None])
    self.Yy = Yy = tf.placeholder(tf.int64, [None])
    self.Ywidth = Ywidth = tf.placeholder(tf.int64, [None])
    self.Yheight = Yheight = tf.placeholder(tf.int64, [None])

    # Position X.
    Wx = tf.Variable(tf.random_normal([np.square(self.side), self.n_classes]))
    bx = tf.Variable(tf.random_normal([self.n_classes]))
    ax = tf.add(tf.matmul(X, Wx), bx)
    costx = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ax, labels=Yx))
    correctx = tf.equal(tf.argmax(ax, axis=1), Yx)

    # Position X.
    Wy = tf.Variable(tf.random_normal([np.square(self.side), self.n_classes]))
    by = tf.Variable(tf.random_normal([self.n_out]))
    ay = tf.add(tf.matmul(X, Wy), by)
    costy = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ay, labels=Yy))
    correcty = tf.equal(tf.argmax(ay, axis=1), Yy)

    # width
    Wwidth = tf.Variable(tf.random_normal([np.square(self.side), self.n_classes]))
    bwidth = tf.Variable(tf.random_normal([self.n_out]))
    awidth = tf.add(tf.matmul(X, Wwidth), bwidth)
    costwidth = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=awidth, labels=Ywidth))
    correctwidth = tf.equal(tf.argmax(awidth, axis=1), Ywidth)

    # height
    Wheight = tf.Variable(tf.random_normal([np.square(self.side), self.n_classes]))
    bheight = tf.Variable(tf.random_normal([self.n_out]))
    aheight = tf.add(tf.matmul(X, Wheight), bheight)
    costheight = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=aheight, labels=Yheight))
    correctheight = tf.equal(tf.argmax(aheight, axis=1), Yheight)

    self.total_cost = total_cost = costx + costy + costwidth + costheight
    self.optimizer = optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_cost)

    correct_pred = tf.stack([correctx, correcty, correctwidth, correctheight])
    self.accuracy = accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    self.init = tf.global_variables_initializer()

  def train(self, images, labels, display_step=10, training_iters=10000, batch_size=128, test=False):
    # Launch the graph
    with tf.Session() as sess:
        sess.run(self.init)

        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            print('---'+str(step)+'----')
            images = dequeue_and_enqueue(images, batch_size)
            batch_x = images[:batch_size]
            labels = dequeue_and_enqueue(labels, batch_size)
            batch_y = labels[:batch_size]
            feed_dict = self.feed_dict(batch_x, batch_y)
            sess.run(self.optimizer, feed_dict=feed_dict)

            # Display logs per epoch step
            if step % display_step == 0:
                c, accuracy = sess.run([self.total_cost, self.accuracy], feed_dict=feed_dict)
                print('Accuracy', accuracy)
                print("Step:", '%04d' % (step), "cost=", "{:.9f}".format(c))
            step +=1

        print("Optimization Finished!")
        feed_dict = self.feed_dict(images, labels)
        training_cost = sess.run(self.cost, feed_dict=feed_dict)
        print("Training cost=", training_cost)

        if test:
          # Testing example, as requested (Issue #2)
          test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
          test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

          print("Testing... (Mean square loss Comparison)")
          testing_cost = sess.run(
              tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
              feed_dict={X: test_X, Y: test_Y})  # same function as cost above
          print("Testing cost=", testing_cost)
          print("Absolute mean square loss difference:", abs(
              training_cost - testing_cost))


  def feed_dict(self, batch_x, batch_y):
    return {
      self.X: batch_x,
      self.Yx: batch_y[:, 0],
      self.Yy: batch_y[:, 1],
      self.Ywidth: batch_y[:, 2],
      self.Yheight: batch_y[:, 3],
    }

preprocess = Preprocess()
print('.')
data = preprocess.load_file('data/train_sequences00.pickle');
print('.')
print('preparing...')
side = 280
images, labels = prepare(data)
data = None
images = images.reshape(images.shape[0], images.shape[1]*images.shape[2])
labels = labels[:, 1:5]
bbox = Bbox(n_out=1)
bbox.define(images.shape[0])
bbox.train(images, labels, display_step=3)
