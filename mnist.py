from tensorflow.examples.tutorials.mnist import input_data
import abc
import argparse
import sys
import tensorflow as tf


class MnistSolver(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self, input_folder, checkpoint_path_prefix, batch_size,
               step_count):
    self.data_sets = input_data.read_data_sets(input_folder, one_hot=True)
    self.checkpoint_path_prefix = checkpoint_path_prefix
    self.batch_size = batch_size
    self.step_count = step_count
    self.period_count = 10
    assert step_count % self.period_count == 0, 'step_count must be a multiple of %d' % self.period_count

  # Outputs x, y_target, train_one_step, compute_accuracy.
  @abc.abstractmethod
  def BuildNetwork(self):
    pass

  # Returns the path of the snapshot.
  def Train(self):
    with tf.Session() as sess:
      saver = tf.train.Saver()
      sess.run(tf.initialize_all_variables())
      step_count_per_period = self.step_count / self.period_count
      for period_no in range(self.period_count):
        print >> sys.stderr, 'Period %d out of %d...' % (period_no + 1,
                                                         self.period_count)
        for step_no in range(step_count_per_period):
          training_examples_batch, training_targets_batch = self.data_sets.train.next_batch(
              self.batch_size)
          self.TrainOneStep(training_examples_batch, training_targets_batch,
                            sess)
      return saver.save(sess,
                        self.checkpoint_path_prefix,
                        global_step=self.step_count)

  def TrainOneStep(self, training_examples, training_targets, sess):
    inputs = {self.x: training_examples, self.y_target: training_targets}
    if hasattr(self, 'keep_prob'):
      inputs[self.keep_prob] = 0.5
    sess.run(self.train_one_step, feed_dict=inputs)

  def Evaluate(self, checkpoint_path):
    inputs = {self.x: self.data_sets.test.images,
              self.y_target: self.data_sets.test.labels}
    if hasattr(self, 'keep_prob'):
      inputs[self.keep_prob] = 1.0  # Why 1.0?

    with tf.Session() as sess:
      saver = tf.train.Saver()
      saver.restore(sess, checkpoint_path)
      return sess.run(self.compute_accuracy, feed_dict=inputs)

  # Ultimately, this method should take an input image and output the digit
  # recognized.
  def Solve(self):
    pass


class LinearSolver(MnistSolver):
  def __init__(self, input_folder, checkpoint_path_prefix):
    MnistSolver.__init__(self,
                         input_folder,
                         checkpoint_path_prefix,
                         batch_size=100,
                         step_count=10000)

  def BuildNetwork(self):
    self.x = tf.placeholder(tf.float32, [None, 784])
    self.y_target = tf.placeholder(tf.float32, [None, 10])
    weight = tf.Variable(tf.zeros([784, 10]), 'weight')
    bias = tf.Variable(tf.zeros([10]), 'bias')
    y = tf.nn.softmax(tf.add(tf.matmul(self.x, weight), bias))
    average_loss = tf.reduce_mean(-tf.reduce_sum(self.y_target * tf.log(y),
                                                 reduction_indices=[1]))
    self.train_one_step = tf.train.GradientDescentOptimizer(0.5).minimize(
        average_loss)
    self.compute_accuracy = tf.reduce_mean(tf.to_float(tf.equal(
        tf.argmax(y, 1), tf.argmax(self.y_target, 1))))


class ConvolutionSolver(MnistSolver):
  def __init__(self, input_folder, checkpoint_path_prefix):
    MnistSolver.__init__(self,
                         input_folder,
                         checkpoint_path_prefix,
                         batch_size=50,
                         step_count=2000)

  def BuildConvolutionAndMaxPoolLayer(self, input_tensor, filter_shape,
                                      bias_shape):
    weight = tf.Variable(
        tf.truncated_normal(filter_shape, stddev=0.1),
        'weight')
    bias = tf.Variable(tf.constant(0.1, shape=bias_shape), 'bias')
    conv = tf.nn.conv2d(input_tensor,
                        weight,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(conv + bias)
    return tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

  def BuildFullyConnectedLayer(self, input_tensor, weight_shape):
    assert len(weight_shape) == 2, 'weight_shape must be rank 2'
    flattened = tf.reshape(input_tensor, [-1, weight_shape[0]])
    weight = tf.Variable(tf.truncated_normal(weight_shape, stddev=0.1))
    bias = tf.Variable(tf.constant(0.1))
    return tf.nn.relu(tf.matmul(flattened, weight) + bias)

  def BuildNetwork(self):
    self.x = tf.placeholder(tf.float32, [None, 784])
    self.y_target = tf.placeholder(tf.float32, [None, 10])
    x_as_images = tf.reshape(self.x, [-1, 28, 28, 1])
    conv1 = self.BuildConvolutionAndMaxPoolLayer(x_as_images,
                                                 filter_shape=[5, 5, 1, 32],
                                                 bias_shape=[32])
    conv2 = self.BuildConvolutionAndMaxPoolLayer(conv1,
                                                 filter_shape=[5, 5, 32, 64],
                                                 bias_shape=[64])
    fc1 = self.BuildFullyConnectedLayer(conv2, weight_shape=[7 * 7 * 64, 1024])
    self.keep_prob = tf.placeholder(tf.float32)
    dropout = tf.nn.dropout(fc1, self.keep_prob)
    fc2 = self.BuildFullyConnectedLayer(dropout, weight_shape=[1024, 10])
    y = tf.nn.softmax(fc2)
    average_loss = tf.reduce_mean(-tf.reduce_sum(self.y_target * tf.log(y),
                                                 reduction_indices=[1]))
    self.train_one_step = tf.train.AdamOptimizer(1e-4).minimize(average_loss)
    self.compute_accuracy = tf.reduce_mean(tf.to_float(tf.equal(
        tf.argmax(y, 1), tf.argmax(self.y_target, 1))))


def main():
  parser = argparse.ArgumentParser(
      description='Solve MNIST using a linear classifier.')
  parser.add_argument(
      'input_folder',
      type=str,
      help='the folder that contains the training and testing data')
  parser.add_argument('checkpoint_path_prefix',
                      type=str,
                      help='the prefix of the path of the checkpoint')
  parser.add_argument('algorithm',
                      type=str,
                      choices=['linear', 'conv'],
                      help='the algorithm used to solve MNIST')
  args = parser.parse_args()

  if args.algorithm == 'linear':
    mnist_solver = LinearSolver(args.input_folder, args.checkpoint_path_prefix)
  else:
    mnist_solver = ConvolutionSolver(args.input_folder,
                                     args.checkpoint_path_prefix)
  mnist_solver.BuildNetwork()
  checkpoint_path = mnist_solver.Train()
  accuracy = mnist_solver.Evaluate(checkpoint_path)
  print 'Achieved accuracy = %.2f%%' % (accuracy * 100)


if __name__ == '__main__':
  main()
