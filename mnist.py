from tensorflow.examples.tutorials.mnist import input_data
import argparse
import sys
import tensorflow as tf


BATCH_SIZE = 100
LEARNING_RATE = 0.5
PERIOD_COUNT = 10
TRAINING_STEP_COUNT = 1000
training_steps_per_period = TRAINING_STEP_COUNT / PERIOD_COUNT


def main():
  parser = argparse.ArgumentParser(
      description='Train and test an MNIST network')
  parser.add_argument(
      'input_folder',
      type=str,
      help='the folder that contains the training and testing data')
  parser.add_argument('checkpoint_path_prefix',
                      type=str,
                      help='the prefix of the path of the checkpoint')
  args = parser.parse_args()

  # Build the network.
  x = tf.placeholder(tf.float32, [None, 784])
  y_target = tf.placeholder(tf.float32, [None, 10])
  w = tf.Variable(tf.zeros([784, 10]), 'w')
  b = tf.Variable(tf.zeros([10]), 'b')
  y = tf.nn.softmax(tf.add(tf.matmul(x, w), b))
  average_loss = tf.reduce_mean(tf.reduce_sum(
      tf.neg(tf.mul(y_target, tf.log(y))),
      reduction_indices=[1]))
  train_one_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(
      average_loss)
  compute_accuracy = tf.reduce_mean(tf.to_float(tf.equal(
      tf.argmax(y, 1), tf.argmax(y_target, 1))))

  with tf.Session() as sess:
    saver = tf.train.Saver()
    # Initialize.
    sess.run(tf.initialize_all_variables())
    # Train.
    mnist = input_data.read_data_sets(args.input_folder, one_hot=True)
    for period in range(PERIOD_COUNT):
      print >> sys.stderr, 'Period %d out of %d...' % (period, PERIOD_COUNT)
      for i in range(training_steps_per_period):
        batch_training_examples, batch_training_targets = mnist.train.next_batch(
            BATCH_SIZE)
        sess.run(train_one_step,
                 feed_dict={x: batch_training_examples,
                            y_target: batch_training_targets})
    # Save the checkpoint.
    checkpoint_path = saver.save(sess,
                                 args.checkpoint_path_prefix,
                                 global_step=TRAINING_STEP_COUNT)

  with tf.Session() as sess:
    # Restore the checkpoint.
    saver.restore(sess, checkpoint_path)
    # Evaluate on test data.
    accuracy = sess.run(compute_accuracy,
                        feed_dict={x: mnist.test.images,
                                   y_target: mnist.test.labels})
    print 'Achieved accuracy = %.2f%%' % (accuracy * 100)


if __name__ == '__main__':
  main()
