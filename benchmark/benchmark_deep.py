# Inspired from: https://www.tensorflow.org/get_started/mnist/pros
from __future__ import absolute_import, print_function, division
import argparse
import numpy as np
import tensorflow as tf


def weight_variable(shape, dtype):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=dtype)
    return tf.Variable(initial)


def bias_variable(shape, dtype):
    initial = tf.constant(0.1, shape=shape, dtype=dtype)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


if __name__ == '__main__':
    np.random.seed(12345678)
    tf.set_random_seed(87654321)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str, default='float32', help='Input and output dtype')
    parser.add_argument("--nbatch", type=int, default=64, help='Batch size of the layer')
    parser.add_argument("--nin", type=int, default=100, help='Input size of the layer')
    parser.add_argument("--nout", type=int, default=10, help='Output size of the layer')
    parser.add_argument("--nsteps", type=int, default=20000, help='Number of training steps')
    args = parser.parse_args()

    length = args.nin
    image_size = length * length

    data = np.random.normal(size=(args.nbatch, length, length, 1)).astype(args.dtype)
    target = np.zeros((args.nbatch, args.nout), dtype=args.dtype)
    target[np.arange(args.nbatch), np.random.randint(0, args.nout, args.nbatch)] = 1

    x_image = tf.constant(data, args.dtype, [args.nbatch, length, length, 1], verify_shape=True)
    y_ = tf.constant(target, args.dtype, [args.nbatch, args.nout], verify_shape=True)

    W_conv1 = weight_variable([5, 5, 1, 32], args.dtype)
    b_conv1 = bias_variable([32], args.dtype)

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64], args.dtype)
    b_conv2 = bias_variable([64], args.dtype)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # (length / 4) * (length / 4) * 64 == length * length * 4

    W_fc1 = weight_variable([length * length * 4, 1024], args.dtype)
    b_fc1 = bias_variable([1024], args.dtype)

    h_pool2_flat = tf.reshape(h_pool2, [-1, length * length * 4])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.constant(0.5, args.dtype)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, args.nout], args.dtype)
    b_fc2 = bias_variable([args.nout], args.dtype)

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(args.nsteps):
            sess.run(train_step)
            if (i + 1) % 100 == 0:
                print("Step %d/%d" % (i + 1, args.nsteps))
        print('End (%d)' % args.nsteps)
