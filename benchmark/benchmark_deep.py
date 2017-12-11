# Inspired from: https://www.tensorflow.org/get_started/mnist/pros
from __future__ import absolute_import, print_function, division
from datetime import datetime
import argparse
import numpy as np
import tensorflow as tf


def weight_variable(shape, dtype):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=dtype)
    return tf.Variable(initial, dtype=dtype)


def bias_variable(shape, dtype):
    initial = tf.constant(0.1, shape=shape, dtype=dtype)
    return tf.Variable(initial, dtype=dtype)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def build_model(args):
    assert args.nin % 4 == 0, "Input size must be a multiple of 4."

    length = args.nin

    data = np.random.normal(size=(args.nbatch, length, length, 1)).astype(args.dtype)
    target = np.zeros((args.nbatch, args.nout), dtype=args.dtype)
    target[np.arange(args.nbatch), np.random.randint(low=0, high=args.nout, size=args.nbatch)] = 1

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

    return y_, y_conv


def build_train(y_, y_conv):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    return train_step


def build_accuracy(y_, y_conv, dtype):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, dtype)
    accuracy = tf.reduce_mean(correct_prediction)
    return accuracy


if __name__ == '__main__':
    np.random.seed(12345678)
    tf.set_random_seed(87654321)

    parser = argparse.ArgumentParser()
    default_dtype = 'float32'
    default_nbatch = 100
    default_nin = 64
    default_nout = 10
    default_nsteps = 1000
    default_nruns = 2
    default_ngpus = 1
    parser.add_argument("--dtype", type=str, default=default_dtype, help='Input and output dtype (default %s)' % default_dtype)
    parser.add_argument("--nbatch", type=int, default=default_nbatch, help='Batch size of the layer (default %d)' % default_nbatch)
    parser.add_argument("--nin", type=int, default=default_nin,
                        help='Input size (size x size) of the layer, should be a multiple of 4 (default %d)' % default_nin)
    parser.add_argument("--nout", type=int, default=default_nout, help='Output size of the layer (default %d)' % default_nout)
    parser.add_argument("--nsteps", type=int, default=default_nsteps, help='Number of training steps (default %d)' % default_nsteps)
    parser.add_argument("--nruns", type=int, default=default_nruns,
                        help='Number of parallel runs, each run will train with nbatch/nruns inputs '
                             '(default %d). nbatch must be a multiple of nruns.' % default_nruns)
    parser.add_argument("--ngpus", type=int, default=default_ngpus,
                        help='Number of GPUs to use (default %d). '
                             'Tensorflow will label GPUs from gpu:0 to gpu:[ngpus-1]. Shoule be <= nruns.' % default_ngpus)
    parser.add_argument("--log", action='store_true', default=False, help='Log device placement (default false)')
    args = parser.parse_args()

    assert args.nbatch > 0, "nbatch must be strictly positive."
    assert args.nbatch % args.nruns == 0, "nbatch must be a multiple of nruns."
    assert args.ngpus <= args.nruns, "Number of GPUs must be less than or equal to number of runs."

    device_names = []
    trains = []
    accuracies = []
    ncpus = args.nruns - args.ngpus

    for i in range(ncpus):
        device_names += ['/cpu:0']
    for i in range(args.ngpus):
        device_names += ['/gpu:%d' % i]

    args.nbatch //= args.nruns
    for i in range(args.nruns):
        with tf.name_scope('benchmark_run_%d' % i):
            with tf.device(device_names[i]):
                y_, y_conv = build_model(args)
                trains += [build_train(y_, y_conv)]
                accuracies += [build_accuracy(y_, y_conv, args.dtype)]
    with tf.name_scope('benchmark_run_final'):
        with tf.device('/cpu:0'):
            sum_accuracy = accuracies[0]
            for i in range(1, args.nruns):
                sum_accuracy += accuracies[i]
            avg_accuracy = sum_accuracy / args.nruns

    print('Testing %d runs.' % args.nruns)
    print()
    config = tf.ConfigProto(log_device_placement=True) if args.log else None
    with tf.Session(config=config) as sess:
        # Start profiling
        time_start = datetime.now()
        sess.run(tf.global_variables_initializer())
        for i in range(args.nsteps):
            sess.run(trains)
            if (i + 1) % 100 == 0:
                print("Step %d/%d" % (i + 1, args.nsteps))
        # end profiling
        time_end = datetime.now()
        if args.nsteps % 100 != 0:
            print('End (%d steps)' % args.nsteps)
        sess.run(avg_accuracy)
        print('End computation')
        time_spent = time_end - time_start
        seconds = time_spent.seconds + time_spent.days * 24 * 3600
        print('Execution time:', seconds, 'sec +', time_spent.microseconds, 'microsec')
