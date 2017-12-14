# Inspired from: https://www.tensorflow.org/get_started/mnist/pros
from __future__ import absolute_import, print_function, division
from datetime import datetime
import argparse
import numpy as np
import tensorflow as tf

id = 0

def float32_variable_storage_getter(getter, name, shape=None, dtype=None, initializer=None, regularizer=None,
                                    trainable=True, *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
    float32 precision and then casts them to the training precision."""
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype, initializer=initializer, regularizer=regularizer,
                      trainable=trainable, *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable


def weight_variable(shape, dtype):
    global id
    id += 1
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=dtype)
    return tf.get_variable(name='weights_%d' % id, initializer=initial, dtype=dtype)


def bias_variable(shape, dtype):
    global id
    id += 1
    initial = tf.constant(0.1, shape=shape, dtype=dtype)
    return tf.get_variable(name='biases_%d' % id, initializer=initial, dtype=dtype)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class Parameters:

    def __init__(self, dtype=None, runsize=None, nsteps=None, nin=None, nout=None):
        self.dtype = dtype
        self.runsize = runsize
        self.nsteps = nsteps
        self.nin = nin
        self.nout = nout


def build_model(args):

    length = args.nin

    target = np.zeros((args.runsize, args.nout), dtype=args.dtype)
    target[np.arange(args.runsize), np.random.randint(low=0, high=args.nout, size=args.runsize)] = 1

    x_image = tf.random_normal(shape=(args.runsize, length, length, 1), dtype=args.dtype, name='input')
    y_ = tf.constant(target, args.dtype, [args.runsize, args.nout], verify_shape=True)

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

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=tf.cast(y_conv, tf.float32)))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    return train_step


def run_benchmark(args, device_names, session_config):
    nruns = len(device_names)
    trains = []
    # Note: This scopes should force trainable variables to be stored as float32
    with tf.variable_scope('fp32_storage', custom_getter=float32_variable_storage_getter):
        for i in range(nruns):
            with tf.name_scope('%s_benchmark_run_%d' % (args.dtype, i)):
                with tf.device(device_names[i]):
                    trains += [build_model(args)]

    print('Testing dtype', args.dtype)
    profile_message = ''
    with tf.Session(config=session_config) as sess:
        # Start profiling
        time_start = datetime.now()
        sess.run(tf.global_variables_initializer())
        for i in range(args.nsteps):
            sess.run(trains)
            if (i + 1) % 100 == 0:
                print("Step %d/%d" % (i + 1, args.nsteps))
        time_end = datetime.now()  # end profiling
        if args.nsteps % 100 != 0:
            print('End (%d steps)' % args.nsteps)
        time_spent = time_end - time_start
        seconds = time_spent.seconds + time_spent.days * 24 * 3600
        # print('execution time:', seconds, 'sec +', time_spent.microseconds, 'microsec')
        profile_message = 'execution time: %s sec + %s microsec' % (seconds, time_spent.microseconds)
    return profile_message

if __name__ == '__main__':
    np.random.seed(12345678)
    tf.set_random_seed(87654321)

    default_nbatch = 100
    default_nin = 64
    default_nout = 10
    default_nsteps = 1000
    default_nruns = 2
    default_ngpus = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", action='append',
                        help='dtypes to test (default: only float32). Ex: --dtype float16 --dtype float32 ...')
    parser.add_argument("--nbatch", type=int, default=default_nbatch,
                        help='Batch size of the layer (default %d)' % default_nbatch)
    parser.add_argument("--nin", type=int, default=default_nin,
                        help='Input size x of the layer (for layer size x * x), should be a multiple of 4 '
                             '(default %d)' % default_nin)
    parser.add_argument("--nout", type=int, default=default_nout,
                        help='Output size of the layer (default %d)' % default_nout)
    parser.add_argument("--nsteps", type=int, default=default_nsteps,
                        help='Number of training steps (default %d)' % default_nsteps)
    parser.add_argument("--nruns", type=int, default=default_nruns,
                        help='Number of parallel runs (default %d). Each run will train with nbatch/nruns inputs. '
                             'nbatch must be a multiple of nruns.' % default_nruns)
    parser.add_argument("--ngpus", type=int, default=default_ngpus,
                        help='Number of GPUs to use (default %d). Tensorflow will label GPUs from 0 to ngpus-1. '
                             'Shoule be <= nruns.' % default_ngpus)
    parser.add_argument("--default-gpu", type=int, default=0,
                        help='Default GPU to use (default 0). '
                             'If value is -1 or if ngpus == 0, then CPU is used as default processing unit.')
    parser.add_argument("--log", action='store_true', default=False,
                        help='Log device placement (default false)')
    args = parser.parse_args()

    assert args.nbatch > 0, "nbatch must be strictly positive."
    assert args.nbatch % args.nruns == 0, "nbatch must be a multiple of nruns."
    assert args.ngpus <= args.nruns, "Number of GPUs must be less than or equal to number of runs."
    assert args.nin % 4 == 0, "Input size must be a multiple of 4."

    tested_dtypes = set(args.dtype) if args.dtype else {'float32'}

    ndefault_gpu = args.nruns - args.ngpus
    default_gpu = '/cpu:0' if (args.ngpus == 0 or args.default_gpu < 0) else '/gpu:%d' % args.default_gpu
    device_names = []
    for i in range(ndefault_gpu):
        device_names += [default_gpu]
    for i in range(args.ngpus):
        device_names += ['/gpu:%d' % i]

    runsize = args.nbatch // args.nruns
    print('Testing %d runs with %d samples per run (total %d samples).' % (args.nruns, runsize, args.nbatch))
    print()
    session_config = tf.ConfigProto(log_device_placement=True) if args.log else None
    profiles = {}
    for dtype in tested_dtypes:
        parameters = Parameters(dtype=dtype, runsize=runsize, nsteps=args.nsteps, nin=args.nin, nout=args.nout)
        profiles[dtype] = run_benchmark(parameters, device_names, session_config)
    print()
    for dtype in sorted(profiles.keys()):
        print('%s:' % dtype, profiles[dtype])
