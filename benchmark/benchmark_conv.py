# Inspired from: https://www.tensorflow.org/get_started/mnist/pros
from __future__ import absolute_import, print_function, division

import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf

variables_id = 0


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
    global variables_id
    variables_id += 1
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=dtype)
    return tf.get_variable(name='weights_%d' % variables_id, initializer=initial, dtype=dtype)


def bias_variable(shape, dtype):
    global variables_id
    variables_id += 1
    initial = tf.constant(0.1, shape=shape, dtype=dtype)
    return tf.get_variable(name='biases_%d' % variables_id, initializer=initial, dtype=dtype)


def conv2d(inputs, thetas):
    return tf.nn.conv2d(inputs, thetas, strides=[1, 1, 1, 1], padding='SAME')


def layer(inputs, filter_size, n_output_channels, dtype):
    thetas = weight_variable([filter_size, filter_size, 1, n_output_channels], dtype)
    biases = bias_variable([n_output_channels], dtype)
    return conv2d(inputs, thetas) + biases


class Parameters:

    def __init__(self, dtype=None, runsize=None, nsteps=None, nin=None, layers=None, filter_size=None):
        self.dtype = dtype
        self.runsize = runsize
        self.nsteps = nsteps
        self.nin = nin
        self.layers = layers
        self.filter_size = filter_size


def build_inputs(args):
    # To save time, we will build inputs as variables once, then
    # these input variables will be initialized once and reused as-is
    #  in all computations.
    x_image_initializer = tf.random_normal(shape=(args.runsize, args.nin, args.nin, 1), dtype=args.dtype)
    x_image = tf.get_variable(name='input', initializer=x_image_initializer, dtype=args.dtype)
    return x_image


def build_model(args, x_image):
    outputs = [x_image]
    for i in range(args.layers):
        out = layer(outputs[-1], args.filter_size, 1, args.dtype)
        outputs.append(out)
    return outputs[-1]


def run_benchmark(args, device_names, session_config):
    nruns = len(device_names)
    trains = []
    with tf.variable_scope('benchmark_%s' % args.dtype):
        x_image = build_inputs(args)
        # Note: This scopes should force trainable variables to be stored as float32
        with tf.variable_scope('fp32_storage', custom_getter=float32_variable_storage_getter):
            for i in range(nruns):
                with tf.name_scope('run_%d' % i):
                    with tf.device(device_names[i]):
                        trains += [build_model(args, x_image)]

    print('Testing dtype', args.dtype)
    with tf.Session(config=session_config) as sess:
        # Let's NOT profile variables initialization.
        print('Initializing variables (not profiled) ...')
        sess.run(tf.global_variables_initializer())
        print('... End initialization (not profiled). Starting train ...')
        # Start profiling
        time_start = datetime.now()
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

    default_nbatch = 4096
    default_nin = 32
    default_nsteps = 1000
    default_layers = 5
    default_filter_size = 32
    default_nruns = 2
    default_ngpus = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", action='append',
                        help='dtypes to test (default: only float32). Ex: --dtype float16 --dtype float32 ...')
    parser.add_argument("--nbatch", type=int, default=default_nbatch,
                        help='Batch size of the layer (default %d)' % default_nbatch)
    parser.add_argument("--nin", type=int, default=default_nin,
                        help='Input size of the layer => input shape: nin * nin (default %d)' % default_nin)
    parser.add_argument("--nsteps", type=int, default=default_nsteps,
                        help='Number of training steps (default %d)' % default_nsteps)
    parser.add_argument("--layers", type=int, default=default_layers,
                        help='Number of layers (default %d)' % default_layers)
    parser.add_argument("--filter-size", type=int, default=default_filter_size,
                        help='Conv filter size => filter shape: filter-size * filter-size (default %d)' % default_filter_size)
    parser.add_argument("--nruns", type=int, default=default_nruns,
                        help='Number of parallel runs (default %d). Each run will train with nbatch/nruns inputs. '
                             'nbatch must be a multiple of nruns.' % default_nruns)
    parser.add_argument("--ngpus", type=int, default=default_ngpus,
                        help='Number of GPUs to use (default %d). Tensorflow will label GPUs from 0 to ngpus-1. '
                             'Shoule be <= nruns.' % default_ngpus)
    parser.add_argument("--default-gpu", type=int, default=0,
                        help='Default GPU to use (default 0). '
                             'If value is -1 or ngpus == 0, then CPU is used as default processing unit.')
    parser.add_argument("--log", action='store_true', default=False,
                        help='Log device placement (default false)')
    args = parser.parse_args()

    assert args.filter_size > 0, "Conv filter size must be strictly positive."
    assert args.nbatch > 0, "nbatch must be strictly positive."
    assert args.nbatch % args.nruns == 0, "nbatch must be a multiple of nruns."
    assert args.ngpus <= args.nruns, "Number of GPUs must be less than or equal to number of runs."

    tested_dtypes = set(args.dtype) if args.dtype else {'float32'}

    ndefault_gpu = args.nruns - args.ngpus
    default_gpu = '/cpu:0' if (args.ngpus == 0 or args.default_gpu < 0) else '/gpu:%d' % args.default_gpu
    device_names = []
    for i in range(ndefault_gpu):
        device_names += [default_gpu]
    for i in range(args.ngpus):
        device_names += ['/gpu:%d' % i]

    runsize = args.nbatch // args.nruns
    print('Dtypes:', ', '.join(sorted(tested_dtypes)))
    print('Samples: %d, input: %d * %d, nsteps: %d, layers: %d, conv filter size: %d'
          % (args.nbatch, args.nin, args.nin, args.nsteps, args.layers, args.filter_size))
    print('Testing %d runs with %d samples per run.' % (args.nruns, runsize))
    print('Devices for runs:', ', '.join(device_names))
    print()
    session_config = tf.ConfigProto(log_device_placement=True) if args.log else None
    profiles = {}
    for dtype in tested_dtypes:
        parameters = Parameters(dtype=dtype, runsize=runsize, nsteps=args.nsteps, nin=args.nin,
                                layers=args.layers, filter_size=args.filter_size)
        profiles[dtype] = run_benchmark(parameters, device_names, session_config)
    print()
    for dtype in sorted(profiles.keys()):
        print('%s:' % dtype, profiles[dtype])
