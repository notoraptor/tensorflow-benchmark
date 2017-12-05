# Inspired from: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_softmax.py

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import, print_function, division

import argparse
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    np.random.seed(12345678)
    tf.set_random_seed(87654321)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str, default='float32', help='Input and output dtype')
    parser.add_argument("--nbatch", type=int, default=64, help='Batch size of the layer')
    parser.add_argument("--nin", type=int, default=100, help='Input size of the layer')
    parser.add_argument("--nout", type=int, default=10, help='Output size of the layer')
    parser.add_argument("--nsteps", type=int, default=1000, help='Number of training steps')
    args = parser.parse_args()

    data = np.random.normal(size=(args.nbatch, args.nin)).astype(args.dtype)
    target = np.zeros((args.nbatch, args.nout), dtype=args.dtype)
    target[np.arange(args.nbatch), np.random.randint(0, args.nout, args.nbatch)] = 1

    # Create the model
    x = tf.constant(data, args.dtype, [args.nbatch, args.nin], verify_shape=True)
    W = tf.Variable(tf.zeros([args.nin, args.nout], dtype=args.dtype))
    b = tf.Variable(tf.zeros([args.nout], dtype=args.dtype))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.constant(target, args.dtype, [args.nbatch, args.nout], verify_shape=True)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Train
        for i in range(args.nsteps):
            sess.run(train_step)
            if (i + 1) % 100 == 0:
                print("Step %d/%d" % (i + 1, args.nsteps))
    print('End')
