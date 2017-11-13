# Copyright 2015 The TensorFlow Authors. All Rights Reserved
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

""" A binary to train CIFAR-10 using multiple GPUs with synchronous updates.

Accuracy:
    cifar10_multi_gpu_train.py achieves ~86% accuracy after 100K steps (256
    epochs of data) as judged by cifar10_eval.py.

Speed: TBD

Usage:
    Please see the tutorial and website for how to download the CIFAR-10
    data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
import tensorflow as tf
import cifar10

parser = cifar10.parser

parser.add_argument('--train_dir', type=str, default='/tmp/cifar10_train',
                    help='Directory where to write event logs and checkpoint')
parser.add_argument('--max_steps', type=int, default=1000000,
                    help='Number of batches to run.')
parser.add_argument('--num_gpus', type=int, default=1,
                    help='How many GPUs to use.')
parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to log device placement.')

def tower_loss(scope, images, labels):
    """Calculate the total loss on a single tower running the CIFAR model.

    Args:
        scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
        images: Images. 4-D Tensor of shape [batch_size, height, width, 3].
        labels: Labels. 1-D Tensor of shape [batch_size].

    Returns:
        Tensor of shape [] containing the total loss for a batch of data
    """

    # Build inference Graph.
    logits = cifar10.inference(images)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = cifar10.loss(logits, labels)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name = 'total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0_9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*' % cifar10.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, 1)

    return total_loss


def average_gradients(tower_grads):
    """
    Calculate the average gradient for each shared variable across all towers.

    :param tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    :return:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """

    average_grads=[]
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train():
    """
    Train CIFAR-10 for a number of steps.
    :return:
    """

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False
        )

        # Calculate the learning rate schedule.
        num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                 FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        cifar10.LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.GradientDescentOptimizer(lr)

        # Get images and labels for CIFAR-10.
        images, labels = cifar10.distorted_inputs()
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [images, labels], capacity=2 * FLAGS.num_gpus
        )
        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' %(cifar10.TOWER_NAME, i)) as scope:
                        







if __name__ == '__main__':
    FLAGS = parser.parse_args()
    tf.app.run()