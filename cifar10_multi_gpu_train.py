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
# python2.x와의 호환성을 위한 import문
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
    # 해당 스코프 내의 값만 get 하기위해 get_collection으로 losses를 정의
    _ = cifar10.loss(logits, labels)

    # Assemble all of the losses for the current tower only.
    # 해당 scope(한 tower내) 에서 losses를 get_collection 해옴. (cifar10.py에서 모델 내에 losses가 정의되어 있음)
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name = 'total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0_9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*' % cifar10.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)

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
        # g : grad0_gpu0, ..., grad0_gpuN, _ : var0_gpu0, ... , var0_gpuN
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            # tf.expand_dims(a,b) : a tensor의 b위치에 dimension 하나 늘림
            expanded_g = tf.expand_dims(g, 0) # ->

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable
        # 타워마다 모든 vars는 다 같으니깐, 첫번째 타워의 vars만 가져와서 grad_and_var에
        # 다시 저장시키고 average_grads에 append 시키고 리턴
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
            # capacity : queue를 얼마나 많이 쌓아 놓을 것인가? 아마도 넉넉잡아서 gpu갯수 2배로 한듯함(gpu가 데이터를 처리하므로)
            [images, labels], capacity=2 * FLAGS.num_gpus
        )
        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' %(cifar10.TOWER_NAME, i)) as scope:
                        # Dequeues one batch for the GPU
                        image_batch, label_batch = batch_queue.dequeue()
                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constucts the entire CIFAR model but shares the variables across
                        # all towers.
                        loss = tower_loss(scope, image_batch, label_batch)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        #Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add hisograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar10.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU implementations
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement
        ))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() -start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec =num_examples_per_step/ duration
                sec_per_batch = duration / FLAGS.num_gpus

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f'
                              'sec/batch')
                print(format_str %(datetime.now(), step, loss_value,
                                   examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step+1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    cifar10.FLAGS = parser.parse_args()
    tf.app.run()
