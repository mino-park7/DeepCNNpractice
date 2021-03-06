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
"""Builds the CIFAR-10 network.

Summary of available functions:

    # Compute input images and labels for training. If you would like to run
    # evaluations, use inputs() instead.
    inputs, labels = distorted_inputs()


    # Compute inference on the model inputs to make a prediction.
    predictions = inference(inputs)

    # Compute the total loss of the prediction with respect to the labels.
    loss = loss(predictions, labels)

    # Create a graph to run noe step of training with respect to the loss.
    train_op = train(loss, global_step)

"""

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse # CmdLine 툴의 Argv처리 라이브러리
import os
import re #regular expression
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import cifar10_input

parser = argparse.ArgumentParser()

# Basic model parameters.
# 모델 파라미터를 CLI에서 바로 입력 가능하게 해줌 ex) python3 cifar10.py --batch_size=200 --data_dir=~/datadir
parser.add_argument('--batch_size',type=int, default=128,
                    help='Number of images to process in a batch.')
parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
                    help='Path to the CIFAR-10 data directory.')
parser.add_argument('--use_fp16', type=bool, default=False,
                    help='Train the model using fp16.')



# Global constants describing the CIFAR-10 data set.
# CIFAR-10 data set에서 global constant 불러옴
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999 # variable average 를 이용한 boosting 효과
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1 # 처음에는 update 를 크게, 나중에 갈수록 update 를 세밀하게
INITIAL_LEARNING_RATE = 0.1

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def _activation_summary(x):
    """
    Helper to create summaries for activation

    :param x: Tensor
    :return:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/'%TOWER_NAME,'',x.op.name)
    tf.summary.histogram(tensor_name+'/activations',x)
    #tf.nn.zero_fraction(tensor) -> 해당 텐서중에 0값의 비율 (sparsity)
    tf.summary.scalar(tensor_name+'/sparsity',tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """
    Helper to create a Variable stored on CPU memory.
    CPU memory에 올려놓는 variable 생성을 위한 function
    :param name: name of the variable
    :param shape: list of ints
    :param initializer: initializer for Variable

    :return:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)

    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    """
    Helper to create an initialized variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified

    :param name: name of the variable
    :param shape: list of ints
    :param stddev: standard deviation of a truncated Gaussian
    :param wd: add L2Loss weight decay multiplied by this float. If None, weight
               decay is not added for this Variable.

    :return:
        Variable Tensor
    """

    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
    ) # activation function으로 ReLu를 사용하는데, ReLu는 0이하일 때 기울기가 0이므로, dead neuron 발생 방지를 위해
      # 양의 값으로 truncated(절단된) normal distribution을 이용한 initializer 이용

    # weight decay를 통해 L2 regularization 을 추가하여 특정 weight에 의해 정답이 결정되어 버리는 현상을 방지
    if wd is not None :
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var
# cifar10_input.distorted_inputs()를 이용한 인풋 (for training)
def distorted_inputs():
    """
    Construct distorted input for CIFAR training using the Reader ops.

    :return:
        images: Images. 4-D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1-D tensor of [batch_size] size.

    :raises:
        ValueError : If no data_dir
    """

    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir,'cifar-10-batches-bin')
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                          batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels
# cifar10_inputs()를 이용한 input (for evaluation)
def inputs(eval_data):
    """
    Construct input for CIFAR evaluation using the Reader ops

    :param eval_data: bool, indicating if one should use the train or eval data set.

    :return:
        images : Images. 4-D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1-D tensor of [batch_size] size.
    :raises:
        ValueError: if no data_dir
    """

    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.inputs(eval_data=eval_data,
                                          data_dir=data_dir,
                                          batch_size=FLAGS.batch_size)

    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels

# network model 에 해당하는 부분
def inference(images):
    """
    Build the CIFAR-10 model

    :param images: Images returned from distorted_inputs() or inputs()

    :return:
        Logits.
    """

    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # conv1
    with tf.variable_scope('conv1') as scope:
        # 5x5크기의 64개 filter 를 width x height x RGB channel 크기의 input에 적용 -> [5, 5, 3, 64]
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 3, 64],
                                             stddev=5e-2,
                                             wd=0.0) # weight decay 를 통한 l2 reg 을 사용한 weight 설정
        # width, height 방향으로 stride 1씩만하고, padding해서 activation map 크기 작아지지 않게
        conv = tf.nn.conv2d(images, kernel, [1,1,1,1], padding='SAME')
        # biases 설정
        biases = _variable_on_cpu('biases',[64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv,biases)
        # activation function 으로 relu 사용
        conv1 = tf.nn.relu(pre_activation, name = scope.name)
        _activation_summary(conv1)

    # pool1
    # 3x3 판으로 stride 2씩 하면서 max pooling
    pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1],
                           padding='SAME', name='pool1')

    # norm1
    # local_response_normalization : generalization error를 줄이기 위한 정규화, http://nmhkahn.github.io/Casestudy-CNN 참조
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75,
                      name= 'norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5,5,64,64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1,1,1,1], padding="SAME")
        biases = _variable_on_cpu('biases',[64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75,
                      name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1],
                           strides=[1,2,2,1], padding='SAME', name='pool2')

    # local3
    # fully-connected layer
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights',shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))

        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384,192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights)+biases, name=scope.name)
        _activation_summary(local4)

    # linear layer(WX+b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                              stddev=1/192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights),biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear

# logit과 label을 받아 loss function을 계산 하는 function
def loss(logits, labels):
    """
    Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg"

    :param logits: Logits from inference().
    :param labels: Labels from distorted_inputs or inputs(). 1-D tensor of shape [batch_size]

    :return:
        Loss tensor of type float
    """

    # Calculate average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits= logits, name='cross-entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    # tf.Graph() 내에 losses를 추가
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    # loss가 cross_entropy + L2 term으로 되어 있는 것을 tf.add_n을 통해서 합쳐서 리턴
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """
    Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    # mini-batch training에서 loss function이 진동을 많이 하기 때문에, moving average를 사용해서 smoothing 된 값으로
    # loss function minimize 시킴
    :param total_loss: Total loss from loss().

    :return:
        loss_averages_op: op for generating moving averages of losses.
    """

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name="avg")
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def train(total_loss, global_step):
    """
    Train CIFAR-10 model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    :param total_loss: Total loss from loss().
    :param global_step: Integer Variable counting the number of training steps
                        processed.
    :return:
        train_op: op for training.
    """

    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size # 50000/128 ~= 390
    # 390.xx * 380 ~= 148437, 약 148k step 마다 learning rate decaying
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)
    # Optimizer 사용법 : 기본적으로 ~Optimizer.minimize(loss)를 하면 자동으로 해주지만, 중간에 원하는 연산이 필요 할때에는
        # opt = tf.train.GradientDescentOptimizer(learning_rate)로 optimizer 객체 생성
        # opt.compute_gradients(loss)로 gradient 계산 ( return: list of (gradient, variable) pair)
        # opt.apply_gradients()로 minimize 적용
    # 이 code에서는 중간에 gradients와 trainable variables를 tensorboard summary에 저장시키기 위해서 중간과정 사용)
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]): #tf.control_dependencies() -> 얘 먼저 계산하고, 뒤에것들 계산한다
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads,global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name+'/gradients',grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY,global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def maybe_download_and_extract():
    """
    Download and extract the tarball from Alex's website.

    """

    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1] #cifar-10-binary.tar.gz
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%'%(filename,float(count*block_size) / float(total_size)
                                                           * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully douwnloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


if __name__ == "__main__":
    FLAGS = parser.parse_args()





















