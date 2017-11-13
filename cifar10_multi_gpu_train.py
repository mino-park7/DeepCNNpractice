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
    """Calculte the total loss on a single tower running the CIFAR model.

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
