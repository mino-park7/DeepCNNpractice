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
"""Routine for decoding the CIFAR-10 binary file format."""

# python2.x와의 호환성을 위한 import문
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
#from six.moves import xrange #from six.moves import xrange # python3.x에서는 그냥 range쓰면 됩니다
import tensorflow as tf

#Process images of this size. NOte that this differs from the original CIFAR
#image size of 32 x 32. If one alters this number, then the entire model
#architecture will change and any model would need to be retrained

# 전역 변수 설정
IMAGE_SIZE = 24 # 원본 크기는 32x32x3이지만, data augmentation을 위해 사이즈를 줄임, 자세한 부분은 distorted_input에서 설명
# Data augmentation이란? : training 인풋 이미지를 변형시켜서 오버피팅을 억제하는 기법 http://nmhkahn.github.io/CNN-Practice 참조
NUM_CLASSES = 10 # label 갯수
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000 # training data의 1 epoch 갯수
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000 # evaluation data의 1 epoch 갯수

def read_cifar10(filename_queue) :
    """Reads and parses examples from CIFAR10 data files.

    Recommendation: if you want N-way read parallelism, call this function N times.
    this will give you N independent Readers reading different files & positions within
    those files, which will give better mixing of examples.

    :param filename_queue:A queue of strings with the filenames to read from.

    :return:
        An object representing a single example, with the following fields:
            height: number of rows in the result (32)
            width: number of columns in the result (32)
            depth: number of color channels in the result (3)
            key: a scalar string Tensor describing the filename & record number for this example
            label: an int32 Tensor with the label in the range 0...9.
            uint8image: a [height, width, depth] uint8 Tensor with the image data
    """
        #uint8? : unsigned integer, 부호없는 정수, uint8의 경우 0 ~ 255 까지 표현 가능(이미지에 적합)

    # read_cifar10의 function return 값들을 담기 위한 class(object) 선언
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    #Dimensions of the images in the CIRFAR-10 dataset.
    #See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the input format.
    # binary file의 경우 한 데이터 당 <1 x label><3072 x pixel> 로 구성
    label_bytes = 1 # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height*result.width*result.depth
    #Every record consists of a label followed by the image, with a fixed number of bytes for each.
    record_bytes = label_bytes+image_bytes

    #Read a record, getting filenames from the filename_queue.
    #No header or footer in the CIFAR-10 format, so we leave header_bytes and
    #footer_bytes at their default of 0.

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes) # tf.F~ 로 정해진 길이로 parsing하는 클래스 선언
    result.key, value = reader.read(filename_queue) # tf.F~의 read func이용하여 key와 value 추출

    # Convert form a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8) # sting data(value)를 uint로 decode시킴

    # The first bytes represent the label, which we convert from uint8->int32.
    # binary data에서 label 만 따로 tf.strided_slice() function을 이용해서 떼내고, tf.cast를 이용해서 int32로 형변환
    result.label = tf.cast(tf.strided_slice(record_bytes, [0],[label_bytes]), tf.int32)

    #The remaining bytes after the label represent the image, which w reshape
    #from [depth * height * width] to [depth, height, width].
    # record_bytes에서 image부분을 tf.strided_slice()로 떼어내고, 3072-> 3x32x32(depth, height, width) 로 변환(reshape)
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                         [label_bytes + image_bytes]),
        [result.depth, result.height, result.width]
    )

    #Convert from [depth, height, width] to [height, width, depth].
    # 위의 [depth, height, width]를 tf.transpose()를 이용해서 [height, width, depth]로 변환 (필터 맞춰주기 위해서)
    result.uint8image = tf.transpose(depth_major, [1,2,0])

    # result class 내에 result.key, height, width, depth, uint8image가 모두 저장되어있다
    return result

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels

    :param image: 3-D Tensor of [height, width, 3] of type.float32
    :param label: 1-D Tensor of type.int32
    :param min_queue_examples: int32, minimum number of samples to retain in the queue
        that provides of batches of examples.
    :param batch_size: number of images per batch
    :param shuffle: boolean indicating whether to use a shuffling queue

    :return:
        images: Images. 4D Tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """

    #Create a queue that shuffles the examples, and then
    #read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    # shuffle 할 건지, 안 할 건지 다른 function으로 구현
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples+3*batch_size,
            min_after_dequeue=min_queue_examples
        )
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples+3*batch_size
        )

    #Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, tf.reshape(label_batch,[batch_size])

def distorted_inputs(data_dir, batch_size):
    """Construct distorted input for CIFAR training using the Reader ops.

    :param data_dir: path to the CIFAR-10 data directory
    :param batch_size: Number of images per batch.

    :return:
        images : Images. 4-D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels : Labels. 1-D tensor of [batch_size] size
    """
    # filenames 배열 안에 filename들 저장
    filenames = [os.path.join(data_dir,'data_batch_%d.bin'%i) # os.path.join : 각 os에 맞게 경로를 합쳐 줌
                 for i in range(1, 6)] # cifar-10-binary.tar.gz 압축 풀면 안에 data_batch_i.bin이 1~5까지 있음
    # 모든 cifar-10-binary*.bin 파일이 존재하는지 체크
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: '+f)

    # Create a queue that produces the filenames to read
    # filenames 배열 안에 있는 filename들을 queue로 생성한다
    filename_queue = tf.train.string_input_producer(filenames)

    #Read examples from files in the filename queue.
    # read_cifar10() function사용해서 image, label 불러와서 이미지만 float32로 형변환
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    # data augment를 위해 32x32x3에서 24x24x3만 tf.random_crop을 통해 추출해낸다.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # Randomly flip the image horizontally.
    # 랜덤하게 왼쪽 오른쪽 플립시켜 줌 (data augment)
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # NOTE: since per_image_standardization zeros the mean and makes
    # the stddev unit, this likely has no effect see tensorflow#1458

    distorted_image= tf.image.random_brightness(distorted_image,max_delta=63)
    distorted_image= tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    # image standardization 시켜 줌 범위는 약 -3~3정도 되려나
    float_image = tf.image.per_image_standardization(distorted_image)

    # Set the shapes of tensors.
    # float_image, read_input은 tf.Tensor 클래스, set_shape 메서드가 tf.Tensor 클래스 내에 존재
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    #Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print("Filling queue with %d CIFAR images before starting to train."
          "This will take a few minutes"%min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size, shuffle=True)

def inputs(eval_data, data_dir, batch_size):
    """
    Construct input for CIFAR evaluation using the Reader ops.

    :param eval_data: bool, indicating if one should use the train or eval data set.
    :param data_dir: Path to the CIFAR-10 data directory.
    :param batch_size: Number of images per batch.

    :return:
        images: Images. 4-D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1-D tensor of [batch_size] size
    """
    # distorted_input() 과는 다르게 inputs()는 evaluation data input에도 쓰일 수 있으므로 eval_data(bool)을 통해 구분
    if not eval_data :
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin'%i)
                     for i in range(1,6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: '+f)

    # 나머지는 distorted_input과 거의 비슷... (data augment부분 제외하고)
    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,height,width)


    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch*min_fraction_of_examples_in_queue)

    #Generate a batch of images and labels by building up a queue
    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size,
                                           shuffle=False)
