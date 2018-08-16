本工程文件的是TensorFlow实现基于CNN数字识别的代码
1.前期准备训练数据的准备，数据保存在当前目录MNIST_DATA中
训练数据下载：
wget https://devlab-1251520893.cos.ap-guangzhou.myqcloud.com/t10k-images-idx3-ubyte.gz
wget https://devlab-1251520893.cos.ap-guangzhou.myqcloud.com/t10k-labels-idx1-ubyte.gz
wget https://devlab-1251520893.cos.ap-guangzhou.myqcloud.com/train-images-idx3-ubyte.gz
wget https://devlab-1251520893.cos.ap-guangzhou.myqcloud.com/train-labels-idx1-ubyte.gz
2.CNN模型的构建，在当前文件夹下新建文件mnist_model.py,文件代码内容如下：
#!/usr/bin/python
# -*- coding: utf-8 -*

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def deepnn(x):

  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  #第一层卷积层，卷积核为5*5，生成32个feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #激活函数采用relu

  # 第一层池化层，下采样2.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # 第二层卷积层，卷积核为5*5，生成64个feature maps
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)#激活函数采用relu

  # 第二层池化层，下采样2.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  #第一层全连接层，将7x7x64个feature maps与1024个features全连接
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  #dropout层，训练时候随机让某些隐含层节点权重不工作
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # 第二层全连接层，1024个features和10个features全连接
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob

#卷积
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#池化
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
#权重
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#偏置
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

3.在当前目录下创建train_mnist_model.py，代码如下：
#!/usr/bin/python
# -*- coding: utf-8 -*

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import mnist_model

FLAGS = None


def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  #输入变量，mnist图片大小为28*28
  x = tf.placeholder(tf.float32, [None, 784])

  #输出变量，数字是1-10
  y_ = tf.placeholder(tf.float32, [None, 10])

  # 构建网络，输入—>第一层卷积—>第一层池化—>第二层卷积—>第二层池化—>第一层全连接—>第二层全连接
  y_conv, keep_prob = mnist_model.deepnn(x)

  #第一步对网络最后一层的输出做一个softmax，第二步将softmax输出和实际样本做一个交叉熵
  #cross_entropy返回的是向量
  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)

  #求cross_entropy向量的平均值得到交叉熵
  cross_entropy = tf.reduce_mean(cross_entropy)

  #AdamOptimizer是Adam优化算法：一个寻找全局最优点的优化算法，引入二次方梯度校验
  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  #在测试集上的精确度
  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  #将神经网络图模型保存本地，可以通过浏览器查看可视化网络结构
  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  #将训练的网络保存下来
  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})#输入是字典，表示tensorflow被feed的值
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    test_accuracy = 0
    for i in range(200):
      batch = mnist.test.next_batch(50)
      test_accuracy += accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}) / 200;

    print('test accuracy %g' % test_accuracy)

    save_path = saver.save(sess,"mnist_cnn_model.ckpt")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='./',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


然后执行python train_mnist_model.py，训练的时间较长，耐心等待。
测试完成后会输出该模型训练的结果：test accuracy 0.9849
3.测试CNN模型
先下载测试的图片
wget https://devlab-1251520893.cos.ap-guangzhou.myqcloud.com/test_num.zip






