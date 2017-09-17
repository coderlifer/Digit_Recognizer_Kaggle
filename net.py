
from collections import namedtuple
import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages


HParams = namedtuple('HParams',
                     'data_size, batch_size, num_classes, lrn_rate, mode, weight_decay_rate')


class Net(object):
    def __init__(self, hps):
        self.hps = hps
        self.X = tf.placeholder(tf.float32, [None, 784], name='X')
        self.y = tf.placeholder(tf.int32, [None, 1], name='y')

        indices = tf.reshape(tf.range(0, self.hps.batch_size, 1), [self.hps.batch_size, 1])
        self.labels = tf.sparse_to_dense(
            tf.concat(values=[indices, self.y], axis=1),
            [self.hps.batch_size, self.hps.num_classes], 1.0, 0.0)

        self._extra_train_ops = []

    def build_graph(self):
        self.global_step = tf.contrib.framework.get_or_create_global_step()

        self._build_model()

        if self.hps.mode == 'train':
            self._build_train_op()

        self.summaries = tf.summary.merge_all()

    def _build_model(self):
        filters = [64, 64, 128, 256]
        strides = [1, 2]

        x = tf.reshape(self.X, [-1, 28, 28, 1])

        with tf.variable_scope('init'):
            x = self._conv('conv0', x, 3, 1, filters[0], strides[0])
            x = self._batch_norm('bn0', x)
            x = self._leaky_relu(x, 0.01)

        with tf.variable_scope('unit_1'):
            x = self._conv('conv1', x, 3, filters[0], filters[1], strides[0])
            x = self._batch_norm('bn1', x)
            x = self._leaky_relu(x, 0.01)
            x = self._maxpool(x, 2, 2)

        with tf.variable_scope('unit_2'):
            x = self._conv('conv2', x, 3, filters[1], filters[2], strides[0])
            x = self._batch_norm('bn2', x)
            x = self._leaky_relu(x, 0.01)

        with tf.variable_scope('unit_3'):
            x = self._conv('conv3', x, 3, filters[2], filters[3], strides[0])
            x = self._batch_norm('bn3', x)
            x = self._leaky_relu(x, 0.01)
            x = self._maxpool(x, 2, 2)
            # x = self._global_avg_pool(x)

        with tf.variable_scope('logits'):
            x = self._fully_connected('fc1', x, 7*7*256, 1024)
            logits = self._fully_connected('fc2', x, 1024, self.hps.num_classes)
            self.predictions = tf.nn.softmax(logits)

        with tf.variable_scope('costs'):
            xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
            self.cost = tf.reduce_mean(xent, name='xent')
            self.cost += self._decay()

            tf.summary.scalar('cost', self.cost)

    def _build_train_op(self):
        """Build training specific ops for the graph."""
        self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
        tf.summary.scalar('learning_rate', self.lrn_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lrn_rate)
        apply_op = optimizer.minimize(self.cost,
                                      global_step=self.global_step,
                                      name='train_step')
        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

    def _decay(self):
        """L2 weight decay loss."""

        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
                # tf.summary.histogram(var.op.name, var)

        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        with tf.variable_scope(name):
            # n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(name='DW',
                                     shape=[filter_size, filter_size, in_filters, out_filters],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())
            return tf.nn.conv2d(x,
                                filter=kernel,
                                strides=[1, strides, strides, 1],
                                padding='SAME',
                                name='conv2d')

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            if self.hps.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)
            # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            x_bn = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            x_bn.set_shape(x.get_shape())

            return x_bn

    def _leaky_relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _maxpool(self, x, ksize, strides):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, strides, strides, 1],
                              padding='SAME',
                              name='max_pool')

    def _fully_connected(self, name, x, in_dim, out_dim):
        with tf.variable_scope(name):
            x = tf.reshape(x, [self.hps.batch_size, -1])

            w = tf.get_variable(name='DW',
                                shape=[in_dim, out_dim],
                                dtype=tf.float32,
                                initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
            b = tf.get_variable(name='bias',
                                shape=[out_dim],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])
