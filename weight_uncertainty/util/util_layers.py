import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.layers.python.layers import layers
import logging
import time
from tensorflow.python.ops.nn_ops import conv2d
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops


def get_random_normal_variable(name, prior, shape, prior_mean=0.0, dtype=tf.float32):
    """
    A wrapper around tf.get_variable which lets you get a "variable" which is
     explicitly a sample from a normal distribution.
    """

    # Inverse of a softplus function, so that the value of the standard deviation
    # will be equal to what the user specifies, but we can still enforce positivity
    # by wrapping the standard deviation in the softplus function.
    # standard_dev = tf.log(tf.exp(standard_dev) - 1.0) * tf.ones(shape)

    # it's important to initialize variances with care, otherwise the model takes too long to converge
    rho_max_init = tf.log(tf.exp(prior.sigma_prior / 5.0) - 1.0)
    rho_min_init = tf.log(tf.exp(prior.sigma_prior / 20.0) - 1.0)
    std_init = tf.random_uniform_initializer(rho_min_init, rho_max_init)

    # this is constant, original paper/email is not constant
    # initializer=tf.constant_initializer(mean)
    mean = tf.get_variable(name + "_mean", shape,
                           dtype=dtype)
    tf.add_to_collection('random_mean', mean)
    mean += prior_mean

    standard_deviation = tf.get_variable(name + "_standard_deviation", shape,
                                         initializer=std_init,
                                         dtype=dtype)
    tf.add_to_collection('random_presigma', standard_deviation)

    standard_deviation = tf.nn.softplus(standard_deviation) + 1e-5
    tf.add_to_collection('all_sigma', standard_deviation)
    tf.summary.scalar(name + '_sigma', tf.reduce_mean(standard_deviation), family='sigma')

    # Do the masking in case of pruning
    mask = tf.get_variable(name + '_mask', shape=shape, initializer=tf.ones_initializer)
    tf.add_to_collection('masks', mask)

    weights = mean + (standard_deviation * tf.random_normal(shape, 0.0, 1.0, dtype))

    return weights * tf.stop_gradient(mask), mean, standard_deviation


class SoftmaxLayer:
    def __init__(self, num_classes, prior):
        self.num_classes = num_classes
        self.prior = prior

    def get_kl(self):
        theta_kl = self.prior.get_kl_divergence((self.softmax_w_mu, self.softmax_w_std))
        theta_kl += self.prior.get_kl_divergence((self.softmax_b_mu, self.softmax_b_std))
        return theta_kl

    def __call__(self, inputs):
        tf.assert_rank(inputs, 2)
        out_dim = inputs.shape[1]
        self.softmax_w, self.softmax_w_mu, self.softmax_w_std = get_random_normal_variable("softmax_w", self.prior,
                                                                            tf.TensorShape([out_dim, self.num_classes]),
                                                                            dtype=tf.float32)
        self.softmax_b, self.softmax_b_mu, self.softmax_b_std = get_random_normal_variable("softmax_b", self.prior,
                                                                            [self.num_classes],
                                                                            dtype=tf.float32)


        logits = tf.matmul(inputs, self.softmax_w) + self.softmax_b
        return logits


class BayesianConvCell:
    def __init__(self,
                 name,
                 num_filters,
                 filter_shape,
                 stride,
                 prior,
                 activation=None):
        self.name = name
        self.num_filters = num_filters
        self.filter_shape = filter_shape
        self.prior = prior
        self.stride = stride
        self.activation = activation

    def get_kl(self):
        theta_kl = self.prior.get_kl_divergence((self.W_mu, self.W_std))
        theta_kl += self.prior.get_kl_divergence((self.b_mu, self.b_std))
        return theta_kl

    def __call__(self, inputs):
        with tf.variable_scope(self.name):
            filter_shape = tf.TensorShape(self.filter_shape + [inputs.shape[-1], self.num_filters])
            self.W, self.W_mu, self.W_std = get_random_normal_variable('conv_weight',
                                                                       self.prior,
                                                                       shape=filter_shape)
            self.b, self.b_mu, self.b_std = get_random_normal_variable('conv_bias', self.prior,
                                                                       shape=[self.num_filters])

        vert_stride = 1 if 1 in self.filter_shape else self.stride
        act = conv2d(inputs, filter=self.W, strides=[1, self.stride, vert_stride, 1], padding='SAME', data_format='NHWC') + self.b
        return self.activation(act)


class BayesianLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    """
    Implementation of Bayesian LSTM Cell from
    https://gist.github.com/windweller/500ddc19d0c3cf1eb03cf73cc6b88fe3/revisions
    """
    def __init__(self, num_units,
                 prior,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 layer_norm=True,
                 norm_gain=1.0,
                 norm_shift=0.0,
                 activation=tf.tanh):

        # once generated they stay the same across time-steps
        # must construct different cell for each layer
        self.prior = prior
        self.W, self.b = None, None

        self.W_mu, self.W_std = None, None
        self.b_mu, self.b_std = None, None

        self._layer_norm = layer_norm
        self._norm_gain = norm_gain
        self._norm_shift = norm_shift

        super(BayesianLSTMCell, self).__init__(num_units=num_units,
                                               forget_bias=forget_bias,
                                               state_is_tuple=state_is_tuple,
                                               activation=activation)  # input_size

    # we'll see if this implementation is correct
    def get_W(self, total_arg_size, output_size, dtype):
        """
        Gets the weight parameter
        On each call, a new parameter will be sampled.
        At test time, it returns the MAP
        :param total_arg_size:
        :param output_size:
        :param dtype:
        :return:
        """
        with tf.variable_scope("CellWeight"):
            if self.W is None:
                # can use its own init_scale
                self.W, self.W_mu, self.W_std = get_random_normal_variable("Matrix", self.prior,
                                                                           [total_arg_size, output_size], dtype=dtype)
        return self.W

    def get_b(self, output_size, dtype):
        """
        Gets the bias parameter
        On each call, a new parameter will be sampled.
        At test time, it returns the MAP
        :param output_size:
        :param dtype:
        :return:
        """
        with tf.variable_scope("CellBias"):
            if self.b is None:
                self.b, self.b_mu, self.b_std = get_random_normal_variable("Bias", self.prior,
                                                                           [output_size], dtype=dtype)
        return self.b

    def get_kl(self):
        """
        get the KL divergence for both the weights and the biases
        :return:
        """
        theta_kl = self.prior.get_kl_divergence((self.W_mu, self.W_std))
        theta_kl += self.prior.get_kl_divergence((self.b_mu, self.b_std))
        return theta_kl

    def _norm(self, inp, scope, dtype=dtypes.float32):
        shape = inp.get_shape()[-1:]
        gamma_init = init_ops.constant_initializer(self._norm_gain)
        beta_init = init_ops.constant_initializer(self._norm_shift)
        with vs.variable_scope(scope):
            # Initialize beta and gamma for use by layer_norm.
            vs.get_variable("gamma", shape=shape, initializer=gamma_init, dtype=dtype)
            vs.get_variable("beta", shape=shape, initializer=beta_init, dtype=dtype)
        normalized = layers.layer_norm(inp, reuse=True, scope=scope)
        return normalized

    def stochastic_linear(self, args, output_size, bias=True, bias_start=0.0, scope=None):
        # Local reparameterization trick
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape().as_list() for a in args]
        for shape in shapes:
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
            else:
                total_arg_size += shape[1]

        dtype = [a.dtype for a in args][0]

        # Now the computation.
        with tf.variable_scope(scope or "Linear"):
            matrix = self.get_W(total_arg_size, output_size, dtype=dtype)
            if len(args) == 1:
                res = tf.matmul(args[0], matrix)
            else:
                res = tf.matmul(tf.concat(args, axis=1), matrix)

            if not bias:
                return res
            else:
                bias_term = self.get_b(output_size, dtype=dtype)
                return res + bias_term

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                one = constant_op.constant(1, dtype=dtypes.int32)
                c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one) # tf.split(state, 2, axis=1
            concat = self.stochastic_linear([inputs, h], 4 * self._num_units, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(concat, 4, axis=1)
            if self._layer_norm:
                i = self._norm(i, "input", dtype=inputs.dtype)
                j = self._norm(j, "transform", dtype=inputs.dtype)
                f = self._norm(f, "forget", dtype=inputs.dtype)
                o = self._norm(o, "output", dtype=inputs.dtype)

            new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
                              tf.nn.tanh(j))
            new_h = tf.nn.tanh(new_c) * tf.sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat([new_c, new_h], axis=1)
            return new_h, new_state