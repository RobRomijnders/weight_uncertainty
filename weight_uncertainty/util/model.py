from weight_uncertainty.util.util import MixturePrior, make_train_op
from weight_uncertainty.util.util_layers import BayesianLSTMCell, BayesianConvCell, SoftmaxLayer
from weight_uncertainty import conf

import tensorflow as tf
import numpy as np


class Model(object):
    def __init__(self, num_classes, size_sample):
        # Set up the placeholders
        self.x_placeholder = tf.placeholder(tf.float32, [None] + list(size_sample), name='input')
        self.y_placeholder = tf.placeholder(tf.int32, [None, ], name='target')

        # Instantiate a prior over the weights
        self.prior = MixturePrior(conf.sigma_prior)

        self.is_time_series = len(size_sample) == 1

        # Make the model
        use_rnn = False
        if use_rnn:
            # Be careful, rnn's take long to train, a cnn does just as well
            outputs = self.add_RNN()
        else:
            outputs = self.add_CNN()

        # Get the logits from the outputs of the CNN or RNN
        logits = self.softmax_layer(outputs, num_classes)

        # And softmax them to get proper predictions between [0, 1]
        self.predictions = tf.nn.softmax(logits, name='predictions')

        # --- The rest is all about losses and tensorflow bookkeeping ---

        # Classification loss
        class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.y_placeholder)
        self.loss = tf.identity(tf.reduce_mean(class_loss), name='classification_loss')

        # KL loss
        # Sum the KL losses from the collection. This just simplifies the handling of loss elements
        self.kl_loss = tf.add_n(tf.get_collection('kl_losses'))

        # Weigh the kl loss across all the batches
        # See equation 9 in
        # Weight uncertainty in neural networks
        # https://arxiv.org/abs/1505.05424
        # pi = 1./num_batches
        # Both the Weight uncertainty paper and "Practical Variational Inference for Neural Networks" recommend to
        # anneal pi during training. I found that ramping up works best
        pi = ramp_and_clip(1/10000., max((1/10., conf.batch_size / conf.num_samples)), 3000, 20000, global_step=None)
        total_loss = self.loss + pi*self.kl_loss

        # Set up the optimizer
        tvars = tf.trainable_variables()
        shapes = [tvar.get_shape() for tvar in tvars]
        print("# params: %d" % np.sum([np.prod(s) for s in shapes]))

        # Clip the gradients if desired
        grads = tf.gradients(total_loss, tvars)
        if conf.clip_norm > 0.0:
            grads, grads_norm = tf.clip_by_global_norm(grads, conf.clip_norm)
        else:
            grads_norm = tf.global_norm(grads)

        # Make the final train op
        self.train_op = make_train_op(conf.optimizer_name, grads, tvars)

        # Calculate accuracy
        decisions = tf.argmax(logits, axis=1, output_type=tf.int32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(decisions, self.y_placeholder), tf.float32), name='accuracy')

        # Add summaries for tensorboard if desired
        self.add_tensorboard_summaries(grads_norm)

        # Calculate total number of bits. Useful to get some intuition for the compression
        self.total_bits = tf.constant(0.0, dtype=tf.float32)
        sigma_collection = tf.get_collection('all_sigma')
        for sigm in sigma_collection:
            self.total_bits += tf.reduce_mean(tf.log(sigm) / tf.log(2.))
        self.total_bits = tf.identity(self.total_bits, "total_bits") / len(sigma_collection)
        tf.summary.scalar('Total bits', self.total_bits)

        # Finally the usual necessary Tensorflow ops
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.add_to_collections()
        print('Finished model')

    def add_RNN(self):
        inputs = tf.unstack(tf.expand_dims(self.x_placeholder, axis=1), axis=2)
        # Stack many BayesianLSTMCells
        # Note that by this call, the epsilon is equal for all time steps
        layers = []
        for _ in range(conf.num_layers):
            lstm_cell = BayesianLSTMCell(conf.hidden_dim, self.prior,
                                         forget_bias=1.0, state_is_tuple=True, layer_norm=False)
            layers.append(lstm_cell)

        cell = tf.nn.rnn_cell.MultiRNNCell(layers, state_is_tuple=True)

        # Make the RNN
        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.static_rnn(cell, inputs,
                                              dtype=tf.float32)

        for layer in layers:
            # Add kl losses to the collection
            tf.add_to_collection('kl_losses', layer.get_kl())
        outputs = outputs[-1]  # Perform classification on the final state
        return outputs

    def add_CNN(self):
        if self.is_time_series:
            # We will still use 2D convolutions for time series. So we expand dims
            inputs = tf.expand_dims(tf.expand_dims(self.x_placeholder, axis=2), axis=3)
        else:
            inputs = self.x_placeholder

        filter_shape = conf.get_filter_shape(self.is_time_series)

        # Create the layers of the CNN in a loop :)
        for i, num_filt in enumerate(conf.num_filters):
            conv_layer = BayesianConvCell(f'conv{i}',
                                          num_filters=num_filt,
                                          filter_shape=filter_shape,
                                          stride=conf.stride,
                                          prior=self.prior,
                                          activation=tf.nn.selu)
            inputs = conv_layer(inputs)
        final_output = inputs

        # Some final reshaping
        final_output_shape = final_output.shape  # Final shape depends on the number of layers, stride and kernel size
        return tf.reshape(final_output, shape=[-1, final_output_shape[1:].num_elements()])

    def softmax_layer(self, outputs, num_classes):
        """
        Takes the outputs of a neural network and maps it to num_classes neurons
        :param outputs:
        :param num_classes:
        :return:
        """
        softmaxlayer = SoftmaxLayer(num_classes, self.prior)
        return softmaxlayer(outputs)

    def add_tensorboard_summaries(self, grads_norm=0.0):
        """
        Add some nice summaries for Tensorboard
        :param grads_norm:
        :return:
        """
        # Summaries for TensorBoard
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("kl_loss", self.kl_loss)
            tf.summary.scalar("grads_norm", grads_norm)
            tf.summary.scalar("accuracy", self.accuracy)

        self.all_sigma = tf.concat([tf.reshape(s, [-1]) for s in tf.get_collection('all_sigma')], axis=0)
        tf.summary.histogram('Sigmas', self.all_sigma, family='sigmas')

    def add_to_collections(self):
        """
        Add the variables to a collection that we will use when restoring a model

        TENSORFLOW MAGIC ...
        :return:
        """
        for var in [self.x_placeholder,
                    self.y_placeholder,
                    self.predictions,
                    self.loss,
                    self.accuracy]:
            tf.add_to_collection('restore_vars', var)


def ramp_and_clip(value_start, value_stop, step_start, step_stop, global_step=None):
    """
    Utility function to clip ramping coefficients

    Before step_start, return value_start
    Between step_start and step_stop, ramp up to value_stop
    After step_stop, return value_stop
    :param value_start:
    :param value_stop:
    :param step_start:
    :param step_stop:
    :param global_step:
    :return:
    """
    if not global_step:
        global_step = tf.train.get_or_create_global_step()
    pi = value_start + (value_stop - value_start) * \
                tf.clip_by_value(1. / (step_stop- step_start) * \
                (tf.cast(global_step, tf.float32) - step_start), 0., 1.)

    # And add a summary to tensorboard
    tf.summary.scalar('pi', pi, family='summaries')
    return pi
