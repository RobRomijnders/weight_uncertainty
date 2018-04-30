from weight_uncertainty.util.util import MixturePrior, make_train_op
from weight_uncertainty.util.util_layers import BayesianLSTMCell, BayesianConvCell, SoftmaxLayer
import tensorflow as tf
from weight_uncertainty import conf
import numpy as np


class TSCModel(object):
    def __init__(self, num_classes, size_sample):
        # Set up the placeholders
        self.x_placeholder = tf.placeholder(tf.float32, [None] + list(size_sample), name='input')
        self.y_placeholder = tf.placeholder(tf.int32, [None, ], name='target')

        # Instantiate a prior over the weights
        self.prior = MixturePrior(conf.sigma_prior)

        self.is_time_series = len(size_sample) == 1
        use_rnn = False

        self.layers = []  # Store the parameters of each layer, so we can compute the cost function later
        if use_rnn:
            outputs = self.add_RNN()
        else:
            outputs = self.add_CNN()

        logits = self.softmax_layer(outputs, num_classes)

        self.predictions = tf.nn.softmax(logits, name='predictions')

        # Classification loss
        class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.y_placeholder)
        self.loss = tf.identity(tf.reduce_mean(class_loss), name='classification_loss')

        # KL loss
        # Sum the KL losses of each layer in layers
        self.kl_loss = 0.0
        for layer in self.layers:
            self.kl_loss += layer.get_kl()

        # Weigh the kl loss across all the batches
        # See equation 9 in
        # Weight uncertainty in neural networks
        # https://arxiv.org/abs/1505.05424
        num_batches = conf.max_steps  # Make explicit that this represents the number of batches
        pi = 1./num_batches
        pi = 1/10.
        total_loss = self.loss + pi*self.kl_loss

        # Set up the optimizer
        tvars = tf.trainable_variables()
        shapes = [tvar.get_shape() for tvar in tvars]
        print("# params: %d" % np.sum([np.prod(s) for s in shapes]))

        # Clip the gradients if desired
        grads = tf.gradients(total_loss, tvars)
        # for grad, tvar in zip(grads, tvars):
        #     name = str(tvar.name).replace(':', '_')
        #     if 'mask' in name:
        #         continue
        #
        #     tf.summary.histogram(name + '_var', tvar)
        #     try:
        #         tf.summary.histogram(name + '_grad', grad)
        #     except ValueError as e:
        #         print(name)
        #         raise Exception(e)
        if conf.clip_norm > 0.0:
            grads, grads_norm = tf.clip_by_global_norm(grads, conf.clip_norm)
        else:
            grads_norm = tf.global_norm(grads)

        self.train_op = make_train_op(conf.optimizer_name, grads, tvars)

        # Calculate accuracy
        decisions = tf.argmax(logits, axis=1, output_type=tf.int32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(decisions, self.y_placeholder), tf.float32), name='accuracy')

        self.add_tensorboard_summaries(grads_norm)

        # Calculate total number of bits
        self.total_bits = tf.constant(0.0, dtype=tf.float32)
        sigma_collection = tf.get_collection('all_sigma')
        for var in sigma_collection:
            self.total_bits += tf.reduce_mean(var)
        # Total bits is the -log of the average standard deviation
        self.total_bits = -tf.log(self.total_bits/float(len(sigma_collection))) / tf.log(2.)
        tf.summary.scalar('Total bits', self.total_bits)

        # Add the pruning ops
        self.add_pruning_snr()

        # Final Tensorflow bookkeeping
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
        for _ in range(conf.num_layers):
            lstm_cell = BayesianLSTMCell(conf.hidden_dim, self.prior,
                                         forget_bias=1.0, state_is_tuple=True, layer_norm=False)
            self.layers.append(lstm_cell)

        cell = tf.nn.rnn_cell.MultiRNNCell(self.layers, state_is_tuple=True)

        # Make the RNN
        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.static_rnn(cell, inputs,
                                              dtype=tf.float32)
        outputs = outputs[-1]  # Perform classification on the final state
        return outputs

    def add_CNN(self):
        if self.is_time_series:
            inputs = tf.expand_dims(tf.expand_dims(self.x_placeholder, axis=2), axis=3)
        else:
            inputs = self.x_placeholder

        filter_shape = [6, 1] if self.is_time_series else [6, 6]

        # First layer
        conv_layer1 = BayesianConvCell('conv1', num_filters=80, filter_shape=filter_shape, stride=3, prior=self.prior,
                                         activation=tf.nn.selu)
        hidden1 = conv_layer1(inputs)
        self.layers.append(conv_layer1)
        tf.summary.histogram('Layer1', hidden1, family='activations')

        # Second layer
        conv_layer2 = BayesianConvCell('conv2', num_filters=80, filter_shape=filter_shape, stride=3, prior=self.prior,
                                         activation=tf.nn.selu)
        hidden2 = conv_layer2(hidden1)
        self.layers.append(conv_layer2)

        # Third layer
        conv_layer3 = BayesianConvCell('conv3', num_filters=80, filter_shape=filter_shape, stride=3, prior=self.prior,
                                         activation=tf.nn.selu)
        hidden3 = conv_layer3(hidden2)
        self.layers.append(conv_layer3)

        h3_shape = hidden3.shape
        outputs = tf.reshape(hidden3, shape=[-1, h3_shape[1:].num_elements()])
        return outputs

    def softmax_layer(self, outputs, num_classes):
        # Final output mapping to num_classes
        softmaxlayer = SoftmaxLayer(num_classes, self.prior)
        self.layers.append(softmaxlayer)
        return softmaxlayer(outputs)

    def add_tensorboard_summaries(self, grads_norm=0.0):
        # Summaries for TensorBoard
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("kl_loss", self.kl_loss)
            tf.summary.scalar("grads_norm", grads_norm)
            tf.summary.scalar("accuracy", self.accuracy)

        self.all_sigma = tf.concat([tf.reshape(s, [-1]) for s in tf.get_collection('all_sigma')], axis=0)
        tf.summary.histogram('Sigmas', self.all_sigma, family='sigmas')

        self.all_SNR = tf.concat([tf.reshape(tf.abs(mean)/sig, [-1]) for mean, sig in zip(tf.get_collection('random_mean'),
                                                                       tf.get_collection('all_sigma'))],
                        axis=0)
        tf.summary.histogram('snr', self.all_SNR, family='SNR')

    def add_pruning_snr(self):
        self.prune_threshold = tf.placeholder(tf.float32, name='prune_threshold')

        prune_op_list = []
        mask_ratios = []
        for mean, sigma, mask_ref in zip(tf.get_collection('random_mean'),
                                         tf.get_collection('all_sigma'),
                                         tf.get_collection('masks')):
            log_p_zero = -0.5 * tf.square(mean/sigma) - tf.log(tf.sqrt(2*np.pi)*sigma)
            mask = tf.cast(tf.less_equal(log_p_zero, self.prune_threshold), tf.float32)
            mask_ratios.append(tf.reduce_mean(mask))
            prune_op_list.append(tf.assign(mask_ref, mask))
        self.prune_ratio = tf.reduce_mean(mask_ratios, name='prune_ratio')
        self.prune_op = tf.group(prune_op_list, name='prune_op')

    def add_to_collections(self):
        for var in [self.x_placeholder,
                    self.y_placeholder,
                    self.predictions,
                    self.loss,
                    self.accuracy,
                    self.prune_op,
                    self.prune_threshold,
                    self.prune_ratio]:
            tf.add_to_collection('restore_vars', var)
