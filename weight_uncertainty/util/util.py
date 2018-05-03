import numpy as np
import tensorflow as tf
import os
from weight_uncertainty import conf


def get_optimizer(name):
    """
    Helper function to get the optimizer
    :param name:
    :return:
    """
    if name == "sgd":
        return tf.train.GradientDescentOptimizer
    elif name == "adam":
        return tf.train.AdamOptimizer
    else:
        assert False, 'Only "sgd" or "adam" are implemented for optimizers'


def make_train_op(optimizer_name, grads, tvars):
    """
    Make a train op.

    This specifically adds different learning rate schedules for the sigmas and for all other parameters
    :param optimizer_name:
    :param grads:
    :param tvars:
    :return:
    """
    global_step = tf.train.get_or_create_global_step()

    # Get the optimizer for all parameters except the sigmas
    learning_rate_all = tf.train.exponential_decay(conf.learning_rate, global_step, 1000, 0.99)
    optimizer_all = get_optimizer(optimizer_name)(learning_rate_all)
    tf.summary.scalar("Lr_all", learning_rate_all, family="Learning rates")

    # Get the optimizer for all the sigma parameters
    learning_rate_std = tf.train.exponential_decay(10*conf.learning_rate, global_step, 1000, 0.99)
    optimizer_std = get_optimizer(optimizer_name)(learning_rate_std)
    tf.summary.scalar("Lr_std", learning_rate_std, family="Learning rates")

    # Sort the gradients
    grad_tvar_all = []
    grad_tvar_std = []
    for grad, tvar in zip(grads, tvars):
        if 'standard_deviation' in tvar.name:
            grad_tvar_std.append((grad, tvar))
        else:
            grad_tvar_all.append((grad, tvar))

    # Only update the global step in one of the apply_gradient() calls, otherwise there are two increments per step
    train_op = tf.group(optimizer_all.apply_gradients(grad_tvar_all, global_step=global_step),
                        optimizer_std.apply_gradients(grad_tvar_std))
    return train_op


class MixturePrior(object):
    def __init__(self, sigma_prior):
        """
        Set up a scale mixture prior according to section 3.3 in
        Weight uncertainty of neural networks
        https://arxiv.org/abs/1505.05424
        :param log_sigma_prior:
        """
        self.mean = 0
        self.sigma_prior = sigma_prior

    def get_kl_divergence(self, gaussian1):
        # because the other compute_kl does log(sigma) and this is already set
        mean1, sigma1 = gaussian1
        mean2, sigma2 = self.mean, self.sigma_prior

        kl_divergence = tf.log(sigma2) - tf.log(sigma1) + ((tf.square(sigma1) +
                                                            tf.square(mean1 - mean2)) / (2 * tf.square(sigma2))) - 0.5
        return tf.reduce_mean(kl_divergence)


def print_validation_performance(step, model, dataloader, train_writer, loss_train, kl_loss_train):
    sess = tf.get_default_session()
    x, y = dataloader.sample('val')
    feed_dict = {model.x_placeholder: x, model.y_placeholder: y}
    fetch_list = [model.loss,
                  model.kl_loss,
                  model.accuracy,
                  model.total_bits,
                  model.summary_op]
    loss_val, kl_loss_val, acc_val, total_bits, summ_str = sess.run(fetches=fetch_list, feed_dict=feed_dict)
    train_writer.add_summary(summ_str, step)
    train_writer.flush()

    print(f'At step {step:6.0f}/{conf.max_steps:6.0f} Train/Val: loss {loss_train:6.3f}/{loss_val:6.3f}'
          f'KL loss {kl_loss_train:6.3f}/{kl_loss_val:6.3f} and val accuracy {acc_val:6.3f} '
          f'and total bits {total_bits:6.3f}')


class RestoredModel():
    def __init__(self, model_name):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from model_name into local graph
            saver = tf.train.import_meta_graph(model_name + '.meta', clear_devices=True)
            saver.restore(self.sess, model_name)

            # Set all the ops:
            self.input, self.target, self.prediction, self.loss, self.accuracy, \
                self.prune_op, self.prune_threshold, self.prune_ratio = tf.get_collection('restore_vars')

    def evaluate(self, x, y):
        return self.sess.run([self.loss, self.accuracy], feed_dict={self.input: x, self.target: y})

    def predict(self, x):
        if len(x.shape) == 3:
            x = np.expand_dims(x, axis=-1)
        return self.sess.run(self.prediction, feed_dict={self.input: x})

    def prune(self, threshold):
        return self.sess.run([self.prune_op, self.prune_ratio], {self.prune_threshold: threshold})[1]


def maybe_make_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def reduce_entropy(X, axis=-1):
    return -1 * np.sum(X * np.log(X+1E-12), axis=axis)


def calc_risk(preds, labels=None, weights=None):
    """
    Calculates the parameters we can possibly use to examine risk of a neural net

    :param preds:
    :param labels:
    :param weights: weights for the weighted MC estimate
    :return:
    """
    if isinstance(preds, list):
        preds = np.stack(preds)
    # preds in shape [num_runs, num_batch, num_classes]
    num_runs, num_batch = preds.shape[:2]

    if weights is not None:
        assert weights.shape[0] == num_runs
        weights *= num_runs/np.sum(weights)  # Make the weights sum to num_runs
        preds *= np.expand_dims(np.expand_dims(weights, axis=1), axis=2)

    ave_preds = np.mean(preds, 0)
    pred_class = np.argmax(ave_preds, 1)

    # entropy = np.mean(-1 * np.sum(preds * np.log(preds+1E-12), axis=-1), axis=0)
    entropy = reduce_entropy(ave_preds, -1)  # entropy of the posterior predictive
    entropy_exp = np.mean(reduce_entropy(preds, -1), axis=0)  # Expected entropy of the predictive under the parameter posterior
    mutual_info = entropy - entropy_exp  # Equation 2 of https://arxiv.org/pdf/1711.08244.pdf
    variance = np.std(preds[:, range(num_batch), pred_class], 0)
    ave_softmax = np.mean(preds[:, range(num_batch), pred_class], 0)
    if labels is not None:
        correct = np.equal(pred_class, labels)
    else:
        correct = None
    return entropy, mutual_info, variance, ave_softmax, correct
