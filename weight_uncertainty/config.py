from configparser import ConfigParser
import os
from os.path import join
import datetime


class Config:
    def __init__(self):
        self.cfg = ConfigParser()

        path = os.path.dirname(__file__)
        config_name = 'development.config.ini'
        config_path = join(path, config_name)
        if os.path.exists(config_path):
            self.cfg.read(config_path)
        else:
            path = _find_base_dir(path, config_name)
            config_path = join(path, config_name)
            self.cfg.read(config_path)

        self.log_direc_stamp = None
        return

    @property
    def batch_size(self):
        return self.cfg.getint('default', 'batch_size')

    @property
    def hidden_dim(self):
        return self.cfg.getint('default', 'hidden_dim')

    @property
    def num_layers(self):
        return self.cfg.getint('default', 'num_layers')

    @property
    def sigma_prior(self):
        return self.cfg.getfloat('default', 'sigma_prior')

    @property
    def clip_norm(self):
        return self.cfg.getfloat('default', 'clip_norm')

    @property
    def optimizer_name(self):
        return self.cfg.get('default', 'optimizer_name')

    @property
    def data_direc_ucr(self):
        return self.cfg.get('direc', 'data_direc_ucr')

    @property
    def data_direc_cifar(self):
        return self.cfg.get('direc', 'data_direc_cifar')

    @property
    def data_direc_mnist(self):
        return self.cfg.get('direc', 'data_direc_mnist')

    @property
    def restore_direc(self):
        return self.cfg.get('direc', 'restore_direc')

    @property
    def input_direc(self):
        return self.cfg.get('direc', 'input_direc')

    @property
    def restore_direc_base(self):
        return self.cfg.get('direc', 'restore_direc').rstrip('/').rstrip('save/my-model')

    @property
    def log_direc(self):
        if self.log_direc_stamp is None:
            self.log_direc_stamp = join(self.cfg.get('direc', 'log_direc'),
                                        datetime.datetime.now().strftime("%y-%m-%d__%H:%M:%S"))
        return self.log_direc_stamp

    @property
    def learning_rate(self):
        return self.cfg.getfloat('default', 'learning_rate')

    @property
    def max_steps(self):
        return self.cfg.getint('default', 'max_steps')


    # @property
    # def max_steps(self):
    #     return self.cfg.getint('default', 'max_steps')
    #
    # @property
    # def log_interval(self):
    #     return self.cfg.getint('default', 'log_interval')
    #
    # @property
    # def num_filters(self):
    #     return self.cfg.getint('default', 'num_filters')
    #
    # @property
    # def num_fc(self):
    #     return self.cfg.getint('default', 'num_fc')
    #
    # @property
    # def num_experiments(self):
    #     return self.cfg.getint('sampling', 'num_experiments')
    #
    # @property
    # def lr(self):
    #     return self.cfg.getfloat('default', 'lr')
    #
    # @property
    # def momentum(self):
    #     return self.cfg.getfloat('default', 'momentum')
    #
    # @property
    # def drop_prob(self):
    #     return self.cfg.getfloat('default', 'drop_prob')
    #
    # @property
    # def weight_decay(self):
    #     return self.cfg.getfloat('default', 'weight_decay')
    #
    # @property
    # def num_runs(self):
    #     return self.cfg.getint('sampling', 'num_runs')
    #
    # @property
    # def batch_size_test(self):
    #     return self.cfg.getint('sampling', 'batch_size_test')
    #
    # @property
    # def experiments(self):
    #     for exp in self.cfg.get('default', 'experiments').split('|'):
    #         exp = exp.split(',')
    #         yield exp[0], exp[1], float(exp[2]), float(exp[3])
    #
    # @property
    # def func2var_name(self):
    #     for func, var_name, _, _ in self.experiments:
    #         yield func, var_name
    #
    # @property
    # def burn_in(self):
    #     return self.cfg.getint('langevin', 'burn_in')
    #
    # @property
    # def sample_every(self):
    #     return self.cfg.getint('langevin', 'sample_every')


def _find_base_dir(base_path, config):
    """
    Finds the base dir for the configuration, and returns it as a complete path.
    A FileNotFoundError is raised if no configuration file could be found.

    :param base_path: The base directory
    :param config: Name of the configuration file
    :return: A full path to the configuration file
    """

    # HACK: try the base directory, and try going up until
    # a configuration file has been found.

    # try while path points to a directory
    path = os.path.abspath(base_path)
    while os.path.isdir(path):
        # try to find config file in this directory
        config_path = os.path.abspath(os.path.join(path, config))
        if os.path.isfile(config_path):
            return os.path.abspath(path)

        # try one up
        path = os.path.abspath(os.path.join(path, '../'))
        if not os.path.isdir(path) or path is '/':
            raise FileNotFoundError(
                "Configuration file '{}' not found from path '{}' (and up)".format(config, base_path))
