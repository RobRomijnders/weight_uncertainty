from configparser import ConfigParser
import os
from os.path import join
import datetime


class Config:
    def __init__(self, dataset='UCR'):
        self.cfg = ConfigParser()

        assert dataset in ['ucr', 'cifar', 'mnist'], "Please provide a data set in ['ucr', 'cifar', 'mnist']"

        path = os.path.dirname(__file__)
        config_name = dataset + '.config.ini'
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
    def sigma_init_low(self):
        return self.cfg.getfloat('default', 'sigma_init_low')

    @property
    def sigma_init_high(self):
        return self.cfg.getfloat('default', 'sigma_init_high')

    @property
    def clip_norm(self):
        return self.cfg.getfloat('default', 'clip_norm')

    @property
    def optimizer_name(self):
        return self.cfg.get('default', 'optimizer_name')

    @property
    def data_direc(self):
        return self.cfg.get('direc', 'data_direc')

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

    @property
    def num_filters(self):
        return list(map(lambda x: int(x), self.cfg.get('default', 'num_filters').split(',')))

    def get_filter_shape(self, is_time_series):
        if is_time_series:
            return [self.cfg.getint('default', 'filter_size'), 1]
        else:
            return [self.cfg.getint('default', 'filter_size'), self.cfg.getint('default', 'filter_size')]



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
