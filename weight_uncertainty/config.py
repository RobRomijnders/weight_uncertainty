from configparser import ConfigParser
import os
from os.path import join
import datetime


class Config:
    """
    Configuration class for managing dataset-specific parameters.

    This class loads configuration settings from an INI file based on the
    specified dataset and provides access to these settings through various methods."""

    def __init__(self, dataset="UCR"):
        """
        Initializes the configuration for a given dataset.

            Loads configuration parameters from an INI file specific to the chosen
            dataset ('UCR', 'CIFAR', or 'MNIST').  If the config file is not found in
            the current directory, it searches upwards through parent directories.

            Args:
                dataset: The name of the dataset to load configuration for.
                    Defaults to 'UCR'.

            Returns:
                None
        """
        self.cfg = ConfigParser()

        assert dataset in [
            "ucr",
            "cifar",
            "mnist",
        ], "Please provide a data set in ['ucr', 'cifar', 'mnist']"
        self.dataset = dataset

        path = os.path.dirname(__file__)
        config_name = dataset + ".config.ini"
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
        """
        Returns the batch size configured in the default section of the config file.

          Args:
            None

          Returns:
            int: The batch size as an integer, read from the configuration file.
        """
        return self.cfg.getint("default", "batch_size")

    @property
    def hidden_dim(self):
        """
        Returns the dimension of the hidden layers.

            Args:
                None

            Returns:
                int: The dimension of the hidden layers as specified in the configuration file.
        """
        return self.cfg.getint("default", "hidden_dim")

    @property
    def num_layers(self):
        """
        Returns the number of layers in the model.

            Args:
                None

            Returns:
                int: The number of layers, as specified in the configuration file.
        """
        return self.cfg.getint("default", "num_layers")

    @property
    def sigma_prior(self):
        """
        Returns the value of the sigma prior.

            This method retrieves the 'sigma_prior' configuration value from the
            'default' section of the configuration file.

            Parameters:
                None

            Returns:
                float: The sigma prior value as a float.
        """
        return self.cfg.getfloat("default", "sigma_prior")

    @property
    def sigma_init_low(self):
        """
        Returns the initial low sigma value.

            This method retrieves the 'sigma_init_low' configuration parameter
            from the 'default' section of the configuration file.

            Parameters:
                None

            Returns:
                float: The initial low sigma value as a float.
        """
        return self.cfg.getfloat("default", "sigma_init_low")

    @property
    def sigma_init_high(self):
        """
        Returns the initial high sigma value.

            This method retrieves the 'sigma_init_high' configuration parameter
            from the 'default' section of the configuration file.

            Parameters:
                None

            Returns:
                float: The initial high sigma value as a float.
        """
        return self.cfg.getfloat("default", "sigma_init_high")

    @property
    def clip_norm(self):
        """
        Returns the clipping norm value.

            This method retrieves the 'clip_norm' value from the configuration file
            under the 'default' section. This value is used for gradient clipping.

            Returns:
                float: The clipping norm value as a float.
        """
        return self.cfg.getfloat("default", "clip_norm")

    @property
    def optimizer_name(self):
        """
        Returns the name of the optimizer to be used.

          This method retrieves the optimizer name from the configuration file.

          Args:
            None

          Returns:
            str: The name of the optimizer as specified in the configuration.
        """
        return self.cfg.get("default", "optimizer_name")

    @property
    def data_direc(self):
        """
        Returns the data directory.

          This method retrieves the data directory path from the configuration.
          If 'direc' is not specified in the config, it defaults to 'data_direc'.

          Args:
            None

          Returns:
            str: The path to the data directory.
        """
        return self.cfg.get("direc", "data_direc")

    @property
    def restore_direc(self):
        """
        Retrieves the directory for restoration.

            Returns:
                str: The directory path to use for restoring, defaulting to 'restore_direc'
                     if not found in the configuration.
        """
        return self.cfg.get("direc", "restore_direc")

    @property
    def input_direc(self):
        """
        Returns the input directory.

            This method retrieves the input directory path from the configuration.
            If not explicitly set in the config, it defaults to 'input_direc'.

            Parameters:
                None

            Returns:
                str: The path to the input directory.
        """
        return self.cfg.get("direc", "input_direc")

    @property
    def restore_direc(self):
        """
        Retrieves the directory for restoration.

            Returns:
                str: The directory path to use for restoring, defaulting to 'restore_direc'
                     if not found in the configuration.
        """
        return self.cfg.get("direc", "restore_direc")

    @property
    def log_direc(self):
        """
        Returns the directory for storing logs.

            Creates a timestamped subdirectory within the configured log directory if one doesn't already exist.

            Parameters:
                None

            Returns:
                str: The path to the log directory, including a timestamped subdirectory.
        """
        if self.log_direc_stamp is None:
            self.log_direc_stamp = join(
                self.cfg.get("direc", "log_direc"),
                datetime.datetime.now().strftime("%y-%m-%d__%H:%M:%S"),
            )
        return self.log_direc_stamp

    @property
    def learning_rate(self):
        """
        Returns the learning rate from the configuration file.

          Args:
            None

          Returns:
            float: The learning rate value as specified in the 'default' section of
                   the configuration file under the key 'learning_rate'.
        """
        return self.cfg.getfloat("default", "learning_rate")

    @property
    def max_steps(self):
        """
        Returns the maximum number of training steps.

            Retrieves the 'max_steps' value from the configuration file under the
            'default' section.

            Args:
                None

            Returns:
                int: The maximum number of training steps as specified in the config.
        """
        return self.cfg.getint("default", "max_steps")

    @property
    def num_filters(self):
        """
        Returns a list of the number of filters.

            Retrieves the 'num_filters' string from the configuration, splits it by commas,
            and converts each element to an integer.

            Args:
                None

            Returns:
                list[int]: A list of integers representing the number of filters.
        """
        return list(
            map(lambda x: int(x), self.cfg.get("default", "num_filters").split(","))
        )

    @property
    def stride(self):
        """
        Returns the stride value from the configuration.

          This method retrieves the 'stride' value defined in the 'default' section of
          the configuration file.

          Parameters:
            None

          Returns:
            int: The stride value as an integer.
        """
        return self.cfg.getint("default", "stride")

    def get_filter_shape(self, is_time_series):
        """
        Returns the shape of the filter based on whether it's a time series.

            Args:
                is_time_series:  A boolean indicating if the data is time series.

            Returns:
                list: A list representing the filter shape. The shape will be
                      [filter_size, 1] for time series data and [filter_size, filter_size] otherwise.
        """
        if is_time_series:
            return [self.cfg.getint("default", "filter_size"), 1]
        else:
            return [
                self.cfg.getint("default", "filter_size"),
                self.cfg.getint("default", "filter_size"),
            ]

    @property
    def func2var_name(self):
        """
        Yields function-variable name pairs from experiments.

            Iterates through the experiments and yields each function along with its
            corresponding variable name.

            Args:
                None

            Returns:
                Generator[Tuple[Callable, str]]: A generator that yields tuples of
                                                  (function, variable_name) for each
                                                  experiment.
        """
        for func, var_name, _, _ in self.experiments:
            yield func, var_name

    @property
    def num_runs(self):
        """
        Returns the number of sampling runs.

          This method retrieves the value for 'num_runs' from the configuration
          file under the 'sampling' section.

          Parameters:
            None

          Returns:
            int: The number of sampling runs as specified in the configuration file.
        """
        return self.cfg.getint("sampling", "num_runs")

    @property
    def num_experiments(self):
        """
        Returns the number of experiments to run.

          This method retrieves the value for 'num_experiments' from the
          'sampling' section of the configuration file.

          Args:
            None

          Returns:
            int: The number of experiments as specified in the configuration.
        """
        return self.cfg.getint("sampling", "num_experiments")

    @property
    def experiments(self):
        """
        Yields experiment configurations.

            Iterates through the configured experiments, splitting each into its
            constituent parts (name, description, parameter1, parameter2) and yields them.

            Args:
                None

            Returns:
                generator[tuple]: A generator that yields tuples of strings and floats representing
                                  experiment configurations in the format (name, description, param1, param2).
        """
        for exp in self.cfg.get("default", "experiments").split("|"):
            exp = exp.split(",")
            yield exp[0], exp[1], float(exp[2]), float(exp[3])


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
        path = os.path.abspath(os.path.join(path, "../"))
        if not os.path.isdir(path) or path is "/":
            raise FileNotFoundError(
                "Configuration file '{}' not found from path '{}' (and up)".format(
                    config, base_path
                )
            )
