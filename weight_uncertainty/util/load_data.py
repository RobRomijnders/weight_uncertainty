import numpy as np
from os.path import join, exists
from weight_uncertainty import conf
import pickle
from mnist import MNIST
from random import random
from scipy.ndimage.filters import gaussian_filter


def unpickle(file):
    """
    Load byte data from file
    :param file:
    :return:
    """
    with open(file, "rb") as f:
        data = pickle.load(f, encoding="latin-1")
        return data


def load_mnist():
    """
    Load the MNIST data set
    :return:
    """
    mndata = MNIST(conf.data_direc)
    data = {}

    # train data
    images, labels = mndata.load_training()
    images = np.reshape(normalize(np.array(images)), newshape=[-1, 28, 28, 1])
    labels = np.array(labels).astype(np.int64)

    # Split the train data into a train and val set
    N = images.shape[0]
    ratio = int(0.8 * N)
    ind = np.random.permutation(N)

    data["X_train"] = images[ind[:ratio]]
    data["y_train"] = labels[ind[:ratio]]

    data["X_val"] = images[ind[ratio:]]
    data["y_val"] = labels[ind[ratio:]]

    # test data
    images, labels = mndata.load_testing()
    images = np.reshape(normalize(np.array(images)), newshape=[-1, 28, 28, 1])
    labels = np.array(labels).astype(np.int64)

    data["X_test"] = images
    data["y_test"] = labels
    return data


def load_cifar():
    """
    Loads and preprocesses the CIFAR-10 dataset.

        This method loads the CIFAR-10 dataset from disk, splits it into training,
        validation, and testing sets, and normalizes the image data.

        Args:
            None

        Returns:
            dict: A dictionary containing the preprocessed training, validation, and
                testing data. The dictionary has the following keys:
                'X_train': Training images (normalized).
                'X_val': Validation images (normalized).
                'X_test': Testing images (normalized).
                'y_train': Training labels.
                'y_val': Validation labels.
                'y_test': Testing labels.
    """
    train_data = None
    train_labels = []

    for i in range(1, 6):
        data_dic = unpickle(conf.data_direc + "/data_batch_{}".format(i))
        if i == 1:
            train_data = data_dic["data"]
        else:
            train_data = np.vstack((train_data, data_dic["data"]))
        train_labels += data_dic["labels"]

    test_data_dic = unpickle(conf.data_direc + "/test_batch")
    test_data = test_data_dic["data"]

    train_data = train_data.reshape((len(train_data), 3, 32, 32))
    train_data = np.rollaxis(train_data, 1, 4)
    train_labels = np.array(train_labels)

    test_data = test_data.reshape((len(test_data), 3, 32, 32))
    test_data = np.rollaxis(test_data, 1, 4)

    # Split the train data into a train and val set
    N = train_data.shape[0]
    ratio = int(0.8 * N)
    ind = np.random.permutation(N)

    data = dict()
    data["X_train"] = normalize(train_data[ind[:ratio]].astype(np.float32))
    data["X_val"] = normalize(train_data[ind[ratio:]].astype(np.float32))
    data["X_test"] = normalize(test_data.astype(np.float32))
    # Targets have labels 1-indexed. We subtract one for 0-indexed
    data["y_train"] = train_labels[ind[:ratio]]
    data["y_val"] = train_labels[ind[ratio:]]
    data["y_test"] = np.array(test_data_dic["labels"])
    return data


def load_ucr(dataset_subname="ECG5000"):
    """
    Loads and prepares a UCR time series dataset.

        Args:
            dataset_subname: The name of the UCR dataset subdirectory.
                Defaults to 'ECG5000'.

        Returns:
            dict: A dictionary containing the training, validation, and test sets.
                The dictionary has the following keys:
                'X_train': Training features.
                'X_val': Validation features.
                'X_test': Test features.
                'y_train': Training labels.
                'y_val': Validation labels.
                'y_test': Test labels.
    """
    data_dir = join(conf.data_direc, dataset_subname)
    assert exists(data_dir), f"Not found datadir {data_dir}"

    data_train = np.loadtxt(join(data_dir, dataset_subname) + "_TRAIN", delimiter=",")
    data_test = np.loadtxt(join(data_dir, dataset_subname) + "_TEST", delimiter=",")

    N = data_train.shape[0]

    ratio = int(0.8 * N)
    ind = np.random.permutation(N)

    data = dict()
    data["X_train"] = data_train[ind[:ratio], 1:]
    data["X_val"] = data_train[ind[ratio:], 1:]
    data["X_test"] = data_test[:, 1:]
    # Targets have labels 1-indexed. We subtract one for 0-indexed
    data["y_train"] = data_train[ind[:ratio], 0] - 1
    data["y_val"] = data_train[ind[ratio:], 0] - 1
    data["y_test"] = data_test[:, 0] - 1
    return data


class Dataloader:
    """
    Loads and prepares data for training, validation, and testing."""

    def __init__(self, augment=False):
        """
        Initializes the dataset loader.

            Loads the training and testing data based on the configuration settings.
            Also sets the total number of samples in the configuration.

            Args:
                augment: Whether to apply data augmentation techniques.

            Returns:
                None
        """
        self.augment = augment
        if conf.dataset == "mnist":
            self.data = load_mnist()
        elif conf.dataset == "cifar":
            self.data = load_cifar()
        elif conf.dataset == "ucr":
            self.data = load_ucr()
        else:
            assert False

        conf.num_samples = self.data["X_train"].shape[0]

    @property
    def num_classes(self):
        """
        Returns the number of unique classes in the training data.

          This method determines the number of distinct target values present
          in the 'y_train' array within the dataset.

          Args:
            None

          Returns:
            int: The number of classes represented in the training labels.
        """
        return len(np.unique(self.data["y_train"]))

    @property
    def sequence_length(self):
        """
        Returns the length of a sequence in the training data.

            This property retrieves the number of features (columns) from the
            'X_train' array within the internal 'data' dictionary, effectively
            representing the sequence length.

            Parameters:
                None

            Returns:
                int: The length of a single sequence in the training data.
        """
        return self.data["X_train"].shape[1]

    @property
    def size_sample(self):
        """
        Returns the shape of a single sample in the training data.

            This property returns the dimensions of a single instance (row)
            from the 'X_train' portion of the dataset, excluding the batch size dimension.

            Returns:
                tuple: A tuple representing the shape of a single sample
                       (e.g., (number of features)).
        """
        return self.data["X_train"].shape[1:]

    @property
    def is_time_series(self):
        """
        Checks if the data is a time series.

          Args:
            None

          Returns:
            bool: True if the data represents a time series (i.e., X_train has two dimensions),
                  False otherwise.
        """
        return len(self.data["X_train"]) == 2

    @property
    def is_image(self):
        """
        Checks if the dataset represents images.

          This method determines whether the data stored in the object corresponds to
          an image dataset by checking the dimensions of the training data. Specifically,
          it verifies if the number of dimensions in X_train is equal to 4, which is typical
          for image data (batch size, height, width, channels).

          Returns:
            bool: True if the dataset represents images, False otherwise.
        """
        return len(self.data["X_train"]) == 4

    def sample(self, dataset="train", batch_size=None):
        """
        Samples a batch of data from the specified dataset.

            Args:
                dataset (str): The dataset to sample from ('train', 'val', or 'test').
                batch_size (int): The number of samples to return in the batch.
                    If None, uses the default batch size from conf.batch_size.

            Returns:
                tuple: A tuple containing the sampled images and labels.
        """
        if batch_size is None:
            batch_size = conf.batch_size
        assert dataset in ["train", "val", "test"]

        N = self.data["X_" + dataset].shape[0]
        ind_N = np.random.choice(N, batch_size, replace=False)

        images, labels = (
            self.data["X_" + dataset][ind_N],
            self.data["y_" + dataset][ind_N],
        )
        if self.augment and dataset == "train" and not self.is_time_series:
            images = self.augment_batch(images)
        return images, labels

    @staticmethod
    def augment_batch(X):
        """
        Augments a batch of images with either spatial shifting or Gaussian blur.

            Args:
                X: The input batch of images.

            Returns:
                numpy.ndarray: The augmented batch of images.
        """
        assert (
            len(X.shape) == 4
        ), "we expect a 4D array of [num_images, height, width, num_channels]"
        if random() > 0.5:
            return X

        if random() < 0.8:
            # Shift over x axis
            x, y = np.random.randint(1, 6, size=(2,))
            X_out = np.copy(X)
            if random() < 0.5:  # Forward
                X_out[:, x:, :] = X[:, :-x, :]
            else:  # Backward
                X_out[:, :-x, :] = X[:, x:, :]
            if random() < 0.5:  # Forward
                X_out[:, :, y:] = X[:, :, :-y]
            else:  # Backward
                X_out[:, :, :-y] = X[:, :, y:]
        else:
            # Apply a Gaussian blur
            X_out = np.copy(X)
            for n in range(X.shape[0]):
                X_out[n, :, :, 0] = gaussian_filter(X[n, :, :, 0], sigma=1, order=0)
        return X_out


def normalize(data, reverse=False):
    """
    Normalizes the input data based on the dataset configuration.

        Args:
            data: The input data to be normalized.
            reverse: If True, performs inverse normalization; otherwise, normalizes.

        Returns:
            The normalized or inversely normalized data.
    """
    if conf.dataset == "cifar":
        if reverse:
            return data * 64.0 + 120.0
        else:
            return (data - 120.0) / 64.0
    elif conf.dataset == "mnist":
        if reverse:
            return data * 78.0 + 33.0
        else:
            return (data - 33.0) / 78.0
    else:
        assert False
