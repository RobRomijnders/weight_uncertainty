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
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
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

    data['X_train'] = images[ind[:ratio]]
    data['y_train'] = labels[ind[:ratio]]

    data['X_val'] = images[ind[ratio:]]
    data['y_val'] = labels[ind[ratio:]]

    # test data
    images, labels = mndata.load_testing()
    images = np.reshape(normalize(np.array(images)), newshape=[-1, 28, 28, 1])
    labels = np.array(labels).astype(np.int64)

    data['X_test'] = images
    data['y_test'] = labels
    return data


def load_cifar():
    train_data = None
    train_labels = []

    for i in range(1, 6):
        data_dic = unpickle(conf.data_direc + "/data_batch_{}".format(i))
        if i == 1:
            train_data = data_dic['data']
        else:
            train_data = np.vstack((train_data, data_dic['data']))
        train_labels += data_dic['labels']

    test_data_dic = unpickle(conf.data_direc + "/test_batch")
    test_data = test_data_dic['data']

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
    data['X_train'] = normalize(train_data[ind[:ratio]].astype(np.float32))
    data['X_val'] = normalize(train_data[ind[ratio:]].astype(np.float32))
    data['X_test'] = normalize(test_data.astype(np.float32))
    # Targets have labels 1-indexed. We subtract one for 0-indexed
    data['y_train'] = train_labels[ind[:ratio]]
    data['y_val'] = train_labels[ind[ratio:]]
    data['y_test'] = np.array(test_data_dic['labels'])
    return data


def load_ucr(dataset_subname='ECG5000'):
    data_dir = join(conf.data_direc, dataset_subname)
    assert exists(data_dir), f'Not found datadir {data_dir}'

    data_train = np.loadtxt(join(data_dir, dataset_subname) + '_TRAIN', delimiter=',')
    data_test = np.loadtxt(join(data_dir, dataset_subname) + '_TEST', delimiter=',')

    N = data_train.shape[0]

    ratio = int(0.8 * N)
    ind = np.random.permutation(N)

    data = dict()
    data['X_train'] = data_train[ind[:ratio], 1:]
    data['X_val'] = data_train[ind[ratio:], 1:]
    data['X_test'] = data_test[:, 1:]
    # Targets have labels 1-indexed. We subtract one for 0-indexed
    data['y_train'] = data_train[ind[:ratio], 0] - 1
    data['y_val'] = data_train[ind[ratio:], 0] - 1
    data['y_test'] = data_test[:, 0] - 1
    return data


class Dataloader:
    def __init__(self, augment=False):
        self.augment = augment
        if conf.dataset == 'mnist':
            self.data = load_mnist()
        elif conf.dataset == 'cifar':
            self.data = load_cifar()
        elif conf.dataset == 'ucr':
            self.data = load_ucr()
        else:
            assert False

        conf.num_samples = self.data['X_train'].shape[0]

    @property
    def num_classes(self):
        return len(np.unique(self.data['y_train']))

    @property
    def sequence_length(self):
        return self.data['X_train'].shape[1]

    @property
    def size_sample(self):
        return self.data['X_train'].shape[1:]

    @property
    def is_time_series(self):
        return len(self.data['X_train']) == 2

    @property
    def is_image(self):
        return len(self.data['X_train']) == 4

    def sample(self, dataset='train', batch_size=None):
        if batch_size is None:
            batch_size = conf.batch_size
        assert dataset in ['train', 'val', 'test']

        N = self.data['X_' + dataset].shape[0]
        ind_N = np.random.choice(N, batch_size, replace=False)

        images, labels = self.data['X_' + dataset][ind_N], self.data['y_' + dataset][ind_N]
        if self.augment and dataset == 'train' and not self.is_time_series:
            images = self.augment_batch(images)
        return images, labels

    @staticmethod
    def augment_batch(X):
        assert len(X.shape) == 4,  'we expect a 4D array of [num_images, height, width, num_channels]'
        if random() > 0.5:
            return X

        if random() < 0.8:
            # Shift over x axis
            x, y = np.random.randint(1, 6, size=(2,))
            X_out = np.copy(X)
            if random() < 0.5: # Forward
                X_out[:, x:, :] = X[:, :-x, :]
            else:  # Backward
                X_out[:, :-x, :] = X[:, x:, :]
            if random() < 0.5: # Forward
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
    if conf.dataset == 'cifar':
        if reverse:
            return data * 64. + 120.
        else:
            return (data - 120.)/64.
    elif conf.dataset == 'mnist':
        if reverse:
            return data * 78. + 33.
        else:
            return (data - 33.) / 78.
    else:
        assert False
