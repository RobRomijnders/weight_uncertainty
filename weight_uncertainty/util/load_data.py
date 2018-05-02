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
    def normalize(images, reverse=False):
        """
        Normalize the images with fixed values
        :param images:
        :param reverse:
        :return:
        """
        mean = 33
        std = 78

        conf.range = ((0-33)/78, (255-33)/78)
        if reverse:
            return images*std + mean
        else:
            return (images-mean)/std

    mndata = MNIST(conf.data_direc)
    data = {}

    # train data
    images, labels = mndata.load_training()
    images = np.reshape(normalize(np.array(images)), newshape=[-1, 28, 28, 1])
    labels = np.array(labels).astype(np.int64)

    # Split the train data
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

    # Split the train data
    N = train_data.shape[0]
    ratio = int(0.8 * N)
    ind = np.random.permutation(N)

    data = dict()
    data['X_train'] = (train_data[ind[:ratio]].astype(np.float32) - 120.) / 64.
    data['X_val'] = (train_data[ind[ratio:]].astype(np.float32) - 120.) / 64.
    data['X_test'] = (test_data.astype(np.float32) - 120.) / 64.
    # Targets have labels 1-indexed. We subtract one for 0-indexed
    data['y_train'] = train_labels[ind[:ratio]]
    data['y_val'] = train_labels[ind[ratio:]]
    data['y_test'] = np.array(test_data_dic['labels'])
    return data


def load_ucr(dataset_subname='ECG5000', ratio=[0.8, 0.9]):
    data_dir = join(conf.data_direc, dataset_subname)
    assert exists(data_dir), f'Not found datadir {data_dir}'

    if isinstance(ratio, (list, tuple)):
        ratio = np.array(ratio)

    data_train = np.loadtxt(join(data_dir, dataset_subname) + '_TRAIN', delimiter=',')
    data_test_val = np.loadtxt(join(data_dir, dataset_subname) + '_TEST', delimiter=',')

    DATA = np.concatenate((data_train, data_test_val), axis=0)
    N = DATA.shape[0]
    conf.num_samples = N

    # TODO DROP this subsampling !!!
    ratio = (ratio * N).astype(np.int32)
    ind = np.random.permutation(N)

    data = dict()
    data['X_train'] = DATA[ind[:ratio[0]], 1:]
    data['X_val'] = DATA[ind[ratio[0]:ratio[1]], 1:]
    data['X_test'] = DATA[ind[ratio[1]:], 1:]
    # Targets have labels 1-indexed. We subtract one for 0-indexed
    data['y_train'] = DATA[ind[:ratio[0]], 0] - 1
    data['y_val'] = DATA[ind[ratio[0]:ratio[1]], 0] - 1
    data['y_test'] = DATA[ind[ratio[1]:], 0] - 1
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
        if self.augment and dataset == 'train':
            images = self.augment_batch(images)
        return images, labels

    # @staticmethod
    # def augment_batch(X):
    #     assert len(X.shape) == 4,  'we expect a 4D array of [num_images, height, width, num_channels]'
    #     if random() > 0.5:
    #         return X
    #
    #     random_num = random()
    #     if random_num < 0.33:
    #         # shift pixels inner
    #         x, y = np.random.randint(1, 5, size=(2,))
    #         X_out = np.copy(X)
    #         X_out[:, x:, y:] = X[:, :-x, :-y]
    #         return X_out
    #     elif random_num < 0.66:
    #         # shift pixels outer
    #         x, y = np.random.randint(1, 5, size=(2,))
    #         X_out = np.copy(X)
    #         X_out[:, :-x, :-y] = X[:, x:, y:]
    #         return X_out
    #     else:
    #         X_out = np.copy(X)
    #         for n in range(X.shape[0]):
    #             X_out[n, :, :, 0] = gaussian_filter(X[n, :, :, 0], sigma=1, order=0)
    #         return X_out


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dl = Dataloader()
    x, y = dl.sample()
    plt.imshow(np.squeeze(x[0]), cmap='gray')
    plt.show()
