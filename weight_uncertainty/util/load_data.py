import numpy as np
from os.path import join, exists
from weight_uncertainty import conf
import pickle
from mnist import MNIST
from random import random
from scipy.ndimage.filters import gaussian_filter


def unpickle(file):
 '''Load byte data from file'''
 with open(file, 'rb') as f:
  data = pickle.load(f, encoding='latin-1')
  return data


class DataloaderBase:
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


class DataloaderUCR(DataloaderBase):
    def __init__(self, direc, dataset, ratio=np.array([0.8, 0.9]), augment=False):
        data_dir = join(direc, dataset)
        assert exists(data_dir), f'Not found datadir {data_dir}'
        self.augment = augment

        if isinstance(ratio, (list, tuple)):
            ratio = np.array(ratio)

        data_train = np.loadtxt(join(data_dir, dataset) + '_TRAIN', delimiter=',')
        data_test_val = np.loadtxt(join(data_dir, dataset) + '_TEST', delimiter=',')

        DATA = np.concatenate((data_train, data_test_val), axis=0)
        N = DATA.shape[0]
        conf.num_samples = N

        ratio = (ratio * N).astype(np.int32)
        ind = np.random.permutation(N)

        self.data = {}
        self.data['X_train'] = DATA[ind[:ratio[0]], 1:]
        self.data['X_val'] = DATA[ind[ratio[0]:ratio[1]], 1:]
        self.data['X_test'] = DATA[ind[ratio[1]:], 1:]
        # Targets have labels 1-indexed. We subtract one for 0-indexed
        self.data['y_train'] = DATA[ind[:ratio[0]], 0] - 1
        self.data['y_val'] = DATA[ind[ratio[0]:ratio[1]], 0] - 1
        self.data['y_test'] = DATA[ind[ratio[1]:], 0] - 1


class DataLoaderCIFAR(DataloaderBase):
    def __init__(self, data_dir, augment=False):
        '''Return train_data, train_labels, test_data, test_labels
         The shape of data is 32 x 32 x3'''
        train_data = None
        train_labels = []
        self.augment = augment

        for i in range(1, 6):
            data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
            if i == 1:
                train_data = data_dic['data']
            else:
                train_data = np.vstack((train_data, data_dic['data']))
            train_labels += data_dic['labels']

        test_data_dic = unpickle(data_dir + "/test_batch")
        test_data = test_data_dic['data']

        train_data = train_data.reshape((len(train_data), 3, 32, 32))
        train_data = np.rollaxis(train_data, 1, 4)
        train_labels = np.array(train_labels)

        test_data = test_data.reshape((len(test_data), 3, 32, 32))
        test_data = np.rollaxis(test_data, 1, 4)

        # Split the train data
        N = train_data.shape[0]
        ratio = int(0.8*N)
        ind = np.random.permutation(N)

        self.data = {}
        self.data['X_train'] = (train_data[ind[:ratio]].astype(np.float32) - 120.) / 64.
        self.data['X_val'] = (train_data[ind[ratio:]].astype(np.float32) - 120.) / 64.
        self.data['X_test'] = (test_data.astype(np.float32) - 120.) / 64.
        # Targets have labels 1-indexed. We subtract one for 0-indexed
        self.data['y_train'] = train_labels[ind[:ratio]]
        self.data['y_val'] = train_labels[ind[ratio:]]
        self.data['y_test'] = np.array(test_data_dic['labels'])


class DataLoaderMNIST(DataloaderBase):
    def __init__(self, data_dir, augment=False):
        mndata = MNIST(data_dir)
        self.data = {}
        self.augment = augment

        # train data
        images, labels = mndata.load_training()
        images = np.reshape(self.normalize(np.array(images)), newshape=[-1, 28, 28, 1])
        labels = np.array(labels).astype(np.int64)

        # Split the train data
        N = images.shape[0]
        ratio = int(0.8 * N)
        ind = np.random.permutation(N)

        self.data['X_train'] = images[ind[:ratio]]
        self.data['y_train'] = labels[ind[:ratio]]

        self.data['X_val'] = images[ind[ratio:]]
        self.data['y_val'] = labels[ind[ratio:]]

        # test data
        images, labels = mndata.load_testing()
        images = np.reshape(self.normalize(np.array(images)), newshape=[-1, 28, 28, 1])
        labels = np.array(labels).astype(np.int64)

        self.data['X_test'] = images
        self.data['y_test'] = labels

    @staticmethod
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

    @staticmethod
    def augment_batch(X):
        assert len(X.shape) == 4,  'we expect a 4D array of [num_images, height, width, num_channels]'
        if random() > 0.5:
            return X

        random_num = random()
        if random_num < 0.33:
            # shift pixels inner
            x, y = np.random.randint(1, 5, size=(2,))
            X_out = np.copy(X)
            X_out[:, x:, y:] = X[:, :-x, :-y]
            return X_out
        elif random_num < 0.66:
            # shift pixels outer
            x, y = np.random.randint(1, 5, size=(2,))
            X_out = np.copy(X)
            X_out[:, :-x, :-y] = X[:, x:, y:]
            return X_out
        else:
            X_out = np.copy(X)
            for n in range(X.shape[0]):
                X_out[n, :, :, 0] = gaussian_filter(X[n, :, :, 0], sigma=1, order=0)
            return X_out


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.ndimage.filters import gaussian_filter
    dl = DataLoaderMNIST('/home/rob/Dropbox/ml_projects/bayes_nn/bayes_nn/data/raw')
    x, y = dl.sample()
    plt.imshow(np.squeeze(x[0]), cmap='gray')
    plt.show()