from weight_uncertainty.util.load_data import DataloaderUCR, DataLoaderCIFAR, DataLoaderMNIST
import numpy as np
from weight_uncertainty.util.util_plot import plot_pruning
from weight_uncertainty.util.util import RestoredModel
import tensorflow as tf
from weight_uncertainty import conf


def main(dataloader):
    with tf.Session() as sess:
        # Load our model
        restored_model = RestoredModel(conf.restore_direc)

        # Loop over thresholds for the standard deviation of the parameters
        # We save the results in the results list
        prune_results = []
        for t in np.linspace(-5, 4, 100):
            prune_ratio = restored_model.prune(t)

            # The batchsize is hardcoded, so we run a couple of batches from the validation set and average them
            def test_many(num_val_batches):
                for _ in range(num_val_batches):
                    x, y = dataloader.sample(dataset='val')
                    yield restored_model.evaluate(x, y)

            # Average the performances over some number of batches
            loss_test, acc_test = (np.mean(result) for result in zip(*test_many(25)))

            # Print and save to list
            print(f'For pruning at {t:6.3f} with ratio {prune_ratio:6.3f}, loss {loss_test:6.3f} '
                  f' and accuracy {acc_test:5.3f}')
            prune_results.append((t, prune_ratio, acc_test))

        # and the pyplot fun :)
        plot_pruning(prune_results)


if __name__ == '__main__':
    # dl = DataloaderUCR(conf.data_direc, dataset='ECG5000')
    # dl = DataLoaderCIFAR(conf.data_direc)
    dl = DataLoaderMNIST(conf.data_direc)

    if False:
        plot_ucr(dl.sample('train'))
    main(dl)



