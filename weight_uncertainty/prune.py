from weight_uncertainty.util.load_data import Dataloader
from weight_uncertainty.util.util_plot import plot_pruning
from weight_uncertainty.util.util import RestoredModel
from weight_uncertainty import conf

import tensorflow as tf
import numpy as np


def main(dataloader):
    with tf.Session():
        # Load our model
        restored_model = RestoredModel(conf.restore_direc)

        # We save the results in the results list
        prune_results = []
        count = 0
        prune_ratio = 1.0
        threshold = 1.5

        # Loop over thresholds for the p_zero
        # while prune_ratio > 0.08 and count < 300:
        for threshold in [-10000, -1000, -500, -200, -100, -50, -20] + np.linspace(-10, 3.5, 10).tolist():
            prune_ratio = restored_model.prune(threshold)
            prune_ratio = 1.0 - prune_ratio  # Depends on how you interpret the "ratio"

            # The batchsize is hardcoded, so we run a couple of batches from the validation set and average them
            def test_many(num_val_batches):
                for _ in range(num_val_batches):
                    x, y = dataloader.sample(dataset='val')
                    yield restored_model.evaluate(x, y)

            # Average the performances over some number of batches
            acc_test = np.mean(np.array(list(test_many(5))))

            # Print and save to list
            print(f'For pruning at {threshold:6.3f} with ratio {prune_ratio:6.3f} '
                  f' and accuracy {acc_test:5.3f}')
            prune_results.append((threshold, prune_ratio, acc_test))

            threshold -= 0.2
            count += 1

        # and the pyplot fun :)
        plot_pruning(prune_results)


if __name__ == '__main__':
    dl = Dataloader()
    main(dl)



