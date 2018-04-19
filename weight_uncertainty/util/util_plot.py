import matplotlib.pyplot as plt
import numpy as np
from weight_uncertainty import conf
from os.path import join
import matplotlib.style as style
style.use('fivethirtyeight')


def plot_all_snr(all_SNR):
    if isinstance(all_SNR, list):
        all_SNR = np.concatenate([array.flatten() for array in all_SNR])

    plt.hist(all_SNR, bins=500)
    plt.xlabel('value of SNR')
    plt.ylabel('Bincount')
    plt.savefig(join(conf.log_direc, 'hist_all_snr.png'))
    plt.show()


def plot_snr_pruning(prune_results):
    prune_data = np.array(prune_results)
    plt.plot(prune_data[:, 1], prune_data[:, 2], label='Validation performance')
    plt.plot([0.0999, 0.1001], [0.0, 1.0], '-', label='10% boundary')
    plt.xlabel('Prune ratio')
    plt.xlim([0., 1.])
    plt.ylim([0., 1.05])
    plt.ylabel('Validation accuracy')
    plt.legend()
    plt.show()


def plot_ucr(X_train, y_train):
    plot_row = 3
    all_classes = np.unique(y_train)

    f, axarr = plt.subplots(plot_row, len(all_classes))
    for c in all_classes:  # Loops over classes, plot as columns
        c = int(c)
        ind = np.where(y_train == c)
        ind_plot = np.random.choice(ind[0], size=plot_row)
        for n in range(plot_row):  # Loops over rows
            axarr[n, c].plot(X_train[int(ind_plot[n]), :])
            # Only chops axes for bottom row and left column
            if not n == plot_row-1:
                plt.setp([axarr[n, c].get_xticklabels()], visible=False)
            if not c == 0:
                plt.setp([axarr[n, c].get_yticklabels()], visible=False)
    f.subplots_adjust(hspace=0)  # No horizontal space between subplots
    f.subplots_adjust(wspace=0)  # No vertical space between subplots
    plt.show()
