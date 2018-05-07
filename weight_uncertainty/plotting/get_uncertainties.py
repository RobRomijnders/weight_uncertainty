from weight_uncertainty.util.load_data import Dataloader
from weight_uncertainty.util.util import RestoredModel, maybe_make_dir, calc_risk
from weight_uncertainty import conf
import numpy as np
from weight_uncertainty.util.mutilation import *
import glob
import matplotlib.pyplot as plt
import os
from collections import OrderedDict


def main(dataloader):
    # Load the model
    model = RestoredModel(conf.restore_direc)
    mc_type = 'vif'

    maybe_make_dir('log_risk')

    # Do many runs
    im_test, lbl_test = dataloader.sample('test')

    # Double check if it has sensible performance
    lbl_pred, _ = model.predict(im_test)
    print(f'Accuracy is {np.mean(np.equal(lbl_test, np.argmax(lbl_pred, axis=1)))}')

    # Explore increasing added noise
    for mutilation_name, var_name, low_value, high_value in conf.experiments:  # Read from the experiment method
        mutilation_function = globals()[mutilation_name]

        # Accumulator variables for the risk tuples and the mutilated images
        risks = []
        all_mutilated_images = []
        for i, mutilated_value in enumerate(np.linspace(low_value, high_value, conf.num_experiments)):
            # Mutilate the image and put it to PyTorch on GPU
            mutilated_images = mutilation_function(np.copy(im_test), mutilated_value)

            # Now get the samples from the predictive distribution
            preds = model.sample_prediction(mutilated_images, conf.num_runs)

            entropy, mutual_info, variance, softmax_val, correct = calc_risk(preds, lbl_test)
            risks.append(
                (mutilated_value * np.ones_like(entropy), entropy, mutual_info, variance, softmax_val, correct))

            # Do all the printing and saving bookkeeping
            print(f'At {var_name} {mutilated_value:8.3f} entropy {np.mean(entropy):5.3f} '
                  f'and mutual info {np.mean(mutual_info):5.3f} and variance {np.mean(variance):5.3f} '
                  f'and ave softmax {np.mean(softmax_val):5.3f} and error {1.0 - np.mean(correct):5.3f}')
            all_mutilated_images.append(mutilated_images)
        np.save('log_risk/%s.mc_%s.im.npy' % (mutilation_name, mc_type), np.stack(all_mutilated_images))
        np.save('log_risk/%s.mc_%s.risks.npy' % (mutilation_name, mc_type), np.array(risks))


def plot_risks():
    maybe_make_dir('im')

    filenames = glob.glob('log_risk/*.*.risks.npy')
    assert len(filenames) > 0, 'Did not find any logs'

    var2idx = {exp[0]: i for i, exp in enumerate(conf.experiments)}  # Maps experiment names to rows in the plotting
    colors = {'mc_dropout': 'g',
              'mc_multi': 'b',
              'mc_lang': 'r',
              'mc_vif': 'm'}  # Dictionarly to map types of MC to colors for plotting
    risk_types = ['Entropy', 'mutual info', 'STD of softmax', 'Mean of softmax', 'Accuracy']
    risk_ylims = [(0.0, 2.0), (0.0, 1.0), (0.0, 0.4), (0.0, 1.0), (0.0, 1.0)]

    f, axarr = plt.subplots(len(var2idx), len(risk_types))

    for filename in filenames:
        table = np.load(filename)
        table = np.mean(table, axis=-1)
        # table = np.genfromtxt(filename, delimiter=',')

        _, name = os.path.split(filename)
        mutilation_func, mc_type, _, _ = name.split('.')

        for n in range(1, table.shape[1]):
            # axarr[var2idx[mutilation_func], n-1].scatter(table[:, 0], table[:, n], label=mc_type, c=colors[mc_type], s=5)
            axarr[var2idx[mutilation_func], n - 1].plot(table[:, 0], table[:, n], label=mc_type, c=colors[mc_type])
            # axarr[var2idx[mutilation_func], i].set_title(risk_types[i])
            axarr[var2idx[mutilation_func], n - 1].set_xlabel(dict(conf.func2var_name)[mutilation_func])
            # axarr[var2idx[mutilation_func], i].set_ylim(risk_ylims[i])

    for axrow in axarr:
        for n_ax, ax in enumerate(axrow):
            # Reduce the ticklabels
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))

            ax.set_title(risk_types[n_ax])
            # ax.set_ylim(risk_ylims[n_ax])

    # Next lines remove double entries in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.0, 1.1), loc='upper right')
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.suptitle(str(var2idx))
    # plt.savefig('im/risks.png')
    plt.show()


if __name__ == '__main__':
    dl = Dataloader(augment=False)
    main(dl)
    plot_risks()
