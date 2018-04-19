from weight_uncertainty.util.load_data import DataloaderUCR, DataLoaderCIFAR, DataLoaderMNIST
from weight_uncertainty.util.util_plot import plot_all_snr, plot_ucr
from weight_uncertainty.util.util import print_validation_performance
from weight_uncertainty.util.model import TSCModel
from weight_uncertainty import conf

import tensorflow as tf
from os.path import join


def train(model, dataloader):
    sess = tf.get_default_session()
    train_writer = tf.summary.FileWriter(conf.log_direc)
    try:
        for step in range(conf.max_steps):
            x, y = dataloader.sample()
            *train_losses, _ = sess.run([model.loss,
                                         model.kl_loss,
                                         model.train_op],
                                        feed_dict={model.x_placeholder: x, model.y_placeholder: y})

            if step % 50 == 0:
                print_validation_performance(step, model, dataloader, train_writer, *train_losses)
    finally:
        model.saver.save(sess, join(conf.log_direc, 'save/my-model'))


def main(dataloader):
    with tf.Session() as sess:
        model = TSCModel(dataloader.num_classes, dataloader.size_sample)
        sess.run(model.init_op)

        do_train = True
        if do_train:  # Set to false if you only want to plot the SNR's
            print('Start training')
            train(model, dataloader)
        else:
            model.saver.restore(sess, conf.restore_direc)
            x, y = dataloader.sample()
            loss_test, kl_loss_test, acc_test = sess.run([model.loss, model.kl_loss, model.accuracy],
                                                         feed_dict={model.x_placeholder: x, model.y_placeholder: y})
            print(f'For testing, loss {loss_test:5.3f} kl loss {kl_loss_test:5.3f} and accuracy {acc_test:5.3f}')

        # Plot a histrogram of all the SNR's
        all_snr = sess.run(model.all_SNR)
        plot_all_snr(all_snr)


if __name__ == '__main__':
    dl = DataloaderUCR(conf.data_direc_ucr, dataset='ECG5000')
    dl = DataLoaderCIFAR(conf.data_direc_cifar)
    dl = DataLoaderMNIST(conf.data_direc_mnist)
    do_plot = False
    if do_plot:
        if dl.is_time_series():
            plot_ucr(dl.sample('train'))
        else:
            import matplotlib.pyplot as plt
            plt.imshow(dl.sample()[0][2])
            plt.show()
    import time
    t1 = time.time()
    main(dl)
    print(time.time() - t1)
