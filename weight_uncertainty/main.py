from weight_uncertainty.util.load_data import Dataloader
from weight_uncertainty.util.util_plot import plot_all_snr, plot_ucr
from weight_uncertainty.util.util import print_validation_performance
from weight_uncertainty.util.model import Model
from weight_uncertainty import conf
import tensorflow as tf
from os.path import join


def train(model, dataloader):
    sess = tf.get_default_session()
    train_writer = tf.summary.FileWriter(conf.log_direc)
    try:  # To try-finally, in case we run into a NaN error or when we make keyboard interrupt
        for step in range(conf.max_steps):
            x, y = dataloader.sample()
            *train_losses, _ = sess.run([model.loss,
                                         model.kl_loss,
                                         model.train_op],
                                        feed_dict={model.x_placeholder: x, model.y_placeholder: y})

            if step % 100 == 0:  # every 100 steps, print performances
                print_validation_performance(step, model, dataloader, train_writer, *train_losses)
    finally:
        model.saver.save(sess, join(conf.log_direc, 'save/my-model'))


def main(dataloader):
    with tf.Session() as sess:
        # Make a model
        model = Model(dataloader.num_classes, dataloader.size_sample)
        sess.run(model.init_op)

        do_train = True
        if do_train:  # Set to false if you only want to plot the p_zero's
            print('Start training')
            train(model, dataloader)
        else:
            model.saver.restore(sess, conf.restore_direc)
            x, y = dataloader.sample()
            loss_test, kl_loss_test, acc_test = sess.run([model.loss, model.kl_loss, model.accuracy],
                                                         feed_dict={model.x_placeholder: x, model.y_placeholder: y})
            print(f'For testing, loss {loss_test:5.3f} kl loss {kl_loss_test:5.3f} and accuracy {acc_test:5.3f}')


if __name__ == '__main__':
    dl = Dataloader(augment=True)
    do_plot = False
    if do_plot and dl.is_time_series():
            plot_ucr(dl.sample('train'))

    # Print data set sizes
    print(f'train set {dl.data["X_train"].shape[0]}, '
          f'val set {dl.data["X_val"].shape[0]}, '
          f'test set {dl.data["X_test"].shape[0]} ')

    main(dl)
