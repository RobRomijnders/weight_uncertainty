import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from weight_uncertainty import conf
from weight_uncertainty.util.util import maybe_make_dir
from weight_uncertainty.util.load_data import normalize
import subprocess

mc_type = 'mc_vif'
for mutilation, var_name, _, _ in conf.experiments:
    images = np.load('log_risk/%s.%s.im.npy' % (mutilation, mc_type))
    risks = np.load('log_risk/%s.%s.risks.npy' % (mutilation, mc_type))
    mean_risks = np.mean(risks, axis=-1)
    output_dir = f'im/{conf.dataset}/{mutilation}'
    maybe_make_dir(output_dir)  # Make dir to save images

    num_experiments, num_batch = images.shape[:2]

    num_rows = 4

    for num_experiment in range(num_experiments):
        # if num_experiment > 2: break
        # if num_experiment % 2 == 0: continue
        f, axarr = plt.subplots(num_rows, 3, figsize=(15, 15))

        batch_count = 0
        for num_row in range(num_rows):
            if conf.dataset == 'mnist':
                axarr[num_row, 0].imshow(np.squeeze(images[num_experiment, batch_count]), cmap='gray')
            elif conf.dataset == 'cifar':
                axarr[num_row, 0].imshow(normalize(images[num_experiment, batch_count], reverse=True).astype(np.uint8))
            else:
                assert False
            color = 'g' if risks[num_experiment, 5, batch_count].astype(np.bool) else 'r'
            axarr[num_row, 0].set_title('Entropy %5.3f' % risks[num_experiment, 1, batch_count], color=color)

            axarr[num_row, 1].imshow(np.ones((28, 28)) * risks[num_experiment, 1, batch_count], cmap='coolwarm', vmin=0.0, vmax=1.6)
            axarr[num_row, 1].set_title(f'Entropy {risks[num_experiment, 1, batch_count]:7.2f}')
            axarr[num_row, 2].imshow(np.ones((28, 28)) * risks[num_experiment, 2, batch_count], cmap='coolwarm', vmin=0.0, vmax=1.6)
            axarr[num_row, 2].set_title(f'Mutual information{risks[num_experiment, 2, batch_count]:7.3f}')
            batch_count += 1

        for axrow in axarr:
            for ax in axrow:
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
        f.suptitle('%s %3.3f mean entropy %5.3f' %
                   (var_name, mean_risks[num_experiment, 0], mean_risks[num_experiment, 1]))
        plt.subplots_adjust(wspace=0.1, hspace=0.5)
        plt.savefig(join(output_dir, 'experiment%03i.png' % num_experiment))
        plt.close("all")
        print('Mutilation %s experiment %03i' % (mutilation, num_experiment))

    print('Also make GIF')
    subprocess.call(['convert', '-delay', '40', '-loop', '0', '*.png', f'{mutilation}_uncertain.gif'],
                    cwd=output_dir)

