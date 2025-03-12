import rootutils
rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)

import os
import os.path as p
from src.utils.callbacks.eval_on_dataset import compute_ensemble_metrics


def main():
    eval_root_dir = '/home/yl241/data/rainfall_eval_LiT'

    corrector_gan_dir_pattern = 'CorrectorGAN_epoch_699'
    compute_ensemble_metrics(parent_save_dir=eval_root_dir, save_dir_pattern=corrector_gan_dir_pattern,
                             ensemble_size=13)

def rebuttal():
    eval_root_dir = '/home/yl241/data/rainfall_eval_LiT_rebuttal'
    # corrector_gan_dir_pattern = 'CorrectorGAN_epoch_699'
    dir_pattern = 'logp1_emd_ckpt21'
    # dir_pattern = 'CorrDiff_ep599'
    # dir_pattern = 'CorrDiff_ep739'
    # dir_pattern = 'CorrDiff_ep299'
    # dir_pattern = 'CorrDiff_ep399'
    # dir_pattern = 'CorrDiff_ep499'
    compute_ensemble_metrics(parent_save_dir=eval_root_dir, save_dir_pattern=dir_pattern,
                             ensemble_size=13)

if __name__ == '__main__':
    # main()
    rebuttal()