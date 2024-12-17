import os
import torch
from tqdm import tqdm, trange

__author__ = 'Yuhao Liu'

"""
This script is used to combine batches of predictions into a single file.
"""


def combine_batches(input_batch_dir: str, output_batch_dir: str, factor: int = 3):
    """
    Combine batches of predictions into a single file.
    :param input_batch_dir: Directory containing input batches.
    :param output_batch_dir: Directory to save the combined batch.
    :param factor: Combine every `factor` batches into one.
    """
    assert os.path.exists(input_batch_dir), f"Input batch directory {input_batch_dir} does not exist."
    assert factor >= 1, "Factor must be greater than or equal to 1."
    os.makedirs(output_batch_dir, exist_ok=True)
    in_batches = os.listdir(input_batch_dir)
    in_batches = [b for b in in_batches if b.endswith('.pt')]
    print(f"Found {len(in_batches)} batches in {input_batch_dir}")
    max_out_batch_idx = len(in_batches) // factor
    print(f"Combining every {factor} batches into one. Will generate {max_out_batch_idx} batches.")
    out_idx = 0
    for b in tqdm(in_batches):
        in_batch = torch.load(os.path.join(input_batch_dir, b))
        if out_idx % factor == 0:
            out_batch = in_batch
        else:
            for k, v in in_batch.items():
                out_batch[k] = torch.cat((out_batch[k], v), dim=0)
        if out_idx % factor == factor - 1:
            torch.save(out_batch, os.path.join(output_batch_dir, f'batch_{out_idx // factor}.pt'))
        out_idx += 1
        if out_idx > max_out_batch_idx * factor: # Early stop
            break
    print(f"Saved {out_idx // factor} batches in {output_batch_dir}.")






if __name__ == '__main__':
    in_dir = '/home/yl241/data/rainfall_eval/eval_samples_s256_b12/'
    out_dir = '/home/yl241/data/rainfall_eval/eval_samples_s256_b36_agg/'
    combine_batches(in_dir, out_dir)