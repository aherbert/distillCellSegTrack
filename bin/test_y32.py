#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted.

# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.
# ------------------------------------------------------------------------------

"""
Displays a boxplot of the range of the y32 data in a training dataset.
"""

import argparse
from cp_distill.bin_utils import dir_path

def run(args):
    import os
    import numpy as np
    from cp_distill.datasets import find_images
    import matplotlib.pyplot as plt

    images = find_images(args.directory, 'output32')

    mind, maxd, meand, stdd = [], [], [], []
    for i in images:
        image = np.load(os.path.join(args.directory, f'output32_{i}.npy'))
        x = image.reshape((32, image.shape[1] * image.shape[2]))
        mind.append(np.min(x, axis=1))
        maxd.append(np.max(x, axis=1))
        meand.append(np.mean(x, axis=1))
        stdd.append(np.std(x, axis=1))

    # box plot of each
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.canvas.manager.set_window_title(args.directory)
    ax1.set(ylabel='Min')
    ax1.boxplot(np.array(mind))
    ax2.set(ylabel='Max')
    ax2.boxplot(np.array(maxd))
    ax3.set(ylabel='Mean')
    ax3.boxplot(np.array(meand))
    ax4.set(ylabel='Std')
    ax4.boxplot(np.array(stdd))
    for ax in fig.get_axes():
        ax.xaxis.set_ticklabels([])
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Program to show the stats for the y32 data of a training dataset.')

    parser.add_argument('directory', metavar='DIR',
        type=dir_path,
        help='Dataset directory')

    args = parser.parse_args()
    run(args)
