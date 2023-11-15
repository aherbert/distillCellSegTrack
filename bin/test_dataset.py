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

import argparse
from cp_distill.bin_utils import dir_path

def run(args):
    import os
    import logging
    from cp_distill.datasets import find_images, CPDataset
    if args.load:
        import time
        from torch.utils.data import DataLoader

    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s - %(message)s',
        level=args.log_level)

    # Cannot validate without loading data
    args.min_objects *= args.load
    if args.min_objects:
        from cellpose.dynamics import compute_masks
        import numpy as np

    for d in args.directory:
        logging.info(f'Processing dataset: {d}')
        images = find_images(d, prefix='bad_input')
        if images:
            logging.info(f'Dataset: {d}: invalid images {len(images)}')
            if args.restore_invalid:
                logging.info(f'Dataset: {d}: restoring invalid images')
                for img in images:
                    name = f'bad_input_{img}.npy'
                    to = f'input_{img}.npy'
                    os.rename(os.path.join(d, name), os.path.join(d, to))
        images = find_images(d)
        logging.debug('Image tile numbers: %s', images)
        if not images:
            logging.warning(f"Dataset: {d} has no image tiles")
            continue
        dataset = CPDataset(images, d, load_y32=args.load_y32);
        x, y, y32 = dataset[0]
        logging.debug('Data shapes: %s, %s, %s', x.shape, y.shape, y32.shape)
        for (ch, data, name) in [(2, x, 'x'), (3, y, 'y'), (32, y32, 'y32')]:
            if data.shape[0] != ch:
                logging.warning(f"Dataset: {d} : {name} does not have {ch} channels, shape = %s",
                                data.shape)
            if ch == 2:
                tile = data.shape[1:]
            elif tile != data.shape[1:] and (args.load_y32 or ch != 32):
                logging.warning(f"Dataset: {d} : x and {name} tile size mismatch, %s vs %s",
                                tile, data.shape[1:])
        logging.info(f"Dataset: {d} : tiles = {len(dataset)} : tile dimensions {tile}")

        # Try running through a batch loader
        objects = []
        removed = 0
        if args.load:
            start_time = time.time()
            loader = DataLoader(dataset, batch_size=args.batch_size,
                num_workers=args.num_workers)
            index = 0
            for i, (x, y, y32) in enumerate(loader):
                if i == 0:
                    s1, s2, s3 = x.shape, y.shape, y32.shape
                    logging.info("Data load shapes x=%s, y=%s, y32=%s [batch=0]",
                                  s1, s2, s3)
                else:
                    if x.shape != s1 or y.shape != s2 or y32.shape != s3:
                        logging.info("Data load shapes x=%s, y=%s, y32=%s [batch=%d]",
                                      x.shape, y.shape, y32.shape, i)
                if args.min_objects:
                    # Predict the mask from the y32 data
                    # Identify input with too few objects
                    y = y.detach().cpu().numpy()
                    for j in range(len(y)):
                        img = images[index]
                        index += 1
                        dP, cellprob = y[j,:2], y[j,2]
                        # Run with defaults on the CPU since it does
                        # not really matter about scaling. The purpose is
                        # to identify blank tiles.
                        m, p = compute_masks(dP, cellprob)
                        # Count objects
                        count = np.max(m)
                        path = os.path.join(d, f'input_{img}.npy')
                        if count < args.min_objects and args.rename_invalid:
                            to = f'bad_input_{img}.npy'
                            os.rename(path, os.path.join(d, to))
                            logging.warning(f'Renamed {path} to {to} : Objects = {count}')
                            removed += 1
                        else:
                            objects.append(count)
                            logging.info(f'Image {path} : Objects = {count}')

            if args.min_objects:
                logging.info(f'Dataset {d} : Objects = n=%d; min=%s; max=%s; mean=%.5f, std=%.5f : Removed {removed} (%.2f%%)',
                    len(objects), np.min(objects), np.max(objects), np.mean(objects), np.std(objects),
                    100 * removed / (removed + len(objects)))

            t = time.time() - start_time
            logging.info(f'Loaded dataset {d} (in {t:.5f} seconds)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Program to test a Cellpose tiled dataset.')

    parser.add_argument('directory', nargs='+', metavar='DIR',
        type=dir_path,
        help='Dataset directory')
    parser.add_argument('--log-level', type=int,
        default=20,
        help='Log level (default: %(default)s). WARNING=30; INFO=20; DEBUG=10')
    parser.add_argument('--load',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Load the dataset (default: %(default)s)')
    parser.add_argument('--batch-size', type=int,
        default=8,
        help='Batch size for the data loader (default: %(default)s)')
    parser.add_argument('--load-y32',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Load the 32-channel upsample layer (default: %(default)s)')
    parser.add_argument('--num-workers', type=int,
        default=0,
        help='Number of workers for asynchronous data loading (default: %(default)s)')
    parser.add_argument('--min-objects', type=int,
        default=0,
        help='Predict objects from y32 data and identify invalid tiles (default: %(default)s)')
    parser.add_argument('--restore-invalid',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Restore previous invalid input images by renaming (default: %(default)s)')
    parser.add_argument('--rename-invalid',
        default=True,
        action=argparse.BooleanOptionalAction,
        help='Rename invalid input images [objects < min objects] (default: %(default)s)')

    args = parser.parse_args()
    run(args)
