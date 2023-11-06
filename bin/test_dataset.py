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
    import logging
    from cp_distill.datasets import find_images, CPDataset
    if args.load:
        import time
        from torch.utils.data import DataLoader

    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s - %(message)s',
        level=args.log_level)

    for d in args.directory:
        logging.info(f'Processing dataset: {d}')
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
        if args.load:
            start_time = time.time()
            loader = DataLoader(dataset, batch_size=args.batch_size)
            for i, (x, y, y32) in enumerate(loader):
                if i == 0:
                    s1, s2, s3 = x.shape, y.shape, y32.shape
                    logging.info("Data load shapes x=%s, y=%s, y32=%s [batch=0]",
                                  s1, s2, s3)
                else:
                    if x.shape != s1 or y.shape != s2 or y32.shape != s3:
                        logging.info("Data load shapes x=%s, y=%s, y32=%s [batch=%d]",
                                      x.shape, y.shape, y32.shape, i)
            t = time.time() - start_time
            logging.info(f'Loaded dataset {d} (in {t:.5f} seconds)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Program to test a Cellpose tiled dataset.')

    parser.add_argument('directory', nargs='+', metavar='DIR',
        type=dir_path,
        help='Dataset directory')
    parser.add_argument('--log-level', dest='log_level', type=int,
        default=20,
        help='Log level (default: %(default)s). WARNING=30; INFO=20; DEBUG=10')
    parser.add_argument('--load', dest='load', action='store_true',
        help='Load the dataset')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
        default=8,
        help='Batch size for the data loader (default: %(default)s)')
    parser.add_argument('--load-y32', dest='load_y32', action='store_true',
        help='Load the 32-channel upsample layer.')

    args = parser.parse_args()
    run(args)
