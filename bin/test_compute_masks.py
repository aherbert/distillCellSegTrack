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
from cp_distill.bin_utils import file_path

def run(args):
    import os
    import time
    import logging
    import numpy as np
    import torch
    from cellpose.dynamics import compute_masks

    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s - %(message)s',
        level=logging.INFO)

    start_time = time.time()

    # Find input images
    images = args.image
    if np.isscalar(images):
        images = [images]

    device = torch.device(args.device)
    use_gpu = args.device == 'cuda'

    # Run Cellpose
    for i, image in enumerate(images):
        logging.info(f'Processing image {i+1}: {image}')
        img = np.load(image)

        if img.ndim != 3:
            raise Exception('{image} requires 3 channels')

        # Run Cellpose mask creation
        m, p = compute_masks(img[:2], img[2], use_gpu=use_gpu, device=device,
            niter=args.niter, cellprob_threshold=args.cellprob_threshold,
            flow_threshold=args.flow_threshold, interp=args.interp,
            min_size=args.min_size)

        if args.save:
            # Save mask
            name = os.path.splitext(image)[0] + '.mask.npy'
            logging.info(f'Saving mask {i+1}: {name}')
            np.save(name, m)

    t = time.time() - start_time
    logging.info(f'Done (in {t} seconds)')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Program to run the Cellpose mask computation' +
            ' on the Cellpose output: [flow, flow, probability map]')

    parser.add_argument('image', nargs='+', metavar='IMAGE',
        type=file_path,
        help='Cellpose output (3 x Y x X)')
    parser.add_argument('-d', '--device', dest='device',
        default='cpu',
        help='Device (default: %(default)s)')
    parser.add_argument('-s', '--save', dest='save', action='store_true',
        help='Save the mask output')
    parser.add_argument('--niter', type=int, default=200,
        help='Number of iterations')
    parser.add_argument('--cellprob-threshold', dest='cellprob_threshold',
        type=float, default=0.0,
        help='Cell probability threshold (for the probability map)')
    parser.add_argument('--flow-threshold', dest='flow_threshold', type=float,
        default=0.4,
        help='Flow threshold (for the match between predicted flows and mask flows)')
    parser.add_argument('--interp', type=bool, default=True,
        help='Interpolate flows')
    parser.add_argument('--min-size', dest='min_size', type=int,
        default=15,
        help='Minimum object size')

    args = parser.parse_args()
    run(args)