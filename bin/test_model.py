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
import os
from cp_distill.bin_utils import file_path, file_or_dir_path

def run(args):
    import time
    import logging
    import numpy as np
    import glob
    import torch
    import warnings
    # Always use the CellposeModelX to allow saving tiles
    from cp_distill.cellpose_ext import CellposeModelX, CPnetX
    from cp_distill.image_utils import filter_segmentation
    from cellpose.metrics import aggregated_jaccard_index, average_precision
    from sklearn.metrics import jaccard_score

    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s - %(message)s',
        level=logging.INFO)

    start_time = time.time()

    # Allow the CPnetX IO to be saved to a directory.
    tile_dir = None
    tile_dir2 = None
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        if args.save_tiles:
            tile_dir = os.path.join(args.save_dir, 'cp')
            tile_dir2 = os.path.join(args.save_dir, 'st')
            os.makedirs(tile_dir, exist_ok=True)
            os.makedirs(tile_dir2, exist_ok=True)

    # Find input images
    combined_images = []
    images = args.image
    if np.isscalar(images):
        images = [images]
    for f in images:
        if os.path.isdir(f):
            for filename in glob.glob(os.path.join(f, '*.npy')):
                combined_images.append(filename)
        elif os.path.isfile(f):
            combined_images.append(f)
        else:
            raise FileNotFoundError(f)

    # Load the student model
    logging.info(f'Loading model state: {args.model}')
    state = torch.load(args.model)

    # Configure Cellpose arguments
    cyto = state['cyto']
    # The diameters are currently coded using defaults from Omero Screen.
    if cyto:
        # [[channel to segment, optional nuclear channel]]
        channels = [2, 1]
        diameter = 40
    else:
        # to segment grayscale images, input [0,0]
        channels = [0, 0]
        diameter = 10
    device = torch.device(args.device)

    # Create Cellpose
    cellpose_model = state['model']
    model = CellposeModelX(
        model_type=cellpose_model,
        # Not used in create_dataset.py
        #residual_on=state['residual_on'], style_on=state['style_on'],
        #concatenation=state['concatenation'],
        device=device,
        save_directory=tile_dir
    )

    # Create a modified Cellpose using the student model
    model2 = CellposeModelX(
        model_type=cellpose_model,
        # Not used in create_dataset.py
        #residual_on=state['residual_on'], style_on=state['style_on'],
        #concatenation=state['concatenation'],
        device=device,
        save_directory=tile_dir2
    )
    net = CPnetX(
        nbase=state['nbase'], nout=3, sz=3,
        # Use as trained
        residual_on=state['residual_on'], style_on=state['style_on'],
        concatenation=state['concatenation'],
        mkldnn=model2.net.mkldnn,
    )
    net.load_state_dict(state['model_state_dict'])
    net.eval()
    net = net.to(device)
    # This has been trained to be a drop-in replacement.
    # If the max architecture is smaller than 256 then this must be
    # stored in the CellposeModel as the styles output array is smaller.
    model2.net = net
    model2.nbase = net.nbase

    threshold = args.threshold

    # Run Cellpose
    all_aji = []
    for i, image in enumerate(combined_images):
        logging.info(f'Processing image {i+1}: {image}')
        img = np.load(image)

        # Image should be 2D/3D (with channels)
        # Cellpose requires XYC
        # (Note: Although the CellposeModel.exec method is documented as
        # Z x nchan x Y x X the internal code reshapes the data to channels last.)
        if cyto:
            # Require 2 channels
            if img.ndim != 3:
                raise Exception('{image} requires 2 channels')
            img = np.dstack([img[args.nuclei_channel-1], img[args.cyto_channel-1]])
        else:
            if img.ndim == 3:
                img = img[args.nuclei_channel-1]
            elif img.ndim != 2:
                raise Exception('{image} requires 1 channel')

        # Run Cellpose
        t1 = time.time()
        masks_array, flows, styles = model.eval(
            img,
            channels=channels,
            batch_size=args.batch_size,
            diameter=diameter, normalize=False,
            cellprob_threshold=args.cellprob_threshold,
            flow_threshold=args.flow_threshold, interp=args.interp,
            min_size=args.min_size,
        )
        t1 = time.time() - t1
        m1 = filter_segmentation(masks_array)

        # Run student model
        t2 = time.time()
        masks_array, flows, styles = model2.eval(
            img,
            channels=channels,
            batch_size=args.batch_size,
            diameter=diameter, normalize=False,
            cellprob_threshold=args.cellprob_threshold,
            flow_threshold=args.flow_threshold, interp=args.interp,
            min_size=args.min_size,
            # This is required to avoid loading the original model again
            model_loaded=True
        )
        t2 = time.time() - t2
        m2 = filter_segmentation(masks_array)
        logging.info(f'Time Cellpose {t1:.6f}, Student {t2:.6f} ({t2/t1:.6f})')

        if args.save_dir:
            # Save masks
            name = os.path.splitext(os.path.basename(image))[0]
            np.save(os.path.join(args.save_dir, name + '.cp'), m1)
            np.save(os.path.join(args.save_dir, name + '.st'), m2)

        # Score masks
        # Note: Masks are 2D arrays given the single image input.
        jac = jaccard_score(np.where(m2 > 0, 1, 0).ravel(),
                            np.where(m1 > 0, 1, 0).ravel())

        # Cellpose metrics
        masks_true = [m1]
        masks_pred = [m2]
        # We do not report mask_ious(m1, m2); this reports the IoU for each
        # object in the true mask against the best assignement of the predicted
        # mask. The IoUs are summarised in the aggregated_jaccard_index and
        # average_precision metrics.
        # Catch warning due to 0 / 0 when there is no mask overlap.
        # Cellpose metrics will reset NaN values to zero.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aji = aggregated_jaccard_index(masks_true, masks_pred)
            ap, tp, fp, fn = average_precision(masks_true, masks_pred, threshold=threshold)
        all_aji.append(aji)
        logging.info(f'IoU {jac}, Match IoU {aji}')
        logging.info(f'Precision at {threshold}: {ap[0]}, TP {tp[0]}, FP {fp[0]}, FN {fn[0]}')

        # if tile_dir:
        # TODO:
        # This script could compute the loss function on each batch for the
        # network outputs, and other metrics (see test_rotations.py)

    logging.info('Images=%d; Match IoU: min=%.5f, max=%.5f, mean=%.5f, std=%.5f',
        len(all_aji), np.min(all_aji), np.max(all_aji),
        np.mean(all_aji), np.std(all_aji))
    t = time.time() - start_time
    logging.info(f'Done (in {t} seconds)')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Program to test a student model against Cellpose.' +
            ' Images are assumed to be normalised to [0, 1].')

    parser.add_argument('image', nargs='+', metavar='IMAGE',
        type=file_or_dir_path,
        help='image or image directory')
    parser.add_argument('-m', '--model', dest='model', type=file_path,
        required=True,
        help='CellPose student model state file')
    parser.add_argument('--cyto-channel', dest='cyto_channel', type=int,
        default=2,
        help='Cytoplasm channel (1-based index) (default: %(default)s)')
    parser.add_argument('--nuclei-channel', dest='nuclei_channel', type=int,
        default=1,
        help='Nuclei channel (1-based index) (default: %(default)s)')
    parser.add_argument('-d', '--device', dest='device',
        default='cuda',
        help='Device (default: %(default)s)')
    parser.add_argument('-s', '--save', dest='save_dir', type=str,
        help='Save directory')
    parser.add_argument('--save-tiles', dest='save_tiles', action='store_true',
        help='Save network input/output tiles')
    parser.add_argument('--threshold', nargs='+', type=float,
        default=[0.5, 0.75, 0.9],
        help='Threshold for true positive match (default: %(default)s)')
    group = parser.add_argument_group('Cellpose')
    group.add_argument('--batch-size', dest='batch_size', type=int,
        default=8,
        help='Batch size (default: %(default)s)')
    group.add_argument('--cellprob-threshold', dest='cellprob_threshold',
        type=float, default=0.0,
        help='Cell probability threshold (for the probability map)')
    group.add_argument('--flow-threshold', dest='flow_threshold', type=float,
        default=0.4,
        help='Flow threshold (for the match between predicted flows and mask flows).' +
        ' Use 0.0 to output the maximum number of mask objects.')
    group.add_argument('--interp', type=bool, default=True,
        help='Interpolate flows')
    group.add_argument('--min-size', dest='min_size', type=int,
        default=15,
        help='Minimum object size')

    args = parser.parse_args()
    run(args)
