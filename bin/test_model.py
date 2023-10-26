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
    from cellpose.models import CellposeModel
    from cp_distill.cellpose_ext import CellposeModelX, CPnetX
    from cp_distill.image_utils import filter_segmentation
    from sklearn.metrics import jaccard_score
    from scipy.optimize import linear_sum_assignment

    def iou(x, y):
        # Computes the IoU for the maximum matching between two masks.
        # Create an all-vs-all intersection between prediction and target.
        # i.e. count how many pixels from each object in x overlap each
        # object in y.
        n = np.max(y)
        m = np.max(x)
        if n + m <= 0:
            return 0
        count = np.zeros((m, n))
        for i in range(len(x)):
            if x[i] and y[i]:
                count[x[i]-1][y[i]-1] += 1
        # Find the maximum matching.
        top = np.max(count)
        cost = (top - count) / top
        row, col = linear_sum_assignment(cost)
        # Compute the IoU.
        intersect = count[row, col].sum()
        union = np.sum(x + y > 0)
        return intersect / union
    
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s - %(message)s',
        level=logging.INFO)

    start_time = time.time()

    if args.save_dir and not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

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
    model = CellposeModel(
        model_type=cellpose_model,
        # Not used in create_dataset.py
        #residual_on=state['residual_on'], style_on=state['style_on'],
        #concatenation=state['concatenation'],
        device=device
    )
    
    # Create a modified Cellpose using the student model
    model2 = CellposeModelX(
        model_type=cellpose_model,
        # Not used in create_dataset.py
        #residual_on=state['residual_on'], style_on=state['style_on'],
        #concatenation=state['concatenation'],
        device=device
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
    
    # Run Cellpose
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
        masks_array, flows, styles = model.eval(
            img,
            channels=channels,
            batch_size=args.batch_size,
            diameter=diameter, normalize=False
        )
        m1 = filter_segmentation(masks_array)
        
        # Run student model
        masks_array, flows, styles = model2.eval(
            img,
            channels=channels,
            batch_size=args.batch_size,
            diameter=diameter, normalize=False,
            # This is required to avoid loading the original model again
            model_loaded=True
        )
        m2 = filter_segmentation(masks_array)
        
        if args.save_dir:
            # Save masks
            name = os.path.splitext(os.path.basename(image))[0]
            np.save(os.path.join(args.save_dir, name + '.cp'), m1)
            np.save(os.path.join(args.save_dir, name + '.st'), m2)

        # Score masks
        # Note: Masks are 2D arrays given the single image input.
        jac = jaccard_score(np.where(m2 > 0, 1, 0).ravel(), 
                            np.where(m1 > 0, 1, 0).ravel())
        match_iou = iou(m1.ravel(), m2.ravel())
        logging.info(f'IoU {jac}, Match IoU {match_iou}')

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
    parser.add_argument('--batch-size', dest='batch_size', type=int,
        default=8,
        help='Batch size (default: %(default)s)')

    args = parser.parse_args()
    run(args)
