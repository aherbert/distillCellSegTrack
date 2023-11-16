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

# Adapted from https://stackoverflow.com/a/1094933
# Changed to use KiB as smallest unit and always use B suffix
def sizeof_fmt(num):
    num = num / 1024.0
    for unit in ("KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB"):
        if num < 1048576.0:
            return f"{num:9.2f} {unit}"
        num = num / 1024.0
    return f"{num:.2f} YiB"

def run(args):
    import time
    import logging
    import numpy as np
    import glob
    import torch
    # Always use the CellposeModelX to allow saving tiles
    from cp_distill.cellpose_ext import CellposeModelX, CPnetX
    from cp_distill.torch_utils import memory
    
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s - %(message)s',
        level=logging.INFO)

    start_time = time.time()

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

    device = torch.device(args.device)

    def log_memory():
        m = memory(device)
        logging.info('Allocated %s | max %s : Reserved %s | max %s',
            sizeof_fmt(m[0]), sizeof_fmt(m[1]), sizeof_fmt(m[2]), sizeof_fmt(m[3]))

    # Load the student model
    logging.info(f'Loading model state: {args.model}')
    state = torch.load(args.model, map_location=device)

    # Configure Cellpose arguments
    cyto = state['cyto']
    # The diameters are currently coded using defaults from Omero Screen.
    if cyto:
        # [[channel to segment, optional nuclear channel]]
        channels = [2, 1]
        diameter = args.diameter if args.diameter else 40
    else:
        # to segment grayscale images, input [0,0]
        channels = [0, 0]
        diameter = args.diameter if args.diameter else 10

    # Create Cellpose
    cellpose_model = os.path.abspath(args.cellpose_model) if args.cellpose_model else state['model']
    if not os.path.isfile(cellpose_model):
        logging.error(f'Model not found {cellpose_model}')
        exit(1)
    model = CellposeModelX(
        model_type=cellpose_model,
        # Not used in create_dataset.py
        #residual_on=state['residual_on'], style_on=state['style_on'],
        #concatenation=state['concatenation'],
        device=device,
        bsize=args.tile_size
    )
    if not args.teacher:
        net = CPnetX(
            nbase=state['nbase'], nout=3, sz=3,
            # Use as trained
            residual_on=state['residual_on'], style_on=state['style_on'],
            concatenation=state['concatenation'],
            mkldnn=model.net.mkldnn,
        )
        net.load_state_dict(state['model_state_dict'])
        net.eval()
        net = net.to(device)
        # This has been trained to be a drop-in replacement.
        # If the max architecture is smaller than 256 then this must be
        # stored in the CellposeModel as the styles output array is smaller.
        model.net = net
        model.nbase = net.nbase
    log_memory()
    
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
            diameter=diameter, normalize=False,
            cellprob_threshold=args.cellprob_threshold,
            flow_threshold=args.flow_threshold, interp=args.interp,
            min_size=args.min_size,
            # This is required to avoid loading the original model again
            model_loaded=not args.teacher
        )
        log_memory()

        # logging.info(f'Time Cellpose {t1:.6f}, Student {t2:.6f} ({t2/t1:.6f})')
        # logging.info(f'Network Time Cellpose {t1b[0]:.6f}, Student {t2b[0]:.6f} ({t2b[0]/t1b[0]:.6f})')
        # logging.info(f'IO Time Cellpose {t1b[1]:.6f}, Student {t2b[1]:.6f} ({t2b[1]/t1b[1]:.6f})')

    t = time.time() - start_time
    logging.info(f'Done (in {t} seconds)')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Program to test a teacher/student model memory usage.' +
            ' Images are assumed to be normalised to [0, 1].')

    parser.add_argument('image', nargs='+', metavar='IMAGE',
        type=file_or_dir_path,
        help='image or image directory')
    parser.add_argument('-m', '--model', dest='model', type=file_path,
        required=True,
        help='CellPose student model state file')
    parser.add_argument('-c', '--cellpose-model', dest='cellpose_model',
        type=file_path,
        help='CellPose model file (override the model path in the state file)')
    parser.add_argument('--cyto-channel', type=int,
        default=2,
        help='Cytoplasm channel (1-based index) (default: %(default)s)')
    parser.add_argument('--nuclei-channel', type=int,
        default=1,
        help='Nuclei channel (1-based index) (default: %(default)s)')
    parser.add_argument('--diameter', type=float,
        default=0,
        help='Diameter (default: nuclei=10; cyto=40)')
    parser.add_argument('-d', '--device', dest='device',
        default='cuda',
        help='Device (default: %(default)s)')
    parser.add_argument('-t', '--teacher', dest='teacher', default=False,
        action=argparse.BooleanOptionalAction,
        help='Run for teacher model (Cellpose, default: %(default)s)')
    group = parser.add_argument_group('Cellpose')
    group.add_argument('--batch-size', type=int,
        default=128,
        help='Batch size (default: %(default)s)')
    parser.add_argument('--tile-size', type=int,
        default=224,
        help='Tile size (default: %(default)s). Use zero to disable tiles.')
    group.add_argument('--cellprob-threshold',
        type=float, default=0.0,
        help='Cell probability threshold (default: %(default)s)')
    group.add_argument('--flow-threshold', type=float,
        default=0.4,
        help='Flow threshold (default: %(default)s)' +
        ' (for the match between predicted flows and mask flows).' +
        ' Use 0.0 to output the maximum number of mask objects.')
    group.add_argument('--interp', default=True,
        action=argparse.BooleanOptionalAction,
        help='Interpolate flows')
    group.add_argument('--min-size', type=int,
        default=15,
        help='Minimum object size (default: %(default)s)')

    args = parser.parse_args()
    run(args)
