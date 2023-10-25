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
    import psutil
    import glob
    import torch
    import shutil
    from cp_distill.cellpose_ext import CellposeModelX

    # Debug memory usage
    if args.memory:
        logging.basicConfig(
            format='[%(asctime)s] %(levelname)s - %(mem)s - %(message)s',
            level=logging.INFO)

        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            process = psutil.Process()
            i = process.memory_info()
            record.mem = f'rss={i.rss/1024**3:8.4f}, vms={i.vms/1024**3:8.4f}'
            return record

        logging.setLogRecordFactory(record_factory)
    else:
        logging.basicConfig(
            format='[%(asctime)s] %(levelname)s - %(message)s',
            level=logging.INFO)

    start_time = time.time()

    save_directory=f'{args.save_dir}_{os.path.basename(args.model)}'
    if os.path.isdir(save_directory):
        if not args.delete:
            raise Exception(f'Save directory exists: {save_directory}')
        # Clean output directory
        for root, dirs, files in os.walk(save_directory):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
    else:
        os.makedirs(save_directory)

    # Save options to the directory
    with open(os.path.join(save_directory, 'settings.txt'), 'w') as f:
        print(args, file=f)

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

    # Configure Cellpose arguments
    # The diameters are currently coded using defaults from Omero Screen.
    if args.cyto:
        # [[channel to segment, optional nuclear channel]]
        channels = [2, 1]
        diameter = 40
    else:
        # to segment grayscale images, input [0,0]
        channels = [0, 0]
        diameter = 10
    device = torch.device(args.device)

    # Create Cellpose with option to save input/output
    segmentation_model = CellposeModelX(
        model_type=os.path.abspath(args.model),
        device=device,
        save_directory=save_directory
    )

    # Run Cellpose
    for i, image in enumerate(combined_images):
        logging.info(f'Processing image {i+1}: {image}')
        img = np.load(image)

        # Image should be 2D/3D (with channels)
        # Cellpose requires XYC
        # (Note: Although the CellposeModel.exec method is documented as
        # Z x nchan x Y x X the internal code reshapes the data to channels last.)
        if args.cyto:
            # Require 2 channels
            if img.ndim != 3:
                raise Exception('{image} requires 2 channels')
            img = np.dstack([img[args.nuclei_channel-1], img[args.cyto_channel-1]])
            axes = (1, 2)
        else:
            if img.ndim == 3:
                img = img[args.nuclei_channel-1]
            elif img.ndim != 2:
                raise Exception('{image} requires 1 channel')
            axes = (0, 1)

        # Do rotations
        for k in args.rotations:
            logging.info(f'Processing rotation {k} on {i+1}: {image}')

            # TODO: Should this be saved?
            start = segmentation_model._count
            masks_array, flows, styles = segmentation_model.eval(
                np.rot90(img, k=k, axes=axes),
                channels=channels,
                batch_size=args.batch_size,
                diameter=diameter, normalize=False
            )
            logging.info(f'Saved {save_directory}: {start}-{segmentation_model._count - 1}')

    t = time.time() - start_time
    logging.info(f'Done (in {t} seconds)')

if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__));
    cellpose_model = os.path.join(base, "..", "cellpose_models", "Nuclei_Hoechst")

    parser = argparse.ArgumentParser(
        description='Program to convert input images ([C] x X x Y) to a dataset.' +
            ' Images are assumed to be normalised to [0, 1].')

    parser.add_argument('image', nargs='+', metavar='IMAGE',
        type=file_or_dir_path,
        help='image or image directory')
    parser.add_argument('-m', '--model', dest='model', type=file_path,
        default=cellpose_model,
        help='CellPose model (default: %(default)s)')
    parser.add_argument('--cyto-channel', dest='cyto_channel', type=int,
        default=2,
        help='Cytoplasm channel (1-based index) (default: %(default)s)')
    parser.add_argument('--nuclei-channel', dest='nuclei_channel', type=int,
        default=1,
        help='Nuclei channel (1-based index) (default: %(default)s)')
    parser.add_argument('--cyto', dest='cyto', action='store_true',
        help='Cytoplasm model (run with 2 channels; defaults to nuclei channel only)')
    parser.add_argument('-d', '--device', dest='device',
        default='cuda',
        help='Device (default: %(default)s)')
    parser.add_argument('--memory', dest='memory', action='store_true',
        help='Debug memory usage')
    parser.add_argument('-s', '--save', dest='save_dir', type=str,
        default='test_data',
        help='Save directory prefix (default: %(default)s)')
    parser.add_argument('--rotations', dest='rotations', nargs='+', type=int,
        default=[0],
        help='90-degree rotations, e.g. k=0 1 2 3 (default: %(default)s)')
    parser.add_argument('--delete', dest='delete', action='store_true',
        help='Delete existing data (default is to error)')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
        default=8,
        help='Batch size (default: %(default)s)')

    args = parser.parse_args()
    run(args)
