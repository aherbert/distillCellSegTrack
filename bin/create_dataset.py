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
    import json
    import pickle
    import cv2
    from cp_distill.cellpose_ext import CellposeModelX
    from cp_distill.image_utils import filter_segmentation
    from cellpose.dynamics import masks_to_flows
    from cellpose.transforms import make_tiles, resize_image, pad_image_ND

    # Debug memory usage
    if args.memory:
        logging.basicConfig(
            format='[%(asctime)s] %(levelname)s - %(mem)s - %(message)s',
            level=args.log_level)

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
            level=args.log_level)

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
    with open(os.path.join(save_directory, 'settings.json'), 'w') as f:
        json.dump(vars(args), f)

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
        diameter = args.diameter if args.diameter else 40
    else:
        # to segment grayscale images, input [0,0]
        channels = [0, 0]
        diameter = args.diameter if args.diameter else 10
    device = torch.device(args.device)

    # Create Cellpose with option to save input/output
    segmentation_model = CellposeModelX(
        model_type=args.model,
        device=device,
        save_directory=save_directory,
        save_y32=args.save_y32,
        save_styles=args.save_styles,
    )
    use_gpu = segmentation_model.gpu

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
        else:
            if img.ndim == 3:
                img = img[args.nuclei_channel-1]
            elif img.ndim != 2:
                raise Exception('{image} requires 1 channel')

        # Do rotations. Save the first rotation for the testing data.
        imgs = None
        for k in args.rotations:
            logging.info(f'Processing rotation {k} on {i+1}: {image}')

            start = segmentation_model._count
            rotated = np.rot90(img, k=k, axes=(0, 1))
            masks_array, flows, styles = segmentation_model.eval(
                rotated,
                channels=channels,
                batch_size=args.batch_size,
                diameter=diameter, normalize=False
            )
            logging.info(f'Saved {save_directory}: {start}-{segmentation_model._count - 1}')

            # Create validation data
            if imgs is None:
                orig_dim = img.shape[:2] if args.cyto else img.shape
                rescale = segmentation_model.last_rescale
                imgs = resize_image(rotated, rsz=rescale,
                    no_channels=not args.cyto)
                logging.log(15, 'Resized %s to %s', img.shape, imgs.shape)
                # From cellpose.core.UnetModel._run_net
                # Make image nchan x Ly x Lx for net.
                # Put the channel to segment first (assume cyto channels = [2, 1])
                imgs = np.transpose(imgs[..., [1, 0]], (2,0,1)) if args.cyto else imgs[np.newaxis, ...]
                # pad image for net so Ly and Lx are divisible by 16
                imgs, ysub, xsub = pad_image_ND(imgs)
                # slices from padding
                slc = []
                # Modified here for a single 2D input image.
                slc.append(slice(0, segmentation_model.nclasses + 1))
                slc.append(slice(ysub[0], ysub[-1]+1))
                slc.append(slice(xsub[0], xsub[-1]+1))
                slc = tuple(slc)
                
                # From cellpose.core.UnetModel._run_tiled
                tile_dim = imgs.shape[1:]
                tiles, ysub, xsub, Ly, Lx = make_tiles(imgs)
                ny, nx, nchan, ly, lx = tiles.shape
                tiles = np.reshape(tiles, (ny*nx, nchan, ly, lx))
                
                m1 = filter_segmentation(masks_array)
                # Save
                np.save(os.path.join(save_directory, f'tiles_{i+1}.npy'), tiles)
                np.save(os.path.join(save_directory, f'mask_{i+1}.npy'), m1)
                with open(os.path.join(save_directory, f'tiles_{i+1}.pkl'), 'wb') as f:
                    pickle.dump((orig_dim, rescale, slc, tile_dim, ysub, xsub, Ly, Lx), f)
                logging.info(f'Saved validation tiles {save_directory}: {i+1} {tiles.shape}')

            if not args.compute_flows:
                continue
            logging.info('Computing flows')
            # Use the mask to create flows
            if args.scale_first:
                masks_array = resize_image(masks_array,
                    rsz=segmentation_model.last_rescale,
                    interpolation=cv2.INTER_NEAREST,
                    no_channels=True)
            dP = masks_to_flows(masks_array,
                use_gpu=use_gpu, device=device)
            # Scale up the flows to match the trained Cellpose output
            dP *= 5.
            # Create tiles: Y-flow, X-flow, Map
            # This is smoother if computed on the original mask then resized
            # with interpolation.
            m = np.where(masks_array > 0, 1.0, 0.0)
            y = np.concatenate([dP, m[np.newaxis, ...]])
            if not args.scale_first:
                y = y.transpose((1,2,0))
                y = resize_image(y, rsz=segmentation_model.last_rescale)
                y = y.transpose((2,0,1))
                # Mask interpolation will create non-binary pixels
                y[2] = np.where(y[2] > 0.5, 1.0, 0.0)
            y, *_ = pad_image_ND(y)
            tiles, *_ = make_tiles(y)
            ny, nx, nchan, ly, lx = tiles.shape
            tiles = np.reshape(tiles, (ny*nx, nchan, ly, lx))
            # Save tiles
            for j in range(len(tiles)):
                # TODO: Compute the difference to the current flows
                np.save(os.path.join(save_directory, f'output_{start+i}.npy'),
                    tiles[j])

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
    parser.add_argument('--cyto-channel',type=int,
        default=2,
        help='Cytoplasm channel (1-based index) (default: %(default)s)')
    parser.add_argument('--nuclei-channel', type=int,
        default=1,
        help='Nuclei channel (1-based index) (default: %(default)s)')
    parser.add_argument('--cyto',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Cytoplasm model (default: %(default)s). Run with 2 channels; defaults to nuclei channel only.')
    parser.add_argument('--diameter', type=float,
        default=0,
        help='Diameter (default: nuclei=10; cyto=40)')
    parser.add_argument('-d', '--device', dest='device',
        default='cuda',
        help='Device (default: %(default)s)')
    parser.add_argument('--memory',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Debug memory usage (default: %(default)s)')
    parser.add_argument('-s', '--save', dest='save_dir', type=str,
        default='test_data',
        help='Save directory prefix (default: %(default)s)')
    parser.add_argument('--save-y32',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Save the 32-channel upsample layer (default: %(default)s)')
    parser.add_argument('--save-styles',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Save the styles (default: %(default)s)')
    parser.add_argument('--rotations', nargs='+', type=int,
        default=[0],
        help='90-degree rotations, e.g. k=0 1 2 3 (default: %(default)s)')
    parser.add_argument('--delete',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Delete existing data, otherwise error (default: %(default)s)')
    parser.add_argument('--batch-size', type=int,
        default=8,
        help='Batch size (default: %(default)s)')
    parser.add_argument('--compute-flows',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Compute the flows from the predicted mask (default: %(default)s)')
    parser.add_argument('--scale-first',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Scale the predicted mask before computing flows, otherwise' +
            ' scale the computed flows (default: %(default)s)')
    parser.add_argument('--log-level', type=int,
        default=20,
        help='Log level (default: %(default)s). WARNING=30; INFO=20; DEBUG=10')

    args = parser.parse_args()
    args.model = os.path.abspath(args.model)
    run(args)
