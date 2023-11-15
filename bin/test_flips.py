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
from cp_distill.bin_utils import file_path, dir_path

def run(args):
    import time
    import logging
    import numpy as np
    import torch
    import torch.nn.functional as F
    from cp_distill.cellpose_ext import CPnetX
    from cp_distill.transforms import cp_flip
    from cellpose.core import check_mkl
    from torch.utils import mkldnn as mkldnn_utils
    from torchmetrics.classification import BinaryJaccardIndex

    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s - %(message)s',
        #format='[%(levelname)s - %(message)s',
        level=args.log_level)

    start_time = time.time()

    # Find input images
    images = args.image
    if np.isscalar(images):
        images = [images]

    device = torch.device(args.device)
    gpu = device.type == 'cuda'
    mkldnn = False if gpu else check_mkl(True)

    # Create Cellpose network
    net = CPnetX(
        nbase=[2, 32, 64, 128, 256],
        nout=3,
        sz=3,
        # Defaults from CellposeModelX
        residual_on=True, style_on=True, concatenation=False,
        # For CPU support
        mkldnn=mkldnn,
        # diam mean is ignored in cellpose 2.0 as it is saved in the model
    ).to(device)
    net.load_model(os.path.abspath(args.model), device=device)
    # Put in evaluation mode (train=False)
    net.eval()
    # Support CPU mode
    if mkldnn:
        net = mkldnn_utils.to_mkldnn(net)

    jaccard = BinaryJaccardIndex(threshold=0.5).to(device)

    def _to_device(x):
        X = torch.from_numpy(x).float().to(device)
        return X

    def _from_device(X):
        x = X.detach().cpu().numpy()
        return x

    # Run Cellpose
    for i, image in enumerate(images):
        logging.info(f'Processing image {i}: {image}')
        x = np.load(image)

        # Image should be 2D/3D (with channels)
        # Cellpose net requires
        # Tiles dimensions: ntiles x nchan x bsize x bsize
        # (bsize = size of tiles)
        if x.ndim == 2:
            x = np.array([x, np.zeros(x.shape)])
        if x.ndim != 3:
            raise Exception('{image} requires 2/3 dimensions')
        x = x[np.newaxis,...]

        # From cellpose.core:UnetModel.network
        X = _to_device(x)
        with torch.no_grad():
            # Ignore style and y32
            y, *_ = net(X)
        y = y.squeeze()

        # Set up for Jaccard
        # Use only the probability map (ignore gradients)
        sy = F.sigmoid(y[2])
        p = torch.where(sy > 0.5, 1.0, 0.0)

        loss_flow = F.mse_loss(y[:2,:], y[:2,:])
        loss_p = F.binary_cross_entropy_with_logits(y[2], sy)

        iou = jaccard(p, p)
        logging.info('Flip %d loss : flows mse %10.6g; p bce %10.6g; p IoU %10.6g',
                     0, loss_flow, loss_p, iou)

        y = _from_device(y)
        if args.save_dir:
            name = os.path.splitext(os.path.basename(image))[0]
            np.save(os.path.join(args.save_dir, name + "_0_0"), y)

        # Test with flips
        for k in range(1, 4):
            # Flip the original tensor: 1 x nchan x Y x X
            xx = x
            if k & 1 == 1:
                xx = np.flip(xx, axis=-1)
            if k & 2 == 2:
                xx = np.flip(xx, axis=-2)

            # (note this is padded with an extra dimension)
            X = _to_device(xx.copy())
            with torch.no_grad():
                # Ignore style and y32
                yy, *_ = net(X)
            yy = yy.squeeze()

            if args.save_dir:
                name = os.path.splitext(os.path.basename(image))[0]
                np.save(os.path.join(args.save_dir, name + f"_{k}_0"), _from_device(yy))

            # Flip original results and compare
            fy = cp_flip(y, k=k)
            ry = _to_device(fy)
            rsy = F.sigmoid(ry[2])
            rp = torch.where(rsy > 0.5, 1.0, 0.0)

            loss_flow = F.mse_loss(yy[:2,:], ry[:2,:])
            # Binary cross entropy compare inputs to a target in [0, 1]
            # We could use p in {0, 1} but here use the sigmoid of y.
            loss_p = F.binary_cross_entropy_with_logits(yy[2], rsy) #p)

            pp = F.sigmoid(yy[2])
            iou = jaccard(pp, rp)
            logging.info('Flip %d loss : flows mse %10.6g; p bce %10.6g; p IoU %10.6g',
                         k, loss_flow, loss_p, iou)
            del yy
        del X
        del y

    t = time.time() - start_time
    logging.info(f'Done (in {t:.6g} seconds)')

if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__));
    cellpose_model = os.path.join(base, "..", "cellpose_models", "Nuclei_Hoechst")

    parser = argparse.ArgumentParser(
      description='Program to test Cellpose using a flipped tile.')

    parser.add_argument('image', nargs='+', metavar='IMAGE',
        type=file_path,
        help='image tile')
    parser.add_argument('-m', '--model', dest='model', type=file_path,
        default=cellpose_model,
        help='CellPose model (default: %(default)s)')
    parser.add_argument('--cyto-channel', type=int,
        default=2,
        help='Cytoplasm channel (1-based index) (default: %(default)s)')
    parser.add_argument('--nuclei-channel', type=int,
        default=1,
        help='Nuclei channel (1-based index) (default: %(default)s)')
    parser.add_argument('-d', '--device', dest='device',
        default='cuda',
        help='Device (default: %(default)s)')
    parser.add_argument('-s', '--save', dest='save_dir', type=dir_path,
        help='Save directory prefix (default: %(default)s)')
    parser.add_argument('--log-level', type=int,
        default=20,
        help='Log level (default: %(default)s). WARNING=30; INFO=20; DEBUG=10')

    args = parser.parse_args()
    run(args)
