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

def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def run(args):
    import time
    import logging
    import numpy as np
    import torch
    import torch.nn.functional as F
    from cellpose_ext import CPnetX
    from torch.utils import mkldnn as mkldnn_utils
    from torchmetrics.classification import BinaryJaccardIndex
    from rotations import cp_rotate90
    if args.matching:
        from scipy.optimize import linear_sum_assignment

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
    mkldnn = True if args.device == 'cpu' else None

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

    if args.matching:
        # Cross-channel matching of 3 rotations
        matching = [np.zeros((3, 2, 2)), np.zeros((3, 32, 32))]

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
            # Ignore style
            y, _, y32 = net(X)
        y = y.squeeze()
        y32 = y32.squeeze()
        
        # Set up for Jaccard
        # Use only the probability map (ignore gradients)
        sy = F.sigmoid(y[2])
        p = torch.where(sy > 0.5, 1.0, 0.0)

        loss_32 = F.mse_loss(y32, y32)
        loss_flow = F.mse_loss(y[:2,:], y[:2,:])
        loss_p = F.binary_cross_entropy_with_logits(y[2], sy) #p)

        iou = jaccard(p, p)
        logging.info('Rotation %3d loss : y32 mse %10.6g; flows mse %10.6g; p bce %10.6g; p IoU %10.6g',
                     0, loss_32, loss_flow, loss_p, iou)
        
        if args.save_dir:
            name = os.path.splitext(os.path.basename(image))[0]
            np.save(os.path.join(args.save_dir, name + "_0_0"), _from_device(y))
            np.save(os.path.join(args.save_dir, name + "_0_32"), _from_device(y32))
            
        # Test with rotations
        for k in range(1, 4):
            # rotate the original tensor
            # (note this is padded with an extra dimension)
            X = torch.rot90(X, k=1, dims=(2,3))
            with torch.no_grad():
                # Ignore style
                yy, _, yy32 = net(X)
            yy = yy.squeeze()
            yy32 = yy32.squeeze()
            
            if args.save_dir:
                a = yy
                b = yy32
                if not args.no_rotate_saved:
                    # Rotate back
                    a = torch.rot90(yy, k=-1*k, dims=(1,2))
                    b = torch.rot90(yy32, k=-1*k, dims=(1,2))
                name = os.path.splitext(os.path.basename(image))[0]
                np.save(os.path.join(args.save_dir, name + f"_{k*90}_0"), _from_device(a))
                np.save(os.path.join(args.save_dir, name + f"_{k*90}_32"), _from_device(b))

            if args.matching:
                # Perform all-vs-all score
                # and a minimum weight matching of channels
                for A, B, n, name in [(y, yy, 2, 'y'), (y32, yy32, 32, 'y32')]:
                    A = torch.rot90(A[0:n], k=k, dims=(1,2))
                    s = set(range(n))
                    # for the correlation
                    corr = np.zeros((n, n))
                    B = B[0:n].reshape((n,-1))
                    for i in range(0, n):
                        # Compute with torch
                        # This computes an all-vs-all. We only require the
                        # first row against the rest.
                        M = torch.cat((A[i:i+1].reshape(-1).unsqueeze(0), B), dim=0)
                        cc = torch.corrcoef(M)
                        r = cc[0][1:].detach().cpu().numpy()
                        corr[i] = r
                        # Find the best
                        j = np.argmax(np.abs(r))
                        logging.debug(f'Rotation %3d : {name}[{i:2d}][{j:2d}] R=%8.3f',
                             k*90, r[j])
                        s.discard(j)
                    if s:
                        logging.debug('Rotation %3d : Unmatched y%2d %s', k*90, n, s)
                    #cost = 1 - np.abs(corr)
                    cost = 1 - corr**2   # using r^2
                    row, col = linear_sum_assignment(cost)
                    logging.info(f'Rotation %3d : {name} Average R^2=%8.3f',
                         k*90, 1 - cost[row, col].mean())
                    for z in range(0, n):
                        i, j = row[z], col[z]
                        logging.info(f'Rotation %3d : {name}[{i:2d}][{j:2d}] R=%8.3f',
                             k*90, corr[i][j])
                    index = 0 if n == 2 else 1
                    matching[index][k-1][row, col] += 1

            # Rotate original results and compare
            ry = cp_rotate90(y, k=k)
            ry32 = torch.rot90(y32, k=k, dims=(1,2))
            rsy = torch.rot90(sy, k=k, dims=(0,1))
            rp = torch.rot90(p, k=k, dims=(0,1))

            loss_32 = F.mse_loss(yy32, ry32)
            loss_flow = F.mse_loss(yy[:2,:], ry[:2,:])
            # Binary cross entropy compare inputs to a target in [0, 1]
            # We could use p in {0, 1} but here use the sigmoid of y.
            loss_p = F.binary_cross_entropy_with_logits(yy[2], rsy) #p)
            
            pp = F.sigmoid(yy[2])
            iou = jaccard(pp, rp)
            logging.info('Rotation %3d loss : y32 mse %10.6g; flows mse %10.6g; p bce %10.6g; p IoU %10.6g',
                         k*90, loss_32, loss_flow, loss_p, iou)
            del yy
            del yy32
        del X
        del y
        del y32
        
    if args.matching:
        for k in range(1, 4):
            for index, (n, name) in enumerate([(2, 'y'), (32, 'y32')]):
                cost = 1 - matching[index][k-1] / len(images)
                row, col = linear_sum_assignment(cost)
                cost = 1 - cost
                logging.info(f'Rotation %3d : {name} Average match=%8.3f',
                     k*90, cost[row, col].mean())
                for z in range(0, n):
                    i, j = row[z], col[z]
                    logging.info(f'Rotation %3d : {name}[{i:2d}][{j:2d}] match=%8.3f',
                         k*90, cost[i][j])
            

    t = time.time() - start_time
    logging.info(f'Done (in {t:.6g} seconds)')

if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__));
    cellpose_model = os.path.join(base, "cellpose_models", "Nuclei_Hoechst")

    parser = argparse.ArgumentParser(
      description='Program to test Cellpose using a rotated tile.')

    parser.add_argument('image', nargs='+', metavar='IMAGE',
        type=file_path,
        help='image tile')
    parser.add_argument('-m', '--model', dest='model', type=file_path,
        default=cellpose_model,
        help='CellPose model (default: %(default)s)')
    parser.add_argument('--cyto-channel', dest='cyto_channel', type=int,
        default=2,
        help='Cytoplasm channel (1-based index) (default: %(default)s)')
    parser.add_argument('--nuclei-channel', dest='nuclei_channel', type=int,
        default=1,
        help='Nuclei channel (1-based index) (default: %(default)s)')
    parser.add_argument('-d', '--device', dest='device',
        default='cuda',
        help='Device (default: %(default)s)')
    parser.add_argument('-s', '--save', dest='save_dir', type=dir_path,
        help='Save directory prefix (default: %(default)s)')
    parser.add_argument('--no-rotate-saved', dest='no_rotate_saved', action='store_true',
        help='Save the original rotated output. Default is to rotate back.')
    parser.add_argument('--matching', dest='matching', action='store_true',
        help='Perform an all-vs-all matching on the output flows and 32-channel layers.')
    parser.add_argument('--log-level', dest='log_level', type=int,
        default=20,
        help='Log level (default: %(default)s). WARNING=30; INFO=20; DEBUG=10')

    args = parser.parse_args()
    run(args)