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
    import time
    import logging
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from cp_distill.datasets import find_images, CPDataset
    from cp_distill.cellpose_ext import CPnetX
    from cp_distill.training import MapLoss, train_epoch
    from sklearn.model_selection import train_test_split

    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s - %(message)s',
        level=args.log_level)

    start_time = time.time()
    
    # Create Cellpose network
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    net = CPnetX(
        nbase=args.nbase, nout=3, sz=3,
        residual_on=args.residual_on, style_on=args.style_on,
        concatenation=args.concatenation,
        # cannot train with mkldnn
        mkldnn=False,
    ).to(device)

    # Create data    
    images = find_images(args.directory)
    logging.info(f'Processing dataset: {args.directory} : tiles = {len(images)}')
    size = len(images)
    if args.size > 0:
        size = np.min([size, args.size])
    rng = np.random.default_rng(seed=args.data_seed)
    y, z = train_test_split(rng.choice(images, size, replace=False),
                            test_size=args.test_size, shuffle=False)
    logging.info(f'Size train {len(y)} : validation {len(z)}')
    
    train_loader = DataLoader(CPDataset(y, args.directory), batch_size=args.batch_size)
    validation_loader = DataLoader(CPDataset(z, args.directory), batch_size=args.batch_size)
    
    # Create training objects
    loss_fn = MapLoss(binary=False)
    optimiser = torch.optim.Adam(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=20, gamma=0.1)
    
    best_loss = 1e300
    for epoch in range(args.epochs):
        train_loss, val_loss = \
            train_epoch(net, train_loader, validation_loader, loss_fn, optimiser, 
                       device)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), args.name)
        logging.info('[%d] Loss train %s : validation %s', epoch+1, train_loss, val_loss)
        scheduler.step()
    
    t = time.time() - start_time
    logging.info(f'Done (in {t:.6g} seconds)')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
      description='Program to test training a model using a Cellpose tile dataset.')

    parser.add_argument('directory', metavar='DIR',
        type=dir_path,
        help='Dataset directory')
    parser.add_argument('--size', dest='size', type=int,
        default=0,
        help='Number of tiles to use (default is all tiles)')
    parser.add_argument('--data-seed', dest='data_seed', type=int,
        default=42,
        help='Random seed to select data (default: %(default)s)')
    parser.add_argument('-d', '--device', dest='device',
        default='cuda',
        help='Device (default: %(default)s)')

    group = parser.add_argument_group('Model')
    group.add_argument('-n', '--name', dest='name', type=str,
        default='cp_model',
        help='Model name (default: %(default)s)')
    group.add_argument('--nbase', nargs='+', dest='nbase', type=int,
        default=[2, 32],
        help='Cellpose architecture (default: %(default)s). ' +
             '(Note: Cellpose uses [2, 32, 64, 128, 256].)')
    group.add_argument('--residual_on', dest='residual_on', type=bool,
        default=True,
        help='Residual on (default: %(default)s)')
    group.add_argument('--style_on', dest='style_on', type=bool,
        default=True,
        help='Style on (default: %(default)s)')
    group.add_argument('--concatenation', dest='concatenation', type=bool,
        default=False,
        help='Concatenation (default: %(default)s)')

    group = parser.add_argument_group('Training')
    group.add_argument('--epochs', dest='epochs', type=int,
        default=1,
        help='Training epochs (default: %(default)s)')
    group.add_argument('--batch-size', dest='batch_size', type=int,
        default=8,
        help='Batch size for the data loader (default: %(default)s)')
    group.add_argument('--seed', dest='seed', type=int,
        default=42,
        help='Random seed for initial model (default: %(default)s)')
    group.add_argument('--test-size', dest='test_size', type=float,
        default=0.25,
        help='Size for the test data (default: %(default)s)')

    group = parser.add_argument_group('Misc')
    group.add_argument('--log-level', dest='log_level', type=int,
        default=20,
        help='Log level (default: %(default)s). WARNING=30; INFO=20; DEBUG=10')

    args = parser.parse_args()
    run(args)
