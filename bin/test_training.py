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
from cp_distill.bin_utils import file_or_dir_path
from enum import Enum

class Existing(str, Enum):
    overwrite = 'overwrite'
    load = 'load'
    error = 'error'
    def __str__(self):
        return self.value

class StoppingCriteria:
    def __init__(self, patience=3, min_delta=0.0, min_rel_delta=1e-4):
        """
        Create an instance.

        Parameters
        ----------
        patience : int, optional
            Number of times to allow for no improvement before stopping the execution.
            The default is 3.
        min_delta : float, optional
            The minimum absolute change to be counted as improvement.
            The default is 0.0.
        min_rel_delta : float, optional
            The minimum relative change to be counted as improvement.
            The default is 1e-4.

        Returns
        -------
        None.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.min_rel_delta = min_rel_delta
        self.counter = 0
        self.min_value = 1e300

    def check(self, value):
        """
        Test if the value has converged.

        Parameters
        ----------
        value : float
            Value.

        Returns
        -------
        bool
            If the value has converged.
        """
        obs = value + self.min_delta + self.min_rel_delta * self.min_value
        target = self.min_value
        if (obs < target):
            # Improvement
            self.min_value = value
            self.counter = 0
        else:
            # No improvement. Check how many times this has happened.
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def run(args):
    import os
    import time
    import logging
    import numpy as np
    import torch
    import shutil
    from torch.utils.data import DataLoader
    from cp_distill.datasets import find_images, CPDataset
    from cp_distill.cellpose_ext import CPnetX
    from cp_distill.training import CellposeLoss, train_epoch
    from sklearn.model_selection import train_test_split
    if args.wandb:
        import wandb

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
    )

    # Create optimizer
    # Note: Cellpose uses:
    # optim.RAdam(self.net.parameters(), lr=learning_rate, betas=(0.95, 0.999),
    #                                    eps=1e-08, weight_decay=weight_decay)
    # This changes the beta from the default of (0.9, 0.999)
    # UnetModel._train_net has a default learning_rate=0.2
    epoch = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate,
        betas=(0.95, 0.999))
    best_loss = 1e300

    # Use existing checkpoint
    checkpoint_name = args.name
    if os.path.isfile(checkpoint_name):
        if args.existing == Existing.error:
            logging.error(f'Checkpoint exists: {checkpoint_name}')
            exit(1)
        if args.existing == Existing.load:
            checkpoint = torch.load(checkpoint_name)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            net.train()
            logging.info(f'Loaded checkpoint: {checkpoint_name}')
        if args.existing == Existing.overwrite:
            logging.info(f'Existing checkpoint will be overwritten: {checkpoint_name}')

    net = net.to(device)

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
    loss_fn = CellposeLoss()
    # Worse using BCE on the flows. They must have the same magnitude.
    #loss_fn = MapLoss(binary=True)
    # Worse if the target is not binary
    #loss_fn = MapLoss(binary=False)
    # Much worse to ignore the flows
    #loss_fn = MapLoss(binary=False, channels=[2])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    # Training loss is expected to go down. Stop when at an approximate plateau.
    train_stop = StoppingCriteria(patience=1, min_rel_delta=1e-2)
    # Validation loss may increase due to overtraining. This is the main
    # control point over early termination.
    val_stop = StoppingCriteria(patience=args.patience, min_delta=args.delta,
       min_rel_delta=args.rel_delta)

    if args.wandb:
        wandb.init(project=args.project, config=vars(args))

    for i in range(args.epochs):
        epoch += 1
        train_loss, val_loss = \
            train_epoch(net, train_loader, validation_loader, loss_fn,
                        optimizer, device)
        better = False
        if val_loss < best_loss:
            best_loss = val_loss
            better = True
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'best_loss': best_loss,
            }, checkpoint_name)
        if better:
            shutil.copy2(checkpoint_name, checkpoint_name + '.best')
        logging.info('[%d] Loss train %s : validation %s', epoch, train_loss, val_loss)
        if args.wandb:
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss})
        if train_stop.check(train_loss) and val_stop.check(val_loss):
            logging.info('[%d] Stopping due to no improvement', epoch)
            break
        scheduler.step()

    if args.wandb:
        wandb.finish()
    t = time.time() - start_time
    logging.info(f'Done (in {t:.6g} seconds)')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
      description='Program to test training a model using a Cellpose tile dataset.')

    parser.add_argument('directory', metavar='DIR',
        type=file_or_dir_path,
        help='Dataset directory, or training state file')
    parser.add_argument('--size', dest='size', type=int,
        default=0,
        help='Number of tiles to use (default is all tiles)')
    parser.add_argument('--data-seed', dest='data_seed', type=int,
        default=42,
        help='Random seed to select data (default: %(default)s)')
    parser.add_argument('-d', '--device', dest='device',
        default='cuda',
        help='Device (default: %(default)s)')
    parser.add_argument('-s', '--state', dest='state', type=str,
        default='training.json',
        help='Training state file (default: %(default)s)')

    group = parser.add_argument_group('Model')
    group.add_argument('-n', '--name', dest='name', type=str,
        default='model.pt',
        help='Checkpoint name (default: %(default)s)')
    group.add_argument('--existing', dest='existing', type=Existing,
        choices=list(Existing), default=Existing.error,
        help='Existing checkpoint option (default: %(default)s)')
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
        default=0xdeadbeef,
        help='Random seed for initial model (default: %(default)s)')
    group.add_argument('--test-size', dest='test_size', type=float,
        default=0.25,
        help='Size for the test data (default: %(default)s)')
    group.add_argument('--lr', dest='learning_rate', type=float,
        default=0.01,
        help='The learning rate (default: %(default)s)')
    group.add_argument('--patience', dest='patience', type=int,
        default=3,
        help='Number of times to allow for no improvement before stopping (default: %(default)s)')
    group.add_argument('--delta', dest='delta', type=float,
        default=0,
        help='The minimum absolute change to be counted as improvement (default: %(default)s)')
    group.add_argument('--rel-delta', dest='rel_delta', type=float,
        default=1e-4,
        help='The minimum relative change to be counted as improvement (default: %(default)s)')

    group = parser.add_argument_group('Misc')
    group.add_argument('--log-level', dest='log_level', type=int,
        default=20,
        help='Log level (default: %(default)s). WARNING=30; INFO=20; DEBUG=10')
    parser.add_argument('--wandb', dest='wandb', action='store_true',
        help='Log to Weights and Biases (must be logged in)')
    group.add_argument('--project', dest='project', type=str,
        default='cellpose-distill',
        help='Weights and Biases project (default: %(default)s)')

    args = parser.parse_args()

    # Support save/continue of a training run
    import os
    import json

    args_d = vars(args)
    if os.path.isfile(args.directory):
        # Continue training
        with open(args.directory) as f:
            d = json.load(f)
            # To continue training we must load the existing checkpoint.
            # This avoids have to convert the 'existing' str to an enum.
            d['existing'] = Existing.load
        # Merge with script arguments.
        # Allow some arguments to override the previous training state.
        import sys
        saved = {}
        for s in ['log-level', 'epochs']:
           if '--' + s in sys.argv:
                s = s.replace('-', '_')
                saved[s] = args_d[s]
        args_d.update(d)
        args_d.update(saved)
    else:
        # Save training state
        with open(args.state, 'w') as f:
            json.dump(args_d, f)

    run(args)
