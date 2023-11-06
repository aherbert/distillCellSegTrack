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
    import json
    import logging
    import torch

    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s - %(message)s',
        level=logging.INFO)

    device = torch.device(args.device)
    if not args.file:
        logging.info(f'Loading model state: {args.model}')
        state = torch.load(args.model, map_location=device)
        del state['model_state_dict']
        s = json.dumps(state, indent=2)
        logging.info(f'Model state: {s}')
        return

    if os.path.exists(args.model):
        if not args.force:
            logging.error(f'Output model file exists: {args.model}')
            exit(1)

    # Load the training settings
    logging.info(f'Loading training settings: {args.file}')
    with open(args.file) as f:
        training = json.load(f)

    # Find the training dataset settings file
    filename = os.path.join(training['directory'], 'settings.json')
    logging.info(f'Loading training dataset settings: {filename}')
    with open(filename) as f:
        dataset = json.load(f)

    # Load the (best) checkpoint
    filename = training['name']
    if os.path.exists(filename + '.best'):
        filename += '.best'
    logging.info(f'Loading checkpoint: {filename}')
    checkpoint = torch.load(filename, map_location=device)

    # Create model state
    state = {}
    for k in ['nbase', 'residual_on', 'style_on', 'concatenation']:
        state[k] = training[k]
    for k in ['model', 'cyto']:
        state[k] = dataset[k]
    for k in ['epoch', 'loss']:
        state[k] = checkpoint[k]

    s = json.dumps(state, indent=2)
    logging.info(f'Model state: {s}')

    # Save model
    state['model_state_dict'] = checkpoint['model_state_dict']
    logging.info(f'Saving model state: {args.model}')
    torch.save(state, args.model)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Program to extract a model from a training settings file ' +
            'to a model state file, or display a model state file.')

    parser.add_argument('file', nargs='?', metavar='FILE',
        type=file_path,
        help='Training settings file (JSON)')
    parser.add_argument('model', metavar='MODEL', type=str,
        help='Model state file')
    parser.add_argument('-f', '--force', dest='force', action='store_true',
        help='Overwrite existing model state file')
    parser.add_argument('-d', '--device', dest='device',
        default='cpu',
        help='Device (default: %(default)s)')

    args = parser.parse_args()
    run(args)
