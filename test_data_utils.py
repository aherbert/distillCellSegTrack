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
import psutil

def file_or_dir_path(string):
    if os.path.isfile(string) or os.path.isdir(string):
        return string
    else:
        raise FileNotFoundError(string)

def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)

def run(args):
    from data_utils import get_training_and_validation_loaders
    import time
    import logging

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
    train_loader, validation_loader = get_training_and_validation_loaders(
        args.model, args.image, channel=args.channel, device=args.device,
        augment=args.augment)
    t = time.time() - start_time
    n = len(train_loader)
    m = len(validation_loader)
    logging.info(f'Loaded data : Training {n} : Validation {m} (in {t} seconds)')
    

if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__));
    cellpose_model_directory = os.path.join(base, "cellpose_models", "Nuclei_Hoechst")

    parser = argparse.ArgumentParser(
      description='Program to load the training and validation data.')
    
    parser.add_argument('image', nargs='+', metavar='IMAGE',
        type=file_or_dir_path,
        help='image or image directory')
    parser.add_argument('-m', '--model', dest='model', type=file_path,
        default=cellpose_model_directory,
        help='CellPose model (default: %(default)s)')
    parser.add_argument('-c', '--channel', dest='channel', type=int,
        help='Channel (default: %(default)s)')
    parser.add_argument('-d', '--device', dest='device',
        default='cuda',
        help='Device (default: %(default)s)')
    parser.add_argument('--augment', dest='augment', action='store_true',
        help='Perform argmentations')
    parser.add_argument('--memory', dest='memory', action='store_true',
        help='Debug memory usage')

    args = parser.parse_args()
    run(args)
