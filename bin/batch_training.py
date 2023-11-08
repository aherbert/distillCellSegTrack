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
    import re
    import os
    import logging
    import subprocess

    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s - %(message)s',
        level=logging.INFO)

    # Load the training settings
    logging.info(f'Loading training settings: {args.batch}')
    with open(args.batch) as f:
        training = f.readlines()
    used = set()
    for line in training:
        line = line.strip()
        if len(line) == 0 or line[0] == '#':
            continue
        arguments = re.split(r'\s+', line)
        cmd = ['test_training.py', '--wandb']
        cmd.extend(arguments)
        # Identify the output file
        prefix = os.path.basename(arguments[0])
        n = 0
        while True:
            n += 1
            base = f'{prefix}{n}'
            if base in used:
                continue
            if not (os.path.isfile(base + '.pt') or os.path.isfile(base + '.json')):
                # This output file is OK
                break
        used.add(base)
        cmd.extend(['-n', base + '.pt',
                    '-s', base + '.json'])
        out = base + '.out'

        # Create a name
        if not '--run-name' in arguments:
            print('create run name')
            # Start with the dataset name
            # (remove the default prefix and model suffix)
            name = [arguments[0].replace('test_data_', '').replace('_Hoechst', '')]
            for a in arguments[1:]:
                # Detect new argument
                if a[0] == '-':
                    size = 0
                    # Ignore some arguments
                    if a in ['-d', '--device']:
                        size -= 1
                    else:
                        name.append(a.lstrip('-'))
                    continue
                if size < 0:
                    continue
                if size > 0:
                    name[-1] = name[-1] + ','
                name[-1] = name[-1] + a
                size += 1

            cmd.extend(['--run-name', '_'.join(name)])

        logging.info(f'Run {cmd} > {out}')
        if args.dry_run:
            continue
        f = open(out, "w")
        subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=f)
    logging.info('Done')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Program to run the training script for a batch file of arguments.')

    parser.add_argument('batch', metavar='BATCH', type=file_path,
        help='Batch arguments file')
    parser.add_argument('--dry-run',
        default=False, action=argparse.BooleanOptionalAction,
        help='Perform a dry run (default: %(default)s)')

    args = parser.parse_args()
    run(args)
