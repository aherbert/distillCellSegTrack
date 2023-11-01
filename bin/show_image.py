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

"""
Displays an numpy array image using napari.
"""

import argparse
import os

# Duplicated from cp_distill.bin_utils so this script can
# be run without setting the PYTHONPATH
def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)

def run(args):
    import napari
    import numpy as np

    images = args.image
    if np.isscalar(images):
        images = [images]

    viewer = napari.Viewer()
    for f in images:
        image = np.load(f)
        axis = None if image.ndim == 2 else 0
        viewer.add_image(image, name=os.path.basename(f),
                         channel_axis=axis, blending='additive')
    napari.run()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Program to show the numpy array image[s] using napari.')

    parser.add_argument('image', nargs='+', metavar='IMAGE',
        type=file_path,
        help='image')

    args = parser.parse_args()
    run(args)
