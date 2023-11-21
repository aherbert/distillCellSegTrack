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
Displays a numpy array image using napari.
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
    bsize = args.tile_size
    tile_overlap = args.tile_overlap
    for f in images:
        image = np.load(f)
        axis = None if image.ndim == 2 else args.axis
        viewer.add_image(image, name=os.path.basename(f),
                         channel_axis=axis, blending='additive')
        # Draw tiles
        if bsize and image.ndim < 4:
            Ly, Lx = image.shape if image.ndim == 2 else np.delete(image.shape, axis)
            # Adapted from cellpose.transforms.make_tiles
            bsizeY, bsizeX = min(bsize, Ly), min(bsize, Lx)
            bsizeY = np.int32(bsizeY)
            bsizeX = np.int32(bsizeX)
            # tiles overlap by 10% tile size
            ny = 1 if Ly<=bsize else int(np.ceil((1.+2*tile_overlap) * Ly / bsize))
            nx = 1 if Lx<=bsize else int(np.ceil((1.+2*tile_overlap) * Lx / bsize))
            ystart = np.linspace(0, Ly-bsizeY, ny).astype(int)
            xstart = np.linspace(0, Lx-bsizeX, nx).astype(int)
            polygons = []
            for j in range(len(ystart)):
                y, yy = ystart[j], ystart[j]+bsizeY
                for i in range(len(xstart)):
                    x, xx = xstart[i], xstart[i]+bsizeX
                    polygons.append(np.array([[y, x], [y, xx], [yy, xx], [yy, x], [y, x]]))
            viewer.add_shapes(polygons, shape_type='Path', name='tiles',
                edge_color='#ff5500ff', edge_width=5)

    napari.run()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Program to show the numpy array image[s] using napari.')

    parser.add_argument('image', nargs='+', metavar='IMAGE',
        type=file_path,
        help='image')
    parser.add_argument('--axis', type=int,
        default=0,
        help='Channel axis for ND images (default: %(default)s)')
    parser.add_argument('--tile-size', type=int,
        default=0,
        help='Tile size to overlay (default: %(default)s). Cellpose uses 224.')
    parser.add_argument('--tile-overlap', type=float,
        default=0.1,
        help='Tile overlap (default: %(default)s)')

    args = parser.parse_args()
    run(args)
