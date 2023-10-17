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

# Pytorch datasets

import os
import re
import glob
import numpy as np
from torch.utils.data import Dataset

def find_images(directory):
    """
    Find all the images in a dataset directory. Images are identified using
    the input tile name:

    input_N.npy: input tile
    
    where N is the image number.

    Parameters
    ----------
    directory : pathname
        Dataset directory.

    Returns
    -------
    List of image numbers.
    """
    pattern = re.compile(r".*input_(\d+).npy")
    images = []
    for filename in glob.glob(os.path.join(directory, 'input_*.npy')):
        result = pattern.match(filename)
        if result:
            images.append(int(result.group(1)))
    images.sort()
    return images

# Load Cellpose input and output tiles from a directory.
# This class ignores any style output from Cellpose.
class CPDataset(Dataset):
    """
    Dynamically load images from a directory. Images are input
    and output tiles from the Cellpose network:
    
    input_N.npy: input tile (2 x Y x X) [target, nuclei]
    output_N.npy: output tile (3 x Y x X) [horizontal flow, vertical flow, map]
    output32_N.npy: output tile (32 x Y x X) [penultimate Cellpose upsample layer]

    where N is the image number.
    
    Note: The input tiles are padded with a zero array if they are greyscale
    images to create a 2 channel input.

    Parameters
    -------------------

    images: list of image numbers
        Image numbers 

    image_directory: str
        Image directory
    """
    def __init__(self, images, image_directory):
        self._images = np.array(images).reshape(-1)
        self._directory = image_directory
        if len(images) == 0:
            raise Exception("No image numbers provided")
        # This constructor could validate all images exist.
        # Currently an invalid constructor will throw errors when
        # retrieving a dataset item

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        n = self._images[idx]
        # No validation on inputs
        x = np.load(os.path.join(self._directory, f'input_{n}.npy'))
        y = np.load(os.path.join(self._directory, f'output_{n}.npy'))
        y32 = np.load(os.path.join(self._directory, f'output32_{n}.npy'))
        # Pad greyscale images to 2 channels
        if x.ndim == 2:
            x = np.array([x, np.zeros(x.shape)])
        return x, y, y32
