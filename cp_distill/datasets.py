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
import pickle
from torch import from_numpy
from torch.utils.data import Dataset

def find_images(directory, prefix='input'):
    """
    Find all the images in a dataset directory. Images are identified using
    the name:

    prefix_N.npy: image tile

    where prefix is the image prefix; and N is the image number.

    Parameters
    ----------
    directory : pathname
        Dataset directory.

    prefix : str
        Image prefix.

    Returns
    -------
    List of image numbers.
    """
    pattern = re.compile(r'.*' + prefix + r'_(\d+).npy')
    images = []
    for filename in glob.glob(os.path.join(directory, prefix + '_*.npy')):
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

    Data are returned as torch Tensors.

    Parameters
    -------------------

    images: list of image numbers
        Image numbers

    image_directory: str
        Image directory

    load_y32: bool (default: False)
        Load the 32-channel upsample layer. Only load this if it is required
        for the loss function.
    """
    def __init__(self, images, image_directory,
                 load_y32=False):
        self._images = np.array(images).reshape(-1)
        self._directory = image_directory
        if len(images) == 0:
            raise Exception("No image numbers provided")
        self._load_y32 = load_y32
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
        if self._load_y32:
            y32 = np.load(os.path.join(self._directory, f'output32_{n}.npy'))
        else:
            y32 = np.zeros((32, 0, 0), dtype=np.float32)
        # Pad greyscale images to 2 channels
        if x.ndim == 2:
            # Note: No noticeable benefit from caching this
            x = np.array([x, np.zeros(x.shape, dtype=np.float32)])
        return from_numpy(x), from_numpy(y), from_numpy(y32)

# Load Cellpose tiled images from a directory.
class CPTestingDataset(Dataset):
    """
    Dynamically load images from a directory. Images are input
    tiles to the Cellpose network:

    tiles_N.npy: input tile (N x ch x Y x X) [target, nuclei]
    tiles_N.pkl: Pickled data used to reconstruct the image from network output
    mask_N.npy: Cellpose output mask

    where N is the image number.

    Note: The input tiles are padded with a zero array if they are nuclei
    images to create a 2 channel input.

    The image data are numpy arrays and the data used to reconstruct the
    output image is returned as a tuple.

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

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        n = self._images[idx]
        # No validation on inputs
        x = np.load(os.path.join(self._directory, f'tiles_{n}.npy'))
        m = np.load(os.path.join(self._directory, f'mask_{n}.npy'))
        with open(os.path.join(self._directory, f'tiles_{n}.pkl'), 'rb') as f:
            y = pickle.load(f)

        # Pad greyscale images to 2 channels
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
            x[:,1] = 0
        return x, y, m
