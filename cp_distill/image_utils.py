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

import numpy as np
from skimage.segmentation import clear_border

# Copied from omero_screen.general_functions
def filter_segmentation(mask: np.ndarray) -> np.ndarray:
    """
    removes border objects and filters large abd small objects from segmentation mask
    :param mask: unfiltered segmentation mask
    :return: filtered segmentation mask
    """
    cleared = clear_border(mask, buffer_size=5)
    sizes = np.bincount(cleared.ravel())
    mask_sizes = sizes > 10
    mask_sizes[0] = 0
    cells_cleaned = mask_sizes[cleared]
    return cells_cleaned * mask
