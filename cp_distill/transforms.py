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

import torch

def cp_rotate90(y, k: int=1):
    """
    Rotate the Cellpose outputs to correspond to an input tile rotation of a
    multiple of 90 degrees. This involves rotating the original outputs
    and a mapping to rearrange and/or negate layers.

    Parameters
    ----------
    y : Tensor (3 x Y x X)
        Cellpose output (vertical flow, horizontal flow, map)
    k : int
        Number of 90-degree rotations (should be in [1, 3])

    Raises
    ------
    Exception
        If k is negative.

    Returns
    -------
    y : Tensor (3 x Y x X)
        Rotated Cellpose output (vertical flow, horizontal flow, map)
    """
    if k < 0:
        raise Exception("k must be positive")

    k = k % 4
    if k == 0:
        return y

    ry = torch.rot90(y, k=k, dims=(1,2))

    # Note: No mapping of the output map (y[2]) is required

    if k == 1:
        # 90 degrees
        # map flows
        ry[[0, 1]] = ry[[1, 0]]
        ry[0] = -ry[0]

    elif k == 2:
        # 180 degrees
        # map flows
        ry[0] = -ry[0]
        ry[1] = -ry[1]

    elif k == 3:
        # 270 degrees
        # map flows
        ry[[0, 1]] = ry[[1, 0]]
        ry[1] = -ry[1]

    return ry

def cp_flip(y, k: int=1):
    """
    Flip the Cellpose outputs horizontally and/or vertically. This involves
    flipping the original outputs and negatation of layers.

    Parameters
    ----------
    y : ND array (3 x Y x X)
        Cellpose output (vertical flow, horizontal flow, map)
    k : int
        Flips (should be in [1, 3]). 1 = horizontal; 2 = vertical; 3 = both.

    Raises
    ------
    Exception
        If k is negative.

    Returns
    -------
    y : ND array (3 x Y x X)
        Flipped Cellpose output (vertical flow, horizontal flow, map)
    """
    if k < 0:
        raise Exception("k must be positive")

    k = k % 4
    if k == 0:
        return y

    if k & 1 == 1:
        # Flip horizontal
        y = y[..., ::-1]

    if k & 2 == 2:
        # Flip vertical
        y = y[:, ::-1]

    # Fix view before updating the underlying data
    y = y.copy()

    # map flows
    if k & 1 == 1:
        y[1] = -y[1]
    if k & 2 == 2:
        y[0] = -y[0]

    return y
