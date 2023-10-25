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
    y : Tensor (3 x X x Y)
        Cellpose output (horizontal flow, vertical flow, map)
    k : int
        Number of 90-degree rotations (should be in [1, 3])

    Raises
    ------
    Exception
        If k is negative.

    Returns
    -------
    y : Tensor (3 x X x Y)
        Rotated Cellpose output (horizontal flow, vertical flow, map)
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
