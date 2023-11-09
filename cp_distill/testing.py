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
import numpy as np
from cellpose.transforms import average_tiles, resize_image
from cellpose.dynamics import compute_masks
from cellpose.metrics import aggregated_jaccard_index
from cp_distill.image_utils import filter_segmentation
import warnings

def test_network(net, test_data, device, batch_size,
                 cellprob_threshold=0.0,
                 flow_threshold=0.4, interp=True,
                 min_size=15):
    """
    Test the network using the test data.

    Parameters
    ----------
    net : CPnet
        Cellpose network.
    test_data : CPTestingDataset
        Loader for the training data.
    device : torch.device
        Torch device. The network model must be loaded to the same device.
    batch_size : int
        Batch size.
    cellprob_threshold: float (optional, default 0.0) 
        all pixels with value above threshold kept for masks, decrease to find more and larger masks
    flow_threshold: float (optional, default 0.4)
        flow error threshold (all cells with errors below threshold are kept)
    interp: bool (optional, default True)
        interpolate during 2D dynamics
        (in previous versions it was False)
    min_size: int (optional, default 15)
        minimum number of pixels per mask, can turn off with -1

    Returns
    -------
    aji : list float
        Aggegated jaccard index.
    """
    if batch_size < 1:
        batch_size = 1
    use_gpu = device.type == 'cuda'
    net.eval()
    aji = []
    for IMG, data, m1 in test_data:
        # Assumes IMG has been processed by 
        # cellpose.models.CellposeModel._run_cp: to resize the image
        # cellpose.core.UnetModel._run_net: to pad the image
        # cellpose.core.UnetModel._run_tiled: to make tiles
        
        orig_dim, rescale, slc, tile_dim, ysub, xsub, Ly, Lx = data
        # From cellpose.core.UnetModel._run_tiled
        n, nchan, ly, lx = IMG.shape
        niter = int(np.ceil(IMG.shape[0] / batch_size))
        nout = 3 # assume 3 output classes
        y = np.zeros((IMG.shape[0], nout, ly, lx))
        for k in range(niter):
            irange = slice(batch_size*k, min(IMG.shape[0], batch_size*k+batch_size))
            # cellpose.core.UnetModel._to_device
            X = torch.from_numpy(IMG[irange]).float().to(device)
            with torch.no_grad():
                # Ignore styles (and y32 data)
                y0, *_ = net(X)
            # cellpose.core.UnetModel._from_device
            y0 = y0.detach().cpu().numpy()
            y[irange] = y0.reshape(irange.stop-irange.start, y0.shape[-3], y0.shape[-2], y0.shape[-1])

        y = average_tiles(y, ysub, xsub, Ly, Lx)
        # Here we reconstruct the size of the image before calling make_tiles
        y = y[:,:tile_dim[0],:tile_dim[1]]
        
        # From cellpose.core.UnetModel._run_net
        # slice out padding
        y = y[slc]
        # transpose so channels axis is last again
        y = np.transpose(y, (1,2,0))

        # From cellpose.models.CellposeModel._run_cp
        y = resize_image(y, orig_dim[0], orig_dim[1])
        cellprob = y[:,:,2]
        dP = y[:,:,:2].transpose((2,0,1))
        niter = (1 / rescale * 200)
        masks_array, _ = compute_masks(dP, cellprob, niter=niter,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold, interp=interp, 
            min_size=min_size,
            use_gpu=use_gpu, device=device)
        m2 = filter_segmentation(masks_array)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aji.extend(aggregated_jaccard_index([m1], [m2]))

    return np.array(aji)
