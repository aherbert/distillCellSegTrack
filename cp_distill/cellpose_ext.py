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

# CellPose extensions

import numpy as np
import os
import torch
import numba as nb

from cellpose.resnet_torch import CPnet
from cellpose.models import CellposeModel
from torch.utils import mkldnn as mkldnn_utils

# short-circuiting replacement for np.any()
# see: https://stackoverflow.com/a/53474543
@nb.jit(nopython=True)
def _sc_any(array):
    for x in array.flat:
        if x:
            return True
    return False

# Extend the CellPose network to provide access to the penultimate
# layer in the network
# Adapted from cellpose/resnet_torch.py
class CPnetX(CPnet):
    def __init__(self, nbase, nout=3, sz=3,
                residual_on=True, style_on=True,
                concatenation=False, mkldnn=False,
                diam_mean=30.):
        super(CPnetX, self).__init__(nbase, nout, sz, residual_on, style_on,
                                     concatenation, mkldnn, diam_mean)

    def forward(self, data, training_data=False):
        # This method is copied from the CPnet super-class

        if self.mkldnn:
            data = data.to_mkldnn()
        T0    = self.downsample(data)
        if self.mkldnn:
            style = self.make_style(T0[-1].to_dense())
        else:
            style = self.make_style(T0[-1])
        style0 = style
        if not self.style_on:
            style = style * 0
        T0 = self.upsample(style, T0, self.mkldnn)
        # Extension to save the 32-channel output
        T0_32 = T0
        #print('should be 32:',T0.shape)
        T0    = self.output(T0)

        if self.mkldnn:
            T0 = T0.to_dense()
            T0_32 = T0_32.to_dense()
        # Return in the same order as cellpose and append the 32-channel layer
        return T0, style0, T0_32

# Extend the CellposeModel to use the custom CellPose network.
# Adapted from cellpose/models.py
class CellposeModelX(CellposeModel):
    """

    Parameters
    -------------------

    gpu: bool (optional, default False)
        whether or not to save model to GPU, will check if GPU available

    pretrained_model: str or list of strings (optional, default False)
        full path to pretrained cellpose model(s), if None or False, no model loaded

    model_type: str (optional, default None)
        any model that is available in the GUI, use name in GUI e.g. 'livecell'
        (can be user-trained or model zoo)

    net_avg: bool (optional, default False)
        loads the 4 built-in networks and averages them if True, loads one network if False

    diam_mean: float (optional, default 30.)
        mean 'diameter', 30. is built in value for 'cyto' model; 17. is built in value for 'nuclei' model;
        if saved in custom model file (cellpose>=2.0) then it will be loaded automatically and overwrite this value

    device: torch device (optional, default None)
        device used for model running / training
        (torch.device('cuda') or torch.device('cpu')), overrides gpu input,
        recommended if you want to use a specific GPU (e.g. torch.device('cuda:1'))

    residual_on: bool (optional, default True)
        use 4 conv blocks with skip connections per layer instead of 2 conv blocks
        like conventional u-nets

    style_on: bool (optional, default True)
        use skip connections from style vector to all upsampling layers

    concatenation: bool (optional, default False)
        if True, concatentate downsampling block outputs with upsampling block inputs;
        default is to add

    nchan: int (optional, default 2)
        number of channels to use as input to network, default is 2
        (cyto + nuclei) or (nuclei + zeros)

    save_directory: str (optional, default None)
        directory to save input to and output from the network (images
        will be saved with a counter suffix)

    save_y32: bool (optional, default False)
        if True, save the 32-channel upsample layer

    save_styles: bool (optional, default False)
        if True, save the styles
    """

    def __init__(self, gpu=False, pretrained_model=False,
                    model_type=None, net_avg=False,
                    diam_mean=30., device=None,
                    residual_on=True, style_on=True, concatenation=False,
                    nchan=2,
                    save_directory=None,
                    save_y32=False, save_styles=False):
        super(CellposeModelX, self).__init__(
            gpu=gpu, pretrained_model=pretrained_model,
            model_type=model_type, net_avg=net_avg,
            diam_mean=diam_mean, device=device,
            residual_on=residual_on, style_on=style_on, concatenation=concatenation,
            nchan=nchan)

        # Validate save directory
        self._save_directory = None
        self._save_y32 = save_y32
        self._save_styles = save_styles
        if save_directory:
            if not os.path.isdir(save_directory):
                raise NotADirectoryError(save_directory)
            self._save_directory = save_directory
        self._count = 1

        # The network is created in cellpose.core.UnetModel.
        # Here we replace the network with our custom version.
        # Code adapted from cellpose/core.py:UnetModel.__init__
        self.net = CPnetX(self.nbase,
                          self.nclasses,
                          sz=3,
                          residual_on=residual_on,
                          style_on=style_on,
                          concatenation=concatenation,
                          mkldnn=self.mkldnn,
                          diam_mean=diam_mean).to(self.device)

        # Reinitialise network
        # Code adapted from cellpose/models.py:CellposeModel.__init__
        # This duplication of loading a model roughly doubles construction time
        # from ~1.2s to 2.4s
        if self.pretrained_model:
            self.net.load_model(self.pretrained_model[0], device=self.device)
            self.diam_mean = self.net.diam_mean.data.cpu().numpy()[0]
            self.diam_labels = self.net.diam_labels.data.cpu().numpy()[0]
            # Removed logging of model details

    def network(self, x, return_conv=False):
        """ convert imgs to torch and run network model and return numpy """
        # This method is copied from the UnetModel super-class with
        # modifications to save the network input and output

        X = self._to_device(x)
        self.net.eval()
        if self.mkldnn:
            self.net = mkldnn_utils.to_mkldnn(self.net)
        with torch.no_grad():
            y, style, y32 = self.net(X)
        del X
        y = self._from_device(y)
        style = self._from_device(style)

        # Commented out. This appears to be legacy code that is not used
        # since conv is not defined.
        #if return_conv:
        #    conv = self._from_device(conv)
        #    y = np.concatenate((y, conv), axis=1)

        # Save input/output
        if self._save_directory:
            y32 = self._from_device(y32)

            # save individual tiles
            # Tiles dimensions: ntiles x nchan x bsize x bsize
            # (bsize = size of tiles)
            n = len(x)
            for i in range(n):
                a = x[i]
                # greyscale images are padded with a zero channel
                # which can be removed to save space
                if not _sc_any(a[1]):
                    a = a[0].squeeze()
                np.save(os.path.join(self._save_directory, f'input_{self._count+i}.npy'), a)
                np.save(os.path.join(self._save_directory, f'output_{self._count+i}.npy'), y[i])
                if self._save_styles:
                    np.save(os.path.join(self._save_directory, f'style_{self._count+i}.npy'), style[i])
                if self._save_y32:
                    np.save(os.path.join(self._save_directory, f'output32_{self._count+i}.npy'), y32[i])

            self._count += n

        return y, style

    def _run_cp(self, x, compute_masks=True, normalize=True, invert=False,
                rescale=1.0, net_avg=False, resample=True,
                augment=False, tile=True, tile_overlap=0.1,
                cellprob_threshold=0.0,
                flow_threshold=0.4, min_size=15,
                interp=True, anisotropy=1.0, do_3D=False, stitch_threshold=0.0,
                ):
        masks, styles, dP, cellprob, p = super(CellposeModelX, self)._run_cp(x,
            compute_masks, normalize, invert,
            rescale, net_avg, resample,
            augment, tile, tile_overlap,
            cellprob_threshold,
            flow_threshold, min_size,
            interp, anisotropy, do_3D, stitch_threshold);
        self.last_rescale = rescale
        return masks, styles, dP, cellprob, p
