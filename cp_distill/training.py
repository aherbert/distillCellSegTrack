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
import torch.nn.functional as F
from torch import nn
from cp_distill.transforms import cp_flip, cp_flip_image

class MapLoss(torch.nn.Module):
    def __init__(self, binary=True, channels=None):
        """
        Create a loss function using the Cellpose map/flow output.
        Uses binary cross entropy with logits. The target y can be
        in [0, 1] using a sigmoid; or converted to {0, 1} if above
        the threshold of 0.5.

        Parameters
        ----------
        binary : boolean (optional, default False)
            Convert the [0, 1] sigmoid to binary.
        channels : int array (optional, default None)
            Channels to target in y.

        Returns
        -------
        An instance.
        """
        super(MapLoss, self).__init__()
        self._binary = binary
        self._channels = None
        if channels:
            # Support only 3 channels in y
            self._channels = list(set(channels).intersection([0,1,2]))

    def forward(self, y, y32, y_pred, y32_pred):
        # y is N x Ch x Y x X
        if self._channels:
            # No requirement for sigmoid in binary mode since we have
            # a fixed threshold of 0.5 for the sigmoid
            if self._binary:
                s = torch.where(y[:,self._channels] > 0, 1.0, 0.0)
            else:
                s = F.sigmoid(y[:,self._channels])
            return F.binary_cross_entropy_with_logits(y_pred[:,self._channels], s, reduction='mean')

        if self._binary:
            s = torch.where(y > 0, 1.0, 0.0)
        else:
            s = F.sigmoid(y)
        return F.binary_cross_entropy_with_logits(y_pred, s, reduction='mean')

class CellposeLoss(torch.nn.Module):
    def __init__(self, zero_background=False, soft_margin=False):
        """
        Create a loss function using the Cellpose map/flow output.
        Uses mse_loss for the flows.
        Uses binary cross entropy with logits or soft margin for the map.

        Parameters
        ----------
        binary : boolean (optional, default False)
            Convert the flows in the background to zero
            (outside the probability map).

        Returns
        -------
        An instance.
        """
        super(CellposeLoss, self).__init__()
        self._zero_background = zero_background
        self._soft_margin = soft_margin
        # Based on cellpose.core.UnetModel._set_criterion
        self.criterion  = nn.MSELoss(reduction='mean')
        if soft_margin:
            self.criterion2 = nn.SoftMarginLoss(reduction='mean')
        else:
            self.criterion2 = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y, y32, y_pred, y32_pred):
        # Based on cellpose.models.CellposeModel.loss_fn
        # y is N x Ch x Y x X
        # Note: Here y corresponds to the Cellpose output and not the
        # flows computed from a mask. Thus we do not require the 5-fold
        # factor: veci = 5. * y[:,:2]
        veci = y[:,:2]
        # convert soft_margin flag for other to -1.0 or 0.0
        lbl = torch.where(y[:,2] > 0, 1.0, -self._soft_margin)
        if self._zero_background:
            veci[:,0] *= lbl
            veci[:,1] *= lbl
        loss = self.criterion(y_pred[:,:2], veci)
        loss /= 2.
        loss2 = self.criterion2(y_pred[:,2], lbl)
        loss = loss + loss2
        return loss

class FlipTransformer:
    def __init__(self, size, flips, rng):
        """
        Create a transformer for training data that flips the input and output
        files.

        Parameters
        ----------
        size : int
            Size of the dataset.
        flips : int
            Type of flips (0=None; 1=Horizontal; 2=Vertical; 3=Both)
        rng : numpy.random._generator.Generator
            Source of randomness.
            Must provide an integers(high, size=size) method.

        Returns
        -------
        An instance.
        """
        self._size = size
        self._flips = flips & 3
        self._rng = rng

    def sample(self):
        """
        Create the random flip for each item. Must be called to initialise
        the random transform for each item.
        """
        if self._flips == 2:
            # Sample k in {0, 2}
            self._k = self._rng.integers(2, size=self._size) * 2
        else:
            # Sample k in [0, flips]
            self._k = self._rng.integers(self._flips + 1, size=self._size)

    def transform(self, idx, x, y):
        """
        Transform the tiles.

        Parameters
        ----------
        idx : int
            Item index.
        x : ND array (2 x Y x X)
            Input tile
        x : ND array (3 x Y x X)
            Output tile (vertical flow, horizontal flow, map)

        Returns
        -------
        An instance.
        """
        k = self._k[idx]
        x = cp_flip_image(x, k)
        y = cp_flip(y, k)
        return x, y

    def __call__(self, *args, **kwargs):
        return self.transform(*args)

def train_epoch(net, train_loader, validation_loader, loss_fn, optimiser, device):
    """
    Perform a training epoch.

    Parameters
    ----------
    net : CPnet
        Cellpose network.
    train_loader : DataLoader
        Loader for the training data.
    validation_loader : DataLoader
        Loader for the validation data.
    loss_fn : torch.nn.Module
        Loss function. This must have the following signature for the
        forward method: (y, y32, y_pred, y32_pred) -> loss.
    optimiser : torch.optim.Optimizer
        Optimiser. Used to perform a step for each batch in the training data.
    device : torch.device
        Torch device. The network model must be loaded to the same device.

    Returns
    -------
    train_loss : float
        Training loss.
    val_loss : float
        Validation loss.
    """

    # Asynchronous GPU copy

    # Training
    net.train()
    train_loss = 0
    n = 0
    data_iter = iter(train_loader)

    next_batch = next(data_iter)
    next_batch = [ _.to(device, non_blocking=True) for _ in next_batch ]

    for i in range(len(train_loader)):
        x, y, y32 = next_batch
        if i + 1 != len(train_loader):
            # start copying data of next batch
            next_batch = next(data_iter)
            next_batch = [ _.to(device, non_blocking=True) for _ in next_batch ]

        m = len(x)
        n += m

        y_pred, _, y32_pred = net(x)
        del x

        loss = loss_fn(y, y32, y_pred, y32_pred)
        train_loss += loss.item() * m

        # update model parameters
        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)

        del y
        del y32
        del y_pred
        del y32_pred
        torch.cuda.empty_cache()

    train_loss /= n

    # Validation
    net.eval()
    val_loss = 0
    n = 0
    data_iter = iter(validation_loader)

    next_batch = next(data_iter)
    next_batch = [ _.to(device, non_blocking=True) for _ in next_batch ]

    for i in range(len(validation_loader)):
        x, y, y32 = next_batch
        if i + 1 != len(validation_loader):
            # start copying data of next batch
            next_batch = next(data_iter)
            next_batch = [ _.to(device, non_blocking=True) for _ in next_batch ]

        m = len(x)
        n += m

        with torch.no_grad():
            y_pred, _, y32_pred = net(x)
        del x

        loss = loss_fn(y, y32, y_pred, y32_pred)
        val_loss += loss.item() * m

        del y
        del y32
        del y_pred
        del y32_pred
        torch.cuda.empty_cache()

    val_loss /= n

    torch.cuda.empty_cache()

    return train_loss, val_loss
