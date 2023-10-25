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

class MapLoss(torch.nn.Module):
    def __init__(self, binary=False):
        """
        Create a loss function using the Cellpose map output.
        Uses binary cross entropy with logits. The target y can be
        in [0, 1] using a sigmoid; or converted to {0, 1} if above
        the threshold of 0.5.

        Parameters
        ----------
        binary : boolean (optional, default False)
            Convert the [0, 1] sigmoid to binary.

        Returns
        -------
        An instance.
        """
        super(MapLoss, self).__init__()
        self._binary = binary

    def forward(self, y, y32, y_pred, y32_pred):
        # y is N x Ch x Y x X; extract channel 2
        s = F.sigmoid(y[:,2])
        if self._binary:
            s = torch.where(s > 0.5, 1.0, 0.0)
        return F.binary_cross_entropy_with_logits(y_pred[:,2], s, reduction='mean')

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

    # Training
    net.train()
    train_loss = 0
    for x, y, y32 in train_loader:

        if device is not None:
            # sending the data to the device (cpu or GPU)
            x, y, y32 = x.to(device), y.to(device), y32.to(device)

        y_pred, _, y32_pred = net(x)
        del x

        loss = loss_fn(y, y32, y_pred, y32_pred)
        train_loss += loss.item()

        # update model parameters
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        del y
        del y32
        del y_pred
        del y32_pred
        torch.cuda.empty_cache()

    train_loss /= len(train_loader)

    # Validation
    net.eval()
    val_loss = 0
    for x, y, y32 in validation_loader:

        if device is not None:
            # sending the data to the device (cpu or GPU)
            x, y, y32 = x.to(device), y.to(device), y32.to(device)

        with torch.no_grad():
            y_pred, _, y32_pred = net(x)
        del x

        loss = loss_fn(y, y32, y_pred, y32_pred)
        val_loss += loss.item()

        del y
        del y32
        del y_pred
        del y32_pred
        torch.cuda.empty_cache()

    val_loss /= len(validation_loader)

    torch.cuda.empty_cache()

    return train_loss, val_loss
