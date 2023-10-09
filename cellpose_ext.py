#
# Extend the CellPose network to provide access to the penultimate
# layer in the network
#

from cellpose.resnet_torch import CPnet

class CPnetX(CPnet):
    def __init__(self, nbase, nout, sz,
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
            #T1 = T1.to_dense()
        # Return in the same order as cellpose and append the 32-channel layer
        return T0, style0, T0_32
