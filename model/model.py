from model.base_model import BaseModel
from model.utils import conv, sequential


class DIP(BaseModel):
    def __init__(self, in_nc=1, out_nc=1, nc=100):
        super(DIP, self).__init__()
        head = conv(in_nc, nc, mode="CPR")
        body = [conv(nc, nc, mode="CPR") for _ in range(16)]
        tail = conv(nc, out_nc, mode="CPR")
        self.model = sequential(head, *body, tail)

    def forward(self, x):
        return self.model(x.unsqueeze(0).unsqueeze(0))


class DnCNN(BaseModel):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(DnCNN, self).__init__()

        m_head = conv(in_nc, nc, mode="CR", bias=True)
        m_body = [conv(nc, nc, mode="CR", bias=True) for _ in range(nb - 2)]
        m_tail = conv(nc, out_nc, mode="C", bias=True)

        self.model = sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        n = self.model(x)
        return x - n
