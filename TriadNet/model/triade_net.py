import torch
from torch import nn
import numpy as np

from TriadNet.model.blocks import get_convolution_operator
from TriadNet.model.blocks import DownBlockNd, UpBlockNd, BottomBlockNd
from TriadNet.model.blocks import GridAttentionBlockND

"""
3 output convs instead of 3 output decoder
"""


class TriadeNet(nn.Module):
    def __init__(self,
                 dim: int,
                 in_channels: int,
                 out_channels: int,
                 dropout: float = 0.,
                 norm: str = 'batch'):
        super().__init__()

        n = 16  # start filter
        self.dim = dim
        self.out_channels = out_channels
        self.norm = norm
        self.dropout = dropout
        self.features = None

        # define basic operators
        assert self.dim in [2, 3], 'Supported dimensions are [2, 3] but you provided {}'.format(self.dim)

        # define architecture
        self.dblock1 = DownBlockNd(dim=dim, in_channels=in_channels, out_channels=2 * n, norm=self.norm,
                                   dropout=self.dropout)
        self.dblock2 = DownBlockNd(dim=dim, norm=self.norm, dropout=self.dropout, in_channels=2 * n, out_channels=4 * n)
        self.dblock3 = DownBlockNd(dim=dim, norm=self.norm, dropout=self.dropout, in_channels=4 * n, out_channels=8 * n)
        self.bottom_block = BottomBlockNd(dim=dim, norm=self.norm, dropout=self.dropout, in_channels=8 * n,
                                          out_channels=16 * n)

        # attention gates --> no dropout
        self.attblock1 = GridAttentionBlockND(in_channels=8 * n, gating_channels=16 * n,
                                              inter_channels=8 * n, dropout=0., norm=norm,
                                              dimension=self.dim)
        self.attblock2 = GridAttentionBlockND(in_channels=4 * n, gating_channels=8 * n, inter_channels=4 * n,
                                              dropout=0., norm=norm,
                                              dimension=self.dim)
        self.attblock3 = GridAttentionBlockND(in_channels=2 * n, gating_channels=4 * n, inter_channels=2 * n,
                                              dropout=0., norm=norm,
                                              dimension=self.dim)

        self.ublock1 = UpBlockNd(dim=dim, norm=self.norm, dropout=self.dropout,
                                 down_in_channels=16 * n, left_in_channels=8 * n, out_channels=8 * n)
        self.ublock2 = UpBlockNd(dim=dim, norm=self.norm, dropout=self.dropout,
                                 down_in_channels=8 * n, left_in_channels=4 * n, out_channels=4 * n)
        # heads
        self.ublock3_mean = UpBlockNd(dim=dim, norm=self.norm, dropout=self.dropout,
                                      down_in_channels=4 * n, left_in_channels=2 * n,
                                      out_channels=2 * n)

        self.ublock3_lower = UpBlockNd(dim=dim, norm=self.norm, dropout=self.dropout,
                                       down_in_channels=4 * n, left_in_channels=2 * n,
                                       out_channels=2 * n)

        self.ublock3_upper = UpBlockNd(dim=dim, norm=self.norm, dropout=self.dropout,
                                       down_in_channels=4 * n, left_in_channels=2 * n,
                                       out_channels=2 * n)

        self.head_lower = get_convolution_operator(dim=dim, in_channels=2 * n, out_channels=out_channels,
                                                   kernel_size=1)
        self.head_mean = get_convolution_operator(dim=dim, in_channels=2 * n, out_channels=out_channels,
                                                  kernel_size=1)
        self.head_upper = get_convolution_operator(dim=dim, in_channels=2 * n, out_channels=out_channels,
                                                   kernel_size=1)

    def forward(self, input: torch.Tensor):
        x, skip1 = self.dblock1(input)
        x, skip2 = self.dblock2(x)
        x, skip3 = self.dblock3(x)
        x_bottom = self.bottom_block(x)

        # mean
        g, _ = self.attblock1(skip3, x_bottom)
        xm = self.ublock1(x_bottom, g)

        g, _ = self.attblock2(skip2, xm)
        xm = self.ublock2(xm, g)

        g, _ = self.attblock3(skip1, xm)

        # 3 heads
        x1 = self.ublock3_mean(xm, g)
        mean = self.head_mean(x1)

        x2 = self.ublock3_lower(xm, g)
        lower = self.head_lower(x2)

        x3 = self.ublock3_upper(xm, g)
        upper = self.head_upper(x3)

        return {'logits': mean,
                'upper': upper,
                'lower': lower}

    def prediction(self,
                   x: torch.Tensor,
                   **kwargs) -> np.ndarray:
        pred_dict = self(x)
        avg_prob = torch.sigmoid(pred_dict['logits'])
        upper_prob = torch.sigmoid(pred_dict['upper'])
        lower_prob = torch.sigmoid(pred_dict['lower'])

        foreground_classes = range(1, self.out_channels)
        out_dict = {'logits': pred_dict['logits']}
        for n in foreground_classes:
            mean_prob_n = (avg_prob[:, n] >= 0.5).long().unsqueeze(1)  # mean mask
            upper_prob_n = (upper_prob[:, n] >= 0.5).long().unsqueeze(1)  # upper bound mask
            lower_prob_n = (lower_prob[:, n] >= 0.5).long().unsqueeze(1)  # lower bound mask

            bounds = mean_prob_n + upper_prob_n + lower_prob_n  # superpose masks for visualization
            out_dict[f'uncertainty_bounds_{n}'] = bounds

        return out_dict
