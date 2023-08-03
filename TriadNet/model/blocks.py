from niolo.nets.segmentation.utils import get_convolution_operator, get_normalization_operator, get_transposed_convolution_operator,\
    get_dropout_operator, get_maxpool_operator
import torch
import torch.nn as nn
from torch.nn import functional as F

"""
Attention Unet building blocks 
"""

def get_convolution_operator(dim: int,
                             in_channels,
                             out_channels,
                             kernel_size,
                             stride=1,
                             padding=0,
                             bias: bool = True):
    if dim == 2:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    elif dim == 3:
        return nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    else:
        raise ValueError('Unrecognized dim:{}'.format(dim))


def get_transposed_convolution_operator(dim: int,
                                        in_channels,
                                        out_channels,
                                        kernel_size,
                                        stride=1,
                                        padding=0,
                                        output_padding=0):
    if dim == 2:
        return nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding,
                                  output_padding=output_padding)
    elif dim == 3:
        return nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding,
                                  output_padding=output_padding)
    else:
        raise ValueError('Unrecognized dim:{}'.format(dim))


def get_maxpool_operator(dim: int, kernel_size, stride=None):
    if dim == 2:
        return nn.MaxPool2d(kernel_size, stride)
    elif dim == 3:
        return nn.MaxPool3d(kernel_size, stride)
    else:
        raise ValueError('Unrecognized dim:{}'.format(dim))


def get_dropout_operator(dim: int, dropout_rate: float = 0.):
    if dim == 2:
        return nn.Dropout2d(p=dropout_rate)
    elif dim == 3:
        return nn.Dropout3d(p=dropout_rate)
    else:
        raise ValueError('Unrecognized dim:{}'.format(dim))


def get_normalization_operator(dim: int,
                               n_channels: int,
                               type: str = 'batch'):
    operator = None
    if dim == 2:
        if type == 'batch':
            operator = nn.BatchNorm2d(n_channels)
        elif type == 'instance':
            operator = nn.InstanceNorm2d(n_channels)
        elif type == 'none':
            operator = nn.Identity()
    elif dim == 3:
        if type == 'batch':
            operator = nn.BatchNorm3d(n_channels)
        elif type == 'instance':
            operator = nn.InstanceNorm3d(n_channels)
        elif type == 'none':
            operator = nn.Identity(())

    if operator is None:
        raise ValueError('Could not find a supported operator for dim:{} and type:{}'.format(dim, type))

    return operator


class DownBlockNd(nn.Module):
    def __init__(self,
                 dim,
                 norm,
                 dropout,
                 in_channels: int,
                 out_channels: int):
        super().__init__()

        self.conv1 = get_convolution_operator(dim=dim, in_channels=in_channels, out_channels=out_channels // 2,
                                              kernel_size=3, stride=1, padding=1)
        self.bn1 = get_normalization_operator(dim, out_channels // 2, type=norm)
        self.conv2 = get_convolution_operator(dim=dim, in_channels=out_channels // 2, out_channels=out_channels,
                                              kernel_size=3, stride=1, padding=1)
        self.bn2 = get_normalization_operator(dim, out_channels, type=norm)
        self.maxpool = get_maxpool_operator(dim, kernel_size=2)
        self.dropout = get_dropout_operator(dim=dim, dropout_rate=dropout)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x_down = self.maxpool(x)
        return x_down, x


class UpBlockNd(nn.Module):
    def __init__(self,
                 dim,
                 norm,
                 dropout,
                 down_in_channels: int,
                 left_in_channels: int,
                 out_channels: int) -> None:
        super().__init__()

        self.up_conv = get_transposed_convolution_operator(dim=dim, in_channels=down_in_channels,
                                                           out_channels=down_in_channels, kernel_size=2, stride=2)
        self.conv1 = get_convolution_operator(dim=dim, in_channels=down_in_channels + left_in_channels,
                                              out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = get_normalization_operator(dim=dim, n_channels=out_channels, type=norm)
        self.conv2 = get_convolution_operator(dim=dim, in_channels=out_channels,
                                              out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = get_normalization_operator(dim=dim, n_channels=out_channels, type=norm)
        self.dropout = get_dropout_operator(dim=dim, dropout_rate=dropout)

    def forward(self, x_from_down, x_from_left):
        x_from_down = torch.relu(self.up_conv(x_from_down))
        x = torch.cat((x_from_down, x_from_left), 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x


class BottomBlockNd(nn.Module):
    def __init__(self,
                 dim,
                 norm,
                 dropout,
                 in_channels,
                 out_channels):
        super().__init__()
        self.conv1 = get_convolution_operator(dim=dim, in_channels=in_channels, out_channels=in_channels,
                                              kernel_size=3, padding=1)
        self.bn1 = get_normalization_operator(dim=dim, n_channels=in_channels, type=norm)
        self.conv2 = get_convolution_operator(dim=dim,
                                              in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=3, padding=1)
        self.bn2 = get_normalization_operator(dim=dim, n_channels=out_channels, type=norm)
        self.dropout = get_dropout_operator(dim=dim, dropout_rate=dropout)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x


class GridAttentionBlockND(nn.Module):
    """
    Generic attention block:

    """
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, norm='batch', dropout=0.,
                 sub_sample_factor=2):
        super(GridAttentionBlockND, self).__init__()

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        else:
            self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        assert dimension in [2, 3], "Supported dimensions are 2 and 3."
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.dropout = dropout

        if inter_channels is None:
            self.inter_channels = in_channels // 2
        else:
            self.inter_channels = inter_channels

        if dimension == 3:
            conv_nd = nn.Conv3d
            self.upsample_mode = 'trilinear'
            if self.dropout > 0:
                self.dropout_layer = nn.Dropout3d(p=self.dropout)
        elif dimension == 2:
            conv_nd = nn.Conv2d
            self.upsample_mode = 'bilinear'
            if self.dropout > 0:
                self.dropout_layer = nn.Dropout2d(p=self.dropout)
        else:
            raise ValueError('Network dimension not supported.')

        # Output transform
        bn = get_normalization_operator(n_channels=self.in_channels, dim=self.dimension, type=norm)
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn)

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.Wx = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                          kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0, bias=False)
        self.Wg = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                          kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0,
                           bias=True)

    def forward(self, x, g):
        """
        Forward pass through the block in Figure 2.
        :param x: (b, c, h, d, w) -> skip connection to filter
        :param g: (b, cg, hg, dg, wg) -> gate signal
        :return:
        """

        input_size = x.size()
        if self.dropout > 0:
            x = self.dropout_layer(x)
            g = self.dropout_layer(g)

        theta_x = self.Wx(x)
        theta_x_size = theta_x.size()

        phi_g = F.interpolate(self.Wg(g), size=theta_x_size[2:], mode=self.upsample_mode, align_corners=False)
        sigma1 = F.relu(theta_x + phi_g, inplace=True)

        sigma2 = torch.sigmoid(self.psi(sigma1))

        # Up-sample the attentions and multiply
        alpha = F.interpolate(sigma2, size=input_size[2:], mode=self.upsample_mode, align_corners=False)
        out = alpha.expand_as(x) * x
        out = self.W(out)

        return out, alpha


class GridAttentionBlock2D(GridAttentionBlockND):
    """ 2D Attention Block """

    def __init__(self, in_channels, gating_channels, inter_channels,
                 sub_sample_factor=(2, 2), dropout=0., norm='batch'):
        super(GridAttentionBlock2D, self).__init__(in_channels,
                                                   gating_channels=gating_channels,
                                                   inter_channels=inter_channels,
                                                   dimension=2,
                                                   sub_sample_factor=sub_sample_factor,
                                                   dropout=dropout,
                                                   norm=norm
                                                   )


class GridAttentionBlock3D(GridAttentionBlockND):
    """ 3D Attention Block """

    def __init__(self, in_channels, gating_channels, inter_channels,
                 sub_sample_factor=(2, 2, 2), dropout=0., norm='batch'):
        super(GridAttentionBlock3D, self).__init__(in_channels,
                                                   gating_channels=gating_channels,
                                                   inter_channels=inter_channels,
                                                   dimension=3,
                                                   sub_sample_factor=sub_sample_factor,
                                                   dropout=dropout,
                                                   norm=norm
                                                   )