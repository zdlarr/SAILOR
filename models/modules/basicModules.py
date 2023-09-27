"""
    The modules used in our GeoNet & RenderNet.
    e.g., ResNet resblock, cbam block, se-resblock.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks import get_norm_layer

# the basic convolution module.
class Conv(nn.Module):

    def __init__(self, num_in_chn, num_out_chn, kernel_size, stride, is_3d_conv=False, dilation=1,
                 activate='relu', normalize='group'):
        super(Conv, self).__init__()
        self.use_norm = normalize is not None
        self.use_act  = activate is not None
        pad_size = int(np.floor(dilation * (kernel_size - 1) / 2))

        if not is_3d_conv:
            self.conv_base = nn.Conv2d(num_in_chn, num_out_chn, kernel_size, stride, pad_size, dilation)
            if self.use_norm:
                self.norm = get_norm_layer(norm_type=normalize)(num_out_chn // 4, num_out_chn)
        else:
            self.conv_base = nn.Conv3d(num_in_chn, num_out_chn, kernel_size, stride, pad_size, dilation)
            if self.use_norm:
                if normalize == 'batch':
                    self.norm = nn.BatchNorm3d(num_out_chn)
                else:
                    self.norm = get_norm_layer(norm_type=normalize)(num_out_chn // 4, num_out_chn)
                    
        if self.use_act:
            if activate == 'relu':
                self.act = nn.ReLU()
            elif activate == 'lrelu':
                self.act = nn.LeakyReLU(0.02)
            elif activate == 'prelu':
                self.act = nn.PReLU()
            else:
                raise Exception('Not valid activation operation.')

    def forward(self, x):
        x = self.conv_base(x)
        if self.use_norm:
            x = self.norm(x)
        if self.use_act:
            x = self.act(x)
        return x

class Conv2DLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, k_size, stride, padding=None, dilation=1, norm=None, act=None, bias=False):
        super(Conv2DLayer, self).__init__()
        # use default padding value or (kernel size // 2) * dilation value
        if padding is not None:
            padding = padding
        else:
            padding = dilation * (k_size - 1) // 2

        self.add_module('conv2d', nn.Conv2d(in_channels, out_channels, k_size, stride, padding, dilation=dilation, bias=bias))
        if norm is not None:
            self.add_module('norm', norm)
        if act is not None:
            self.add_module('act', act)

class SElayer(nn.Module):
    # The SE_layer(Channel Attention.) implement, reference to:
    # Squeeze-and-Excitation Networks
    def __init__(self, channel, reduction=16):
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.LeakyReLU(), 
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)

        return x * y


class ResidualBlock(nn.Module):
    # The ResBlock implements: the conv & skip connections here.
    # Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf.
    # Which contains SE-layer implements.

    def __init__(self, channel, norm=nn.BatchNorm2d, dilation=1, bias=False, se_reduction=None, res_scale=1, act=nn.ReLU(True)):
        super(ResidualBlock, self).__init__()

        self.conv1 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=act, bias=bias)
        self.conv2 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=None, bias=None)
        self.se_layer = None
        self.res_scale = res_scale
        if se_reduction is not None:
            self.se_layer = SElayer(channel, se_reduction)
    
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.se_layer:
            x = self.se_layer(x)
        x = x * self.res_scale
        out = x + res
        return out

class ChannelAttention(nn.Module):
    # The channel attention block
    # Original relize of CBAM module.
    # Sigma(MLP(F_max^c) + MLP(F_avg^c)) -> output channel attention feature.
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
    
        self.fc_1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.relu = nn.LeakyReLU()
        self.fc_2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_output = self.fc_2(self.relu(self.fc_1(self.avg_pool(x))))
        max_output = self.fc_2(self.relu(self.fc_1(self.max_pool(x))))
        out = avg_output + max_output
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    # The spatial attention block.
    # Simgoid(conv([F_max^s; F_avg^s])) -> output spatial attention feature.
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in [3, 7], 'kernel size must be 3 or 7.'
        padding_size = 1 if kernel_size == 3 else 3

        self.conv = nn.Conv2d(2, 1, padding=padding_size, bias=False, kernel_size=kernel_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        pool_out = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(pool_out)
        return self.sigmoid(x)


class CBAMlayer(nn.Module):
    # THe CBAM module(Channel & Spatial Attention feature) implement
    # reference from paper: CBAM(Convolutional Block Attention Module)
    def __init__(self, channel, reduction=16):
        super(CBAMlayer, self).__init__()
        self.channel_layer = ChannelAttention(channel, reduction)
        self.spatial_layer = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_layer(x) * x
        x = self.spatial_layer(x) * x
        return x 


class ResidualCbamBlock(nn.Module):
    # The ResBlock which contain CBAM attention module.

    def __init__(self, channel, norm=nn.BatchNorm2d, dilation=1, bias=False, cbam_reduction=None, act=nn.ReLU(True)):
        super(ResidualCbamBlock, self).__init__()

        self.conv1 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=act, bias=bias)
        self.conv2 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=None, bias=None)
        self.cbam_layer = None
        if cbam_reduction is not None:
            self.cbam_layer = CBAMlayer(channel, cbam_reduction)
        
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.cbam_layer:
            x = self.cbam_layer(x)
        
        out = x + res
        return out

class CostVolume(nn.Module):
    """
        Reconstruct cost volume based on different similarity measures.
    """

    def __init__(self, max_disp, feature_similarity='correlation'):
        super(CostVolume, self).__init__()
        
        self.max_disp = max_disp
        self.feature_similarity = feature_similarity
    
    def forward(self, left_feature, right_feature):
        b, c, h, w = left_feature.shape

        if self.feature_similarity == 'difference':
            cost_volume = left_feature.new_zeros(b, c, self.max_disp, h, w)

            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, :, i, :, i:] = left_feature[..., i:] - right_feature[..., :-i]
                else:
                    cost_volume[:, :, i, :, :] = left_feature - right_feature

        elif self.feature_similarity == 'concat':
            cost_volume = left_feature.new_zeros(b, c * 2, self.max_disp, h, w)

            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, :, i, :, i:] = torch.cat([left_feature[..., i:], right_feature[..., :-i]], dim=1)
                else:
                    cost_volume[:, :, i, :, :] = torch.cat([left_feature, right_feature], dim=1)
        
        elif self.feature_similarity == 'correlation':
            # inner production.
            cost_volume = left_feature.new_zeros(b, c, self.max_disp, h, w)

            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, :, i, :, i:] = (left_feature[..., i:] * right_feature[..., :-i]).mean(dim=1)
                else:
                    cost_volume[:, :, i, :, :] = (left_feature * right_feature).mean(dim=1)
        
        else:
            raise NotImplementedError

        cost_volume = cost_volume.contiguous()

        return cost_volume


# discriminator modules.
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""
    
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)	   -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer	  -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)	   -- the number of filters in the last conv layer
            norm_layer	  -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d
            
        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)
        ]
        
        self.net = nn.Sequential(*self.net)
        
    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss