"""
    HGFilters to extract imgs' features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks import get_norm_layer

def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=True):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm='batch'):
        super(ConvBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))
        self.conv4 = conv3x3(out_planes, out_planes)
        self.relu = nn.LeakyReLU()

        if norm == 'batch':
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn4 = nn.BatchNorm2d(in_planes)
        elif norm == 'group':
            self.bn1 = nn.GroupNorm(16, in_planes)
            self.bn2 = nn.GroupNorm(16, int(out_planes / 2))
            self.bn3 = nn.GroupNorm(16, int(out_planes / 4))
            self.bn4 = nn.GroupNorm(16, out_planes)
        
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=True),
                nn.GroupNorm(16, out_planes),
                nn.PReLU(),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.relu(self.bn2(self.conv1(x))) # [B, O / 2, H, W]

        out2 = self.relu(self.bn3(self.conv2(out1))) # [B, O / 4, H, W]

        out3 = self.relu(self.bn3(self.conv3(out2))) # [B, O / 4, H, W]

        out3 = torch.cat((out1, out2, out3), 1) # [B, O, H, W]
        out3 = self.relu(self.bn4(self.conv4(out3))) # [B, O, H, W]

        if self.downsample is not None:
            residual = self.downsample(residual) # [B, O, H, W]

        out3 += residual

        return out3

class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features, norm='batch'):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.norm = norm

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        # NOTE: for newer PyTorch (1.3~), it seems that training results are degraded due to implementation diff in F.grid_sample
        # if the pretrained model behaves weirdly, switch with the commented line.
        # NOTE: I also found that "bicubic" works better.
        up2 = F.interpolate(low3, scale_factor=2, mode='bicubic', align_corners=True)
        # up2 = F.interpolate(low3, scale_factor=2, mode='nearest)

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class HGFilter(nn.Module):
    def __init__(self, opt, nc_input=None):
        super(HGFilter, self).__init__()
        self.num_modules = opt.num_stack
        self.opt = opt

        # Base part
        if nc_input is None:
            if opt.type_input == 'rgbd': # [rgb, depth.]
                n_input_channel = 6
            elif opt.type_input == 'rgb': # [RGB, normal_est] data
                n_input_channel = 3
            elif opt.type_input == 'depth_only': # [normap_map]
                n_input_channel = 1
            else:
                raise Exception('Non valid input type.')
        else:
            n_input_channel = nc_input

        self.conv1 = nn.Conv2d(n_input_channel, 64, kernel_size=7, stride=2, padding=3)

        if self.opt.filter_norm == 'batch':
            self.bn1 = nn.BatchNorm2d(64)
        elif self.opt.filter_norm == 'group':
            self.bn1 = nn.GroupNorm(32, 64)

        if self.opt.hg_down == 'conv64':
            self.conv2 = ConvBlock(64, 64, self.opt.filter_norm)
            self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        elif self.opt.hg_down == 'conv128':
            self.conv2 = ConvBlock(64, 128, self.opt.filter_norm)
            self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        elif self.opt.hg_down == 'ave_pool':
            self.conv2 = ConvBlock(64, 128, self.opt.filter_norm)
        else:
            raise NameError('Unknown Fan Filter setting!')

        self.conv3 = ConvBlock(128, 128, self.opt.filter_norm)
        self.conv4 = ConvBlock(128, 256, self.opt.filter_norm)

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, opt.num_hourglass, 256, self.opt.filter_norm))

            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256, self.opt.filter_norm))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            if self.opt.filter_norm == 'batch':
                self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            elif self.opt.filter_norm == 'group':
                self.add_module('bn_end' + str(hg_module), nn.GroupNorm(32, 256))
                
            self.add_module('l' + str(hg_module), nn.Conv2d(256,
                                                            opt.hourglass_dim, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(opt.hourglass_dim,
                                                                 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        tmpx = x
        if self.opt.hg_down == 'ave_pool':
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.opt.hg_down in ['conv64', 'conv128']:
            x = self.conv2(x)
            x = self.down_conv2(x)
        else:
            raise NameError('Unknown Fan Filter setting!')

        normx = x

        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs
