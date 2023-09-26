"""
    The depth refinement module.
"""

from time import time
from cv2 import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.data_utils import unnormalize_depth
# from models.Filters.HRNetEncoder import HRNetV2_W18, HRNetV2_W18_small_v2_balance, HRNetV2_W18_small_v2_balance2, HRNetV2_W18_small_v2_balance3
from depth_denoising.HRNetDRM import HRNetV2_W18, HRNetV2_W18_small_v2_balance, HRNetV2_W18_small_v2_balance2, HRNetV2_W18_small_v2_balance3, HRNetV2_W18_small_v2_balance4, HRNetV2_W18_small_v2_balance5
# from models.Classifies.TransFusionBlock import MultiHeadSelfAttention
from models.modules.basicModules import CBAMlayer, Conv, ResidualCbamBlock
from models.modules.TwoScaleDepthRefine import BodyDRM, CrossAttentionBlock, CbamASPP, LocalEnhanceModule, LocalEnhanceModule2
from models.data_utils import unnormalize_depth, normalize_depth, normalize_face_depth
# from models.Filters.HRNetFilters import HRNetV2_W18, HRNetV2_W18_small_v2_balance, HRNetV2_W18_small_v2_balance2, HRNetV2_W18_small_v2_balance3
from models.Encoder import ImageEncoder


class BodyDRM2(nn.Module):
    # the body DRM progressive version.

    def __init__(self, opts, device):
        super(BodyDRM2, self).__init__()
        # self.backbone = None
        self.backbone  = HRNetV2_W18_small_v2_balance2(opts, n_input_dim=3) # [64, 100, 160, 240]
        self.backbone2 = HRNetV2_W18_small_v2_balance3(opts, n_input_dim=1) # [16, 60, 80, 120]
        # self.backbone3 = HRNetV2_W18_small_v2_balance3(opts, n_input_dim=1, last_layer=True) # [16, 60, 80, 120]
        # self.backbone = ResNet34()
        # convs for depth maps.
        # color_ft_dims = [240, 180, 120, 64] # original dims: [360, 240, 160, 80, 64]
        self.last_deconv_layer = None
        color_ft_dims = [64, 100, 160, 240] # 
        depths_ft_dims = [1, 8, 16, 60, 80, 120] # depth convs.
        aspp_blocks = []
        deconv_blocks = []
        # depth_convs = []
        # cross_att_blocks = []
        self.num_branches = len(depths_ft_dims)

        for i in range(2): # three aspp blocks, for three features.
            aspp_blocks.append( CbamASPP(color_ft_dims[i] + depths_ft_dims[i+2]) ) # aspp-blocks, three aspp blocks.

        for i in range(self.num_branches - 1): # five deconv blocks.
            if i < self.num_branches - 3:
                deconv_blocks.append( 
                    self._make_deconv_layer(color_ft_dims[::-1][i] + depths_ft_dims[::-1][i], 
                                            color_ft_dims[::-1][i+1] + depths_ft_dims[::-1][i+1]) 
                ) # deconvolutional layers.
            elif i == self.num_branches - 3:
                deconv_blocks.append(
                    self._make_deconv_layer(color_ft_dims[::-1][i] + depths_ft_dims[::-1][i], 64)
                ) # deconvolutional layers.
            else:
                deconv_blocks.append(
                    self._make_deconv_layer(64, 32)
                ) # deconvolutional layers.

        self.deconv_blocks = nn.ModuleList(deconv_blocks)
        self.aspp_blocks = nn.ModuleList(aspp_blocks)

        last_convs = []
        for i in range(self.num_branches - 1): # five last convs.
            if i < self.num_branches - 3:
                last_convs.append(
                    self.make_last_convs(color_ft_dims[::-1][i+1] + depths_ft_dims[::-1][i+1])
                )
            elif i == self.num_branches - 3:
                last_convs.append(
                    self.make_last_convs(64)
                )
            else:
                last_convs.append(
                    self.make_last_convs(32)
                )

        self.last_convs = nn.ModuleList(last_convs)

        self.last_conv_mid = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU()
        )

    def make_last_convs(self, in_dim):
        conv = nn.Sequential(
            nn.Conv2d(in_dim, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
        )
        return conv

    def _make_deconv_layer(self, in_dim, out_dim):
        # the deconv layer is used upsample the features.
        deconv = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.LeakyReLU()
            )
        return deconv

    def unnormalize_depth(self, depths, mids, dists):
        b,c,h,w = depths.shape
        mids_ = mids.view(-1); 
        dists_ = dists.view(-1);
        depths_un = depths.clone()
        # unormalize-depths, the encoder -> (d - mids) / (dists / 2.0)
        for i in range(b):
            depths_un[i] = depths[i] * dists_[i] + mids_[i] # d_un = d' * (dists / 2.0) + mids.
                        
        return depths_un

    def forward(self, input):
        # pass inputs.
        rgbs, depths, masks = input[:, :3], input[:, 3:4], input[:, 4:]

        # self.rgb_mean = self.rgb_mean.type_as(rgbs); self.rgb_std = self.rgb_std.type_as(rgbs);
        # rgbs = (rgbs - self.rgb_mean) / self.rgb_std

        # depths, mid, dis = normalize_face_depth(self.opts, depths, return_prop=True, if_clip=True, dis=None)
        # depths_1, mid, dis = normalize_face_depth(self.opts, depths, return_prop=True, if_clip=True, dis=None)
        # two encoders.
        c_ft0, c_ft1 = self.backbone( rgbs )
        d_ft0, d_ft1 = self.backbone2( depths ) # for the depth encoder, seperately encoding.
        
        # ft3_up = self.deconv_blocks[0]( torch.cat([c_ft3, d_ft3], dim=1) ) # [B, 240, 32, 32]

        # ft2 = torch.cat([c_ft2, d_ft2], dim=1) # [B, 80, 128, 128], no adding from high level features.
        
        # shallow features.
        # ft2_aspp = self.aspp_blocks[2](ft2) # [B, 180 & 60, 32, 32]
        # ft2_up = self.deconv_blocks[1](ft2_aspp) # [B, 160, 64, 64]

        # c_ft1 = c_ft2_up + c_ft1
        # d_ft1 = d_ft2_up + d_ft1 # don't using high-level feature for drm.
        
        # ft1 = ft2_up + torch.cat([c_ft1, d_ft1], dim=1) # [B, 80, 128, 128];
        ft1      = torch.cat([c_ft1, d_ft1], dim=1) # [B, 80, 128, 128];
        
        # for shallow features, obtaining the high global information.(dilations).
        ft1_aspp = self.aspp_blocks[1]( ft1 ) # [B, 160, 64，64]
        ft1_up   = self.deconv_blocks[2](ft1_aspp) # [B, 80, 128, 128]
        
        # break up 
        ft0      = ft1_up + torch.cat([c_ft0, d_ft0], dim=1) # [B, 80, 128, 128];
        # ft0 = torch.cat([c_ft0, d_ft0], dim=1) # [B, 80, 128, 128];
        
        ft0_aspp = self.aspp_blocks[0](ft0) # [B, 80, 128, 128]
        ft0_up = self.deconv_blocks[3](ft0_aspp) # [B, 64, 256, 256]

        ft0_up = self.last_conv_mid(ft0_up) # [B, 64, 512, 512];
        ft0_up = self.deconv_blocks[4](ft0_up) # [B, 32, 512, 512]
        output = self.last_convs[4](ft0_up) # [512,512]

        return output
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                # nn.init.kaiming_normal_(
                #     m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class BodyDRM3(nn.Module):
    # the body DRM progressive version.

    def __init__(self, opts, device):
        super(BodyDRM3, self).__init__()
        # self.backbone = None
        self.backbone  = HRNetV2_W18_small_v2_balance4(opts, n_input_dim=3) # [64, 100, 160, 240]
        self.backbone2 = HRNetV2_W18_small_v2_balance5(opts, n_input_dim=1) # [16, 60, 80, 120]
        # self.backbone3 = HRNetV2_W18_small_v2_balance3(opts, n_input_dim=1, last_layer=True) # [16, 60, 80, 120]
        # self.backbone = ResNet34()
        # convs for depth maps.
        # color_ft_dims = [240, 180, 120, 64] # original dims: [360, 240, 160, 80, 64]
        self.last_deconv_layer = None
        color_ft_dims = [64, 100, 160, 240] # 
        depths_ft_dims = [1, 8, 16, 60, 80, 120] # depth convs.
        aspp_blocks = []
        deconv_blocks = []
        # depth_convs = []
        # cross_att_blocks = []
        self.num_branches = len(depths_ft_dims)

        for i in range(2): # three aspp blocks, for three features.
            aspp_blocks.append( CbamASPP(color_ft_dims[i] + depths_ft_dims[i+2]) ) # aspp-blocks, three aspp blocks.

        for i in range(self.num_branches - 1): # five deconv blocks.
            if i < self.num_branches - 3:
                deconv_blocks.append( 
                    self._make_deconv_layer(color_ft_dims[::-1][i] + depths_ft_dims[::-1][i], 
                                            color_ft_dims[::-1][i+1] + depths_ft_dims[::-1][i+1]) 
                ) # deconvolutional layers.
            elif i == self.num_branches - 3:
                deconv_blocks.append(
                    self._make_deconv_layer(color_ft_dims[::-1][i] + depths_ft_dims[::-1][i], 64)
                ) # deconvolutional layers.
            else:
                deconv_blocks.append(
                    self._make_deconv_layer(64, 32)
                ) # deconvolutional layers.

        self.deconv_blocks = nn.ModuleList(deconv_blocks)
        self.aspp_blocks = nn.ModuleList(aspp_blocks)

        last_convs = []
        for i in range(self.num_branches - 1): # five last convs.
            if i < self.num_branches - 3:
                last_convs.append(
                    self.make_last_convs(color_ft_dims[::-1][i+1] + depths_ft_dims[::-1][i+1])
                )
            elif i == self.num_branches - 3:
                last_convs.append(
                    self.make_last_convs(64)
                )
            else:
                last_convs.append(
                    self.make_last_convs(32)
                )

        self.last_convs = nn.ModuleList(last_convs)

        self.last_conv_mid = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU()
        )

    def make_last_convs(self, in_dim):
        conv = nn.Sequential(
            nn.Conv2d(in_dim, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
        )
        return conv

    def _make_deconv_layer(self, in_dim, out_dim):
        # the deconv layer is used upsample the features.
        deconv = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.LeakyReLU()
            )
        return deconv

    def unnormalize_depth(self, depths, mids, dists):
        b,c,h,w = depths.shape
        mids_ = mids.view(-1); 
        dists_ = dists.view(-1);
        depths_un = depths.clone()
        # unormalize-depths, the encoder -> (d - mids) / (dists / 2.0)
        for i in range(b):
            depths_un[i] = depths[i] * dists_[i] + mids_[i] # d_un = d' * (dists / 2.0) + mids.
                        
        return depths_un

    def forward(self, input):
        # pass inputs.
        rgbs, depths, masks = input[:, :3], input[:, 3:4], input[:, 4:]

        # self.rgb_mean = self.rgb_mean.type_as(rgbs); self.rgb_std = self.rgb_std.type_as(rgbs);
        # rgbs = (rgbs - self.rgb_mean) / self.rgb_std

        # depths, mid, dis = normalize_face_depth(self.opts, depths, return_prop=True, if_clip=True, dis=None)
        # depths_1, mid, dis = normalize_face_depth(self.opts, depths, return_prop=True, if_clip=True, dis=None)
        # two encoders.
        c_ft0, c_ft1 = self.backbone( rgbs )
        d_ft0, d_ft1 = self.backbone2( depths ) # for the depth encoder, seperately encoding.
        
        # ft3_up = self.deconv_blocks[0]( torch.cat([c_ft3, d_ft3], dim=1) ) # [B, 240, 32, 32]

        # ft2 = torch.cat([c_ft2, d_ft2], dim=1) # [B, 80, 128, 128], no adding from high level features.
        
        # shallow features.
        # ft2_aspp = self.aspp_blocks[2](ft2) # [B, 180 & 60, 32, 32]
        # ft2_up = self.deconv_blocks[1](ft2_aspp) # [B, 160, 64, 64]

        # c_ft1 = c_ft2_up + c_ft1
        # d_ft1 = d_ft2_up + d_ft1 # don't using high-level feature for drm.
        
        # ft1 = ft2_up + torch.cat([c_ft1, d_ft1], dim=1) # [B, 80, 128, 128];
        ft1      = torch.cat([c_ft1, d_ft1], dim=1) # [B, 80, 128, 128];
        
        # for shallow features, obtaining the high global information.(dilations).
        ft1_aspp = self.aspp_blocks[1]( ft1 ) # [B, 160, 64，64]
        ft1_up   = self.deconv_blocks[2](ft1_aspp) # [B, 80, 128, 128]
        
        # break up 
        ft0      = ft1_up + torch.cat([c_ft0, d_ft0], dim=1) # [B, 80, 128, 128];
        # ft0 = torch.cat([c_ft0, d_ft0], dim=1) # [B, 80, 128, 128];
        
        ft0_aspp = self.aspp_blocks[0](ft0) # [B, 80, 128, 128]
        ft0_up = self.deconv_blocks[3](ft0_aspp) # [B, 64, 256, 256]

        ft0_up = self.last_conv_mid(ft0_up) # [B, 64, 512, 512];
        ft0_up = self.deconv_blocks[4](ft0_up) # [B, 32, 512, 512]
        output = self.last_convs[4](ft0_up) # [512,512]

        return output
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                # nn.init.kaiming_normal_(
                #     m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DepthRefineModule(nn.Module):

    def __init__(self, opts, device):
        super(DepthRefineModule, self).__init__()
        self.opts = opts
        self.device = device
        # the modified auto encoder model.
        # Auto-Encoder.
        
        # depth: [512, 512, 1] -> [512, 512, 32]; (conv 7x7, padding=3);
        self.conv1_d = nn.Sequential(
            nn.Conv2d(1, 32, 7, 1, 3),
            # nn.GroupNorm(2, 32),
            nn.LeakyReLU()
            )
        
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(4, 32, 7, 1, 3),
        #     # nn.GroupNorm(2, 32),
        #     nn.LeakyReLU()
        #     )

        # RGB, [512, 512, 3] -> [512, 512, 32]; (conv 7x7,padding=3)
        self.conv1_c = nn.Sequential(
            nn.Conv2d(3, 32, 7, 1, 3),
            # nn.GroupNorm(2, 32),
            nn.LeakyReLU()
            )

        # depth: [512, 512, 32] -> [256, 256, 64]; (conv 3x3, padding=1, stride=2)
        self.conv2_d = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            # nn.GroupNorm(4, 64),
            nn.LeakyReLU()
            )
        # RGB, [512, 512, 32] -> [256, 256, 64]; (conv 3x3, padding=1, stride=2)
        self.conv2_c = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            # nn.GroupNorm(4, 64),
            nn.LeakyReLU()
            )
        
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(32, 64, 3, 2, 1),
        #     # nn.GroupNorm(4, 64),
        #     nn.LeakyReLU()
        #     )

        # depth [256, 256, 64] -> [256, 256, 64];
        self.conv3_d = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            # nn.GroupNorm(4, 64),
            nn.LeakyReLU()
            )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(64, 64, 3, 1, 1),
        #     # nn.GroupNorm(4, 64),
        #     nn.LeakyReLU()
        #     )

        # depth [256, 256, 64] -> [128, 128, 128]; (conv 3x3, padding=1, stride=2)
        self.conv4_d = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            # nn.GroupNorm(8, 128),
            nn.LeakyReLU()
            )
        # depth: [128, 128, 128] -> [128, 128, 128]
        self.conv5_d = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            # nn.GroupNorm(8, 128),
            nn.LeakyReLU()
            )
        self.conv6_d = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            # nn.GroupNorm(8, 128),
            nn.LeakyReLU()
            )
        # RGB, [256, 256, 64] -> [128, 128, 128]; (conv 3x3, padding=1, stride=2)
        self.conv4_c = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            # nn.GroupNorm(8, 128),
            nn.LeakyReLU()
            )
        self.conv5_c = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            # nn.GroupNorm(8, 128),
            nn.LeakyReLU()
            )
        
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(64, 128, 3, 2, 1),
        #     # nn.GroupNorm(8, 128),
        #     nn.LeakyReLU()
        #     )
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(128, 128, 3, 1, 1),
        #     # nn.GroupNorm(8, 128),
        #     nn.LeakyReLU()
        #     )
        # self.conv6 = nn.Sequential(
        #     nn.Conv2d(128, 128, 3, 1, 1),
        #     # nn.GroupNorm(8, 128),
        #     nn.LeakyReLU()
        #     )
        
        
        # the depth refinemt modules.
        self.diconv1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 2, dilation = 2),
            # nn.GroupNorm(8, 128),
            nn.LeakyReLU()
            )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 4, dilation = 4),
            # nn.GroupNorm(8, 128),
            nn.LeakyReLU()
            )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 8, dilation = 8),
            # nn.GroupNorm(8, 128),
            nn.LeakyReLU()
            )
        self.diconv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 16, dilation = 16),
            # nn.GroupNorm(8, 128),
            nn.LeakyReLU()
            )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.LeakyReLU()
            )
        self.conv7 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            # nn.GroupNorm(4, 64),
            nn.LeakyReLU()
            )
        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            # nn.GroupNorm(4, 64),
            nn.LeakyReLU()
            )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.LeakyReLU()
            )
        self.conv9 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            # nn.GroupNorm(2, 16),
            nn.LeakyReLU()
            )
        self.output = nn.Sequential(
            nn.Conv2d(16, 1, 3, 1, 1)
            # nn.Tanh()
            )
        
        # two CBAM modules.
        self.cbam_block0 = ResidualCbamBlock(128, norm=None, cbam_reduction=8)
        self.cbam_block1 = ResidualCbamBlock(128, norm=None, cbam_reduction=8)
        # self.tanh = nn.Tanh().to(device=device)
        self.unnormalize = True
        
        self.rgb_mean = torch.Tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        self.rgb_std = torch.Tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    def unnormalize_depth(self, depths, mids, dists):
        b,c,h,w = depths.shape
        mids_ = mids.view(-1); 
        dists_ = dists.view(-1);
        depths_un = depths.clone()
        # unormalize-depths, the encoder -> (d - mids) / (dists / 2.0)
        for i in range(b):
            depths_un[i] = depths[i] * (dists_[i] / 2.0) + mids_[i] # d_un = d' * (dists / 2.0) + mids.
                        
        return depths_un
    
    def normalize_depth(self, depths):
        # (depth - mids) / dists; 
        return normalize_face_depth(self.opts, depths, return_prop=True, if_clip=True, dis=None)
    
    def forward(self, input):
        rgbs, depths, masks = input[:, :3], input[:, 3:4], input[:, 4:]

        # input: rgbs, depths -> outputs: depth_refine.
        # normalize RGBD images.
        # self.rgb_mean = self.rgb_mean.type_as(rgbs); self.rgb_std = self.rgb_std.type_as(rgbs);
        # rgbs = (rgbs - self.rgb_mean) / self.rgb_std # standard normalization.

        # depths, mid, dis = self.normalize_depth(depths) # basic depth normalization.
        # guided fusion (rgb & depth);
        x_c = self.conv1_c(rgbs) # [h, w, 32]
        x_d = self.conv1_d(depths) # [h, w, 32]
        # the added features.
        x_d = x_d + x_c # [h, w, 32], added the features.
        res1 = x_d # the residual is set for depths.

        x_c = self.conv2_c(x_c) # [h / 2, w / 2, 64]
        x_d = self.conv2_d(x_d) # [h / 2, w / 2, 64]
        x_d = self.conv3_d(x_d) # [h / 2, w / 2, 64]
        x_d = x_d + x_c # [h / 2, w / 2, 64]
        res2 = x_d
        
        x_c = self.conv4_c(x_c) # [h / 4, w / 4, 128]
        x_c = self.conv5_c(x_c) # [h / 4, w / 4, 128]
        x_d = self.conv4_d(x_d) # [h / 4, w / 4, 128]
        x_d = self.conv5_d(x_d) # [h / 4, w / 4, 128]
        x_d = self.conv6_d(x_d) # [h / 4, w / 4, 128]
        x_d = x_d + x_c # [h / 4, w / 4, 128];
        # res3 = x_d
        
        # x = self.conv1( torch.cat([rgbs, depths], dim=1) ) # [h, w, 4] -> [h, w, 32]
        # res1 = x
        
        # x = self.conv2(x) # [h / 2, w / 2, 64]
        # x = self.conv3(x) # [h / 2, w / 2, 64]
        # res2 = x
        
        # x = self.conv4(x) # [h / 4, w / 4, 128]
        # x = self.conv5(x) # [h / 4, w / 4, 128]
        # x = self.conv6(x) # [h / 4, w / 4, 128]
        # res3 = x

        # fuse the rgb & depth feature here.
        x = self.diconv1(x_d)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        x = self.cbam_block0(x) # two CBAM blocks, for spatial & channel attentions.
        x = self.cbam_block1(x)
        # x = x + res3 # no res in cbam block.
        
        # The decoder.
        x = self.deconv1(x)
        x = x + res2
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.deconv2(x)
        x = x + res1
        x = self.conv9(x)
        # depths only > 0, only keep the positive values.
        x = self.output(x) # directly unnormalize
        
        # output the refined depths.
        # if self.unnormalize:
        #     x = self.unnormalize_depth(x, mids, dists) * masks
        # else:
        #     x = x * masks
        
        # depth refinement, add depth_raw
        return x


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                # nn.init.kaiming_normal_(
                #     m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=nn.BatchNorm2d):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FeatureNet(nn.Module):
    def __init__(self, norm_act=nn.BatchNorm2d):
        super(FeatureNet, self).__init__()
        # encoding RGB features.
        self.conv0 = nn.Sequential(
                        ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))
        self.conv1 = nn.Sequential(
                        ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act), # kernel=5, stride=2, step=2;
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))
        self.conv2 = nn.Sequential(
                        ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(32, 32, 1) # to player ? WHAT
        self.lat1 = nn.Conv2d(16, 32, 1)
        self.lat0 = nn.Conv2d(8, 32, 1)

        # self.smooth1 = nn.Conv2d(32, 16, 3, padding=1)
        self.smooth0 = nn.Conv2d(32, 16, 3, padding=1) # smooth net -> output features.

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + y

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        feat2 = self.toplayer(conv2)
        feat1 = self._upsample_add(feat2, self.lat1(conv1))
        feat0 = self._upsample_add(feat1, self.lat0(conv0))
        # feat1 = self.smooth1(feat1)
        # [B, 16, H, W] rgb features.
        feat0 = self.smooth0(feat0) # only keep the last features here, using only smooth layer.
        return feat0


class HRNetUNet(nn.Module):
    
    def __init__(self, opts, device):
        super(HRNetUNet, self).__init__()
        self.opts = opts
        self.device = device
        
        ft_channels_depth = [12, 24, 48, 96]
        ft_channels_rgb   = [8, 16, 24, 32]
    
        self._backbone_rgb = ImageEncoder(opts, device, dim_inputs=3, 
                                output_dim=opts.im_feats_dim, encoding_type='hrnet', encoded_data='rgb')
        self._backbone_depth = ImageEncoder(opts, device, dim_inputs=1, 
                                output_dim=opts.im_feats_dim, encoding_type='hrnet', encoded_data='depth')
        self._geo_decoder = nn.Sequential(
            nn.Conv2d(sum(ft_channels_depth), 64, 1, 1, 0),
            nn.GroupNorm(4, 64),
            nn.ReLU(True),
            nn.Conv2d(64, 16, 1, 1, 0)
        )
        
        self._rgb_decoder = nn.Sequential(
            nn.Conv2d(sum(ft_channels_rgb), 64, 1, 1, 0),
            nn.GroupNorm(4, 64),
            nn.ReLU(True),
            nn.Conv2d(64, 16, 1, 1, 0),
            nn.ReLU(True)
        )
        
    def forward(self, rgbs, depths):
        # encoders.
        ft0_rgb, ft1_rgb, ft2_rgb, ft3_rgb = self._backbone_rgb(rgbs)
        ft0_depth, ft1_depth, ft2_depth, ft3_depth = self._backbone_depth(depths)
        
        y1 = F.interpolate(ft1_depth, scale_factor=2, mode='bilinear', align_corners=True)
        y2 = F.interpolate(ft2_depth, scale_factor=4, mode='bilinear', align_corners=True)
        y3 = F.interpolate(ft3_depth, scale_factor=8, mode='bilinear', align_corners=True)
        ft_geo = self._geo_decoder( torch.cat([ft0_depth, y1, y2, y3], dim=1) )
        
        r1 = F.interpolate(ft1_rgb, scale_factor=2, mode='bilinear', align_corners=True)
        r2 = F.interpolate(ft2_rgb, scale_factor=4, mode='bilinear', align_corners=True)
        r3 = F.interpolate(ft3_rgb, scale_factor=8, mode='bilinear', align_corners=True)
        ft_rgb = self._rgb_decoder( torch.cat([ft0_rgb, r1, r2, r3], dim=1) )
        
        return ft_geo, ft_rgb
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def infer(model, device, data):
    import time, os, torch
    # print(data.shape, data.device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    data = data.to('cuda:0')
    output = model(data)
    # print(output.mean())