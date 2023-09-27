"""
    The depth refinement module.
    input: the estimated normal map, 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.modules.basicModules import ResidualCbamBlock 
from models.Filters.HGFilters import ConvBlock
from models.data_utils import unnormalize_depth, normalize_depth, normalize_face_depth
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.Filters.SwinIR import PatchEmbed, PatchUnEmbed, PatchMerging, RSTB, UpsampleOneStep
from models.data_utils import unnormalize_depth, normalize_depth, normalize_face_depth, unnormalize_face_depth, normalize_face_depth_
from models.Filters.HRNetFilters import HRNetV2_W18, HRNetV2_W18_small_v2_balance, HRNetV2_W18_small_v2_balance2, HRNetV2_W18_small_v2_balance3
from models.Classifies.TransFusionBlock import MultiHeadSelfAttention
import torchvision.models as tmodels

class ASPPConv(nn.Sequential):
    
    def __init__(self, in_dim, out_dim, ks=3, padding=1, dilation=1):
        modules = [
            nn.Conv2d(in_dim, out_dim, ks, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(out_dim // 4, out_dim),
            nn.LeakyReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    
    def __init__(self, in_dim, out_dim):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.GroupNorm(out_dim // 4, out_dim),
            nn.LeakyReLU()
        )
        
    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        # return F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        # return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        # return F.interpolate(x, size=[int(size[0]), int(size[1])], mode='nearest')
        # nearest upsampling 
        x_resized = x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1) \
                     .expand(x.size(0), x.size(1), x.size(2), size[0] // x.size(2), x.size(3), size[1] // x.size(3)) \
                     .contiguous().view(x.size(0), x.size(1), size[0], size[1])
        
        return x_resized
        # return x.repeat(1, 1, int(size[0]), int(size[1]))


class CbamASPP(nn.Module):
    
    def __init__(self, in_channels, rates=[2,4,8]):
        super(CbamASPP, self).__init__()
        modules = []
        
        assert in_channels % 5 == 0, 'err feature size.'
        modules.append( self._make_aspp_conv(in_channels, in_channels // 5, 1, padding=0, dilation=1) )
        modules.append( self._make_aspp_conv(in_channels, in_channels // 5, padding=rates[0], dilation=rates[0]) )
        modules.append( self._make_aspp_conv(in_channels, in_channels // 5, padding=rates[1], dilation=rates[1]) )
        modules.append( self._make_aspp_conv(in_channels, in_channels // 5, padding=rates[2], dilation=rates[2]) )
        modules.append( self._make_aspp_pooling(in_channels, in_channels // 5) )
        self.convs = nn.ModuleList(modules)
        
        # Channel-wise & spatial-wise attention.
        self.project = ResidualCbamBlock(in_channels, norm=None, cbam_reduction=16, act=nn.LeakyReLU())
    
    def _make_aspp_conv(self, in_dim, out_dim, kernel_size=3, padding=2, dilation=2):
        return ASPPConv(in_dim, out_dim, kernel_size, padding, dilation)
    
    def _make_aspp_pooling(self, in_dim, out_dim):
        return ASPPPooling(in_dim, out_dim)
        
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
            # print(conv(x).shape)
        
        res = torch.cat(res, dim=1) # [B, in_channel, ...]
        return self.project(res)
        

class BodyDRM(nn.Module):
    
    def __init__(self, opts, device):
        super(BodyDRM, self).__init__()
        self.opts = opts
        self.device = device
        self.backbone = HRNetV2_W18_small_v2_balance(opts)
        
        # for normalize.
        self.rgb_mean = torch.Tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        self.rgb_std = torch.Tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        
        self.depth_mid = self.opts.z_size
        self.depth_dis = self.opts.z_bbox_len
        self.z_near = 0.04 # 4 cm
        self.z_far  = 3.5 # 3.5 m here.
        
        # the num of feature blocks.
        ft_dims = [360, 240, 160, 80, 64]
        self.num_branches = len(ft_dims) - 1
        aspp_blocks = [] # four cbam blocks.
        self.upsample_layers = nn.UpsamplingBilinear2d(scale_factor=2)
        deconv_blocks = []
        
        for i in range(self.num_branches):
            aspp_blocks.append( CbamASPP(ft_dims[i]) ) # aspp-blocks.
            deconv_blocks.append( self._make_deconv_layer(ft_dims[i], ft_dims[i+1]) ) # deconvolutional layers.
            
        self.deconv_blocks = nn.ModuleList(deconv_blocks)
        self.aspp_blocks = nn.ModuleList(aspp_blocks)
        
        self.last_deconv_layer = self._make_deconv_layer(ft_dims[-1], ft_dims[-1] // 2) # [64, 32]
        # [64, 256, 256] -> [32, ]
        self.last_convs = nn.Sequential(
            nn.Conv2d(ft_dims[-1], ft_dims[-1], 3, 1, 1),
            nn.GroupNorm(ft_dims[-1] // 4, ft_dims[-1]),
            nn.LeakyReLU(),
            self.last_deconv_layer,
            nn.Conv2d(ft_dims[-1] // 2, ft_dims[-1] // 4, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(ft_dims[-1] // 4, 1, 3, 1, 1)
            # nn.Tanh()
        )
    
    def _make_deconv_layer(self, in_dim, out_dim):
        # the deconv layer is used upsample the features.
        deconv = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.LeakyReLU()
            )
        return deconv
    
    def forward(self, input):
        # Input : RGBD images.
        rgbs, depths, masks = input
        # normalize (optional).
        self.rgb_mean = self.rgb_mean.type_as(rgbs); self.rgb_std = self.rgb_std.type_as(rgbs);
        rgbs = (rgbs - self.rgb_mean) / self.rgb_std
        depths, mid, dis = normalize_face_depth(self.opts, depths, return_prop=True, if_clip=False, dis=self.opts.z_bbox_len)
        
        ft0, ft1, ft2, ft3 = self.backbone( torch.cat([rgbs, depths], 1) )
        
        # for high-level features.
        ft3_aspp = self.aspp_blocks[0](ft3) # [B, 360, 16, 16]
        ft3_up = self.deconv_blocks[0](ft3_aspp) # [B, 240, 32, 32]
        
        ft2 = ft3_up + ft2
        ft2_aspp = self.aspp_blocks[1](ft2) # [B, 240, 32, 32]
        ft2_up = self.deconv_blocks[1](ft2_aspp) # [B, 160, 64, 64]
        
        ft1 = ft2_up + ft1
        ft1_aspp = self.aspp_blocks[2](ft1) # [B, 160, 64, 64]
        ft1_up = self.deconv_blocks[2](ft1_aspp) # [B, 80, 128, 128]
        
        ft0 = ft1_up + ft0
        ft0_aspp = self.aspp_blocks[3](ft0) # [B, 80, 128, 128]
        ft0_up = self.deconv_blocks[3](ft0_aspp) # [B, 64, 256, 256]
        
        output = self.last_convs(ft0_up) # decode to the resolution of 512;
        
        # unnormalize & add the mask. 240 + 160 + 80 + 64
        # Problem: the error appears in mid.
        un_depths = unnormalize_depth(self.opts, output, mid, dis) * masks
        # un_depths[un_depths > self.z_far] = 0.0 # filtering.
        # un_depths[un_depths < self.z_near] = 0.0 # filtering method
        
        # un_depths = (output * self.depth_dis + self.depth_mid) * masks
        # un_depths = output * masks # no un-normalize.
        
        fts = [ft3_up, ft2_up, ft1_up, ft0_up]

        return [un_depths], fts

class BodyDRM3(nn.Module):
    
    def __init__(self, opts, device):
        super(BodyDRM3, self).__init__()
        self.opts = opts
        self.device = device
        self.backbone = HRNetV2_W18_small_v2_balance(opts)
        
        # for normalize.
        self.rgb_mean = torch.Tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        self.rgb_std = torch.Tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        
        self.depth_mid = self.opts.z_size
        self.depth_dis = self.opts.z_bbox_len
        self.z_near = 0.04 # 4 cm
        self.z_far  = 3.5 # 3.5 m here.
        
        # the num of feature blocks.
        ft_dims = [360, 240, 160, 80, 64]
        self.num_branches = len(ft_dims) - 1
        aspp_blocks = [] # four cbam blocks.
        self.upsample_layers = nn.UpsamplingBilinear2d(scale_factor=2)
        deconv_blocks = []
        cross_att_blocks = []
        
        for i in range(self.num_branches):
            aspp_blocks.append( CbamASPP(ft_dims[i]) ) # aspp-blocks.
            deconv_blocks.append( self._make_deconv_layer(ft_dims[i], ft_dims[i+1]) ) # deconvolutional layers.

        for i in range(2):
            cross_att_blocks.append( CrossAttentionBlock(opts, device, ft_dims[i] * 3 // 4, ft_dims[i] // 4) )

        self.deconv_blocks = nn.ModuleList(deconv_blocks)
        self.aspp_blocks = nn.ModuleList(aspp_blocks)
        self.cross_att_blocks = nn.ModuleList(cross_att_blocks)
        
        self.last_deconv_layer = self._make_deconv_layer(ft_dims[-1], ft_dims[-1] // 2) # [64, 32]
        # [64, 256, 256] -> [32, ]
        self.last_convs = nn.Sequential(
            nn.Conv2d(ft_dims[-1], ft_dims[-1], 3, 1, 1),
            nn.GroupNorm(ft_dims[-1] // 4, ft_dims[-1]),
            nn.LeakyReLU(),
            self.last_deconv_layer,
            nn.Conv2d(ft_dims[-1] // 2, ft_dims[-1] // 4, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(ft_dims[-1] // 4, 1, 3, 1, 1)
            # nn.Tanh()
        )
    
    def _make_deconv_layer(self, in_dim, out_dim):
        # the deconv layer is used upsample the features.
        deconv = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.LeakyReLU()
            )
        return deconv
    
    def forward(self, input):
        # Input : RGBD images.
        rgbs, depths, masks = input
        # normalize (optional.)
        self.rgb_mean = self.rgb_mean.type_as(rgbs); self.rgb_std = self.rgb_std.type_as(rgbs);
        rgbs = (rgbs - self.rgb_mean) / self.rgb_std
        depths, mid, dis = normalize_face_depth(self.opts, depths, return_prop=True, if_clip=False, dis=self.opts.z_bbox_len)
        
        ft0, ft1, ft2, ft3 = self.backbone( torch.cat([rgbs, depths], 1) )
        
        # for high-level features.
        c_ft3_att, d_ft3_att = self.cross_att_blocks[0](ft3[:, :270], ft3[:, 270:]) # [B, 360, 16, 16]
        ft3_att = torch.cat([c_ft3_att, d_ft3_att], dim=1)
        ft3_up = self.deconv_blocks[0](ft3_att) # [B, 240, 32, 32]
        
        ft2 = ft3_up + ft2
        c_ft2_att, d_ft2_att = self.cross_att_blocks[1](ft2[:, :180], ft2[:, 180:]) # [B, 240, 32, 32]
        ft2_att = torch.cat([c_ft2_att, d_ft2_att], dim=1)
        ft2_up = self.deconv_blocks[1](ft2_att) # [B, 160, 64, 64]
        
        ft1 = ft2_up + ft1
        ft1_aspp = self.aspp_blocks[2](ft1) # [B, 160, 64, 64]
        ft1_up = self.deconv_blocks[2](ft1_aspp) # [B, 80, 128, 128]
        
        ft0 = ft1_up + ft0
        ft0_aspp = self.aspp_blocks[3](ft0) # [B, 80, 128, 128]
        ft0_up = self.deconv_blocks[3](ft0_aspp) # [B, 64, 256, 256]
        
        output = self.last_convs(ft0_up) # decode to the resolution of 512;
        
        # unnormalize & add the mask. 240 + 160 + 80 + 64
        # Problem: the error appears in mid.
        un_depths = unnormalize_depth(self.opts, output, mid, dis) * masks
        # un_depths[un_depths > self.z_far] = 0.0 # filtering.
        # un_depths[un_depths < self.z_near] = 0.0 # filtering method
        
        # un_depths = (output * self.depth_dis + self.depth_mid) * masks
        # un_depths = output * masks # no un-normalize.
        
        fts = [ft3_up, ft2_up, ft1_up, ft0_up]

        return [un_depths], fts

        

class ResNet34(torch.nn.Module):
    def __init__(self, requires_grad=True, pretrained=True, progress=True):
        super(ResNet34, self).__init__()
        # assign some parameters of ResNeXt network.
        # kwargs = dict()
        # default norm layer: None Norm -> forward(x) -> x.
        # kwargs['norm_layer'] = norm_layer
        self.resnet = tmodels.resnet34(pretrained=True)
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        # self.layer4 = self.resnet34.layer4

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        features = []
        x = self.conv1(x)
        # no normalize here when utilize NoneNorm layer.
        # Total feature sizes: 64, 128; 64, 64; 128, 32; 256, 16;
        x = self.bn1(x)

        x = self.relu(x)
        features.append(x)

        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)

        x = self.layer2(x)
        features.append(x)

        x = self.layer3(x)
        features.append(x)

        # x = self.layer4(x)
        # features.append(x)

        return features


class AttentionBlock(MultiHeadSelfAttention):
    def __init__(self, num_head, num_in_feature, num_val_feature, num_inner_feature, bias=True, 
                 dropout=0.0, activation=nn.ReLU(inplace=True)):
        super().__init__(num_head, num_in_feature, num_val_feature, num_inner_feature, bias, dropout, activation)
    
    def forward(self, feat0, feat1):
        q, k, v = self.query(feat0), self.key(feat0), self.value(feat1) # [B, N_pixels, inner_dim].

        # activate the feature volumem, calculate k,q,v
        if self.activation is not None:
            q, k, v = self.activation(q), self.activation(k), self.activation(v) 
    
        # reshape operation: [B, N_pixels, inner_dim] -> [B * n_head, N_pixels, sub_dim]
        q, k, v = self.reshape_to_batches(q), self.reshape_to_batches(k), self.reshape_to_batches(v) # [B x n_head, N_pixel, sub_dim]
        # attention map: [B x n_head, N_pixels, N_pixels] x [B * n_head, N_pixels, sub_dim] -> [B * n_head, N_pixels, sub_dim];
        # calculate the cross-attention matrix, using the similarity matrix from q & k to weight the value v;
        y, attention_map = self.scale_dot_production(q, k, v) # [B x nhead, N_pixels, sub_dim]; [B x nhead, N_pixels, N_pixels]
        y = self.reshape_from_batches(y) # [B, N_pixel, inner_dim x n_head]
        y = self.out_linear(y) # [B, N_pixels, C]

        if self.activation is not None:
            y = self.activation(y) # [B, N, C] # add attention to each view.
            
        return y, attention_map


class CrossAttentionBlock(nn.Module):

    def __init__(self, opts, device, num_in_feat0, num_in_feat1):
        super().__init__()
        self.opts = opts
        self.device = device
        self.num_heads = opts.att_num_heads // 2
        all_in_dim = num_in_feat0 + num_in_feat1
        self.num_inner_feat0 = num_in_feat0 * (self.num_heads // 2) // self.num_heads
        self.num_inner_feat1 = num_in_feat1 * (self.num_heads // 2) // self.num_heads
        # two group normalizers.
        # self.gn0 = nn.GroupNorm(num_in_feat0 // 4, num_in_feat0)
        # self.gn1 = nn.GroupNorm(num_in_feat1 // 4, num_in_feat1)
        self.project0 = ResidualCbamBlock(num_in_feat0, norm=nn.GroupNorm(num_in_feat0 // 4, num_in_feat0), cbam_reduction=16, act=nn.LeakyReLU())
        self.project1 = ResidualCbamBlock(num_in_feat1, norm=nn.GroupNorm(num_in_feat1 // 4, num_in_feat1), cbam_reduction=16, act=nn.LeakyReLU())
        # maybe we need another project layer.
        self.iter = 1
        
        att_blocks0 = []
        att_blocks2 = []
        for i in range(self.iter):
            # the cross attention blocks (calculate between color & depth features.) att_blocks0 / 1 for depth features and rgb features.
            att_blocks0.append( AttentionBlock(opts.att_num_heads, num_in_feat0, num_in_feat0, self.num_inner_feat0, activation=nn.LeakyReLU()) )
            att_blocks2.append( AttentionBlock(opts.att_num_heads, num_in_feat0, num_in_feat1, self.num_inner_feat1, activation=nn.LeakyReLU()) )
        
        self.att_blocks0_0 = nn.ModuleList(att_blocks0)
        self.att_blocks2_0 = nn.ModuleList(att_blocks2)
        self.act = nn.LeakyReLU()

    def forward(self, feat0, feat1):
        _, c0, h, w = feat0.shape
        _, c1, h, w = feat1.shape
        # b * n_v, c, h, w -> [b * n_v, h*w, c];
        # used to enhance the color map's feature.
        for i in range(self.iter):
            feat0_ = feat0.permute(0,2,3,1).reshape(-1, h * w, c0)
            feat1_ = feat1.permute(0,2,3,1).reshape(-1, h * w, c1)
            # individual attention feature. aftering enhancing the depth feature, To rehancing the depth feature.
            att_output0, _ = self.att_blocks0_0[i](feat0_, feat0_) # enhance RGB feature.
            att_output1, _ = self.att_blocks2_0[i](feat0_, feat1_) # use rgb feature to enhance depth features.

            att_output0_ = att_output0.view(-1, h, w, c0).permute(0, 3, 1, 2) # [b,c,h,w]
            att_output1_ = att_output1.view(-1, h, w, c1).permute(0, 3, 1, 2)
            # just weight by the matrix, no original informations.
            att_output0_ = self.act(att_output0_)
            att_output1_ = self.act(att_output1_) # can't not using groupnorm here.
            feat1 = att_output1_
            feat0 = att_output0_
        
        return self.project0(feat0), self.project1(feat1)
        

class LocalEnhanceModule(nn.Module):

    def __init__(self, opts, device, in_dim_rgb, in_dim_depth, dilate=2):
        super(LocalEnhanceModule, self).__init__()
        # local feature, RGB just use the original feature.
        # depth feature -> constrast feature.
        self.rgb_dim = in_dim_rgb;
        self.depth_dim = in_dim_depth;
        # self.gn_d = nn.GroupNorm(in_dim_depth // 4, in_dim_depth)
        self.conv33_depth = nn.Conv2d(in_dim_depth, in_dim_depth // 2, 1, 1, 0)
        self.conv_depth_contrast = nn.Conv2d(in_dim_depth, in_dim_depth // 2, 3, 1, 1) # the dilate convolution (kernel s = 5);

        self.global_depth0 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Conv2d(in_dim_depth, in_dim_depth // 2, 1, 1, 0))

        # rgb. keep the global features.
        self.non_local_rgb = nn.Conv2d(in_dim_rgb, in_dim_rgb // 2, 3, 1, 1) # conv2d. robust.
        self.local_rgb = nn.Conv2d(in_dim_rgb, in_dim_rgb // 2, 1, 1, 0) # conv2d.
        # self.gn_rgb = nn.GroupNorm(in_dim_rgb // 4, in_dim_rgb)
        self.act = nn.LeakyReLU() # the leaky relu activation.
    
    def forward(self, x):
        # get the contrastive feature of depth.
        x_rgb = x[:, :self.rgb_dim];
        x_depth = x[:, self.rgb_dim:];

        # rgb feature. (local & global), rgb features have been enhanced here.
        rgb_feat = torch.cat([self.non_local_rgb(x_rgb), self.local_rgb(x_rgb)], dim=1)
        # rgb_feat = self.act(self.gn_rgb(rgb_feat))

        # depth feature. (contrast). since we input the raw depth values. we will calculate the contrast feature.
        x_local = self.conv33_depth(x_depth)
        x_contrast = self.conv_depth_contrast(x_depth)
        x_global = self.global_depth0(x_depth).expand_as(x_local)
        depth_feat = torch.cat([(x_local - x_contrast), (x_local - x_global)], dim=1) # the constrast feature.
        # depth_feat = self.act(self.gn_d(depth_feat))
        return torch.cat([rgb_feat, depth_feat], dim=1)


class LocalEnhanceModule2(nn.Module):

    def __init__(self, opts, device, in_dim, dilate=2):
        super(LocalEnhanceModule2, self).__init__()
        self.depth_dim = in_dim
        self.conv33_depth0 = nn.Conv2d(in_dim, in_dim // 4, 1, 1, 0)
        self.conv_depth_contrast0 = nn.Conv2d(in_dim, in_dim // 4, 3, 1, 1) # the dilate convolution (kernel s = 5);

        self.global_depth0 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Conv2d(in_dim, in_dim // 4, 1, 1, 0))

        self.non_local_rgbd = nn.Conv2d(in_dim, in_dim // 4, 3, 1, 1) # conv2d.
        self.local_rgbd = nn.Conv2d(in_dim, in_dim // 4, 1, 1, 0) # conv2d.
    
        self.gn_d0 = nn.GroupNorm(in_dim // 8, in_dim // 2)
        self.gn_d1 = nn.GroupNorm(in_dim // 8, in_dim // 2)
        self.act = nn.LeakyReLU() # the leaky relu activation.

    def forward(self, x):
        
        x_local = self.conv33_depth0(x)
        x_contrast = self.conv_depth_contrast0(x)
        x_global = self.global_depth0(x).expand_as(x_local)
        
        # contrast_depth0 = self.act(self.gn_d0(x_local - x_contrast))
        # contrast_depth1 = self.act(self.gn_d0(x_local - x_global))
        contrast_ft = torch.cat([(x_local - x_contrast), (x_local - x_global)], dim=1)

        rgbd_feat = torch.cat([self.non_local_rgbd(x), self.local_rgbd(x)], dim=1) # in_dim // 2;
        # rgbd_feat = self.act(self.gn_d1(rgbd_feat))

        return torch.cat([rgbd_feat, contrast_ft], dim=1)


class BodyDRM2(BodyDRM):
    # the body DRM progressive version.

    def __init__(self, opts, device):
        super(BodyDRM2, self).__init__(opts, device)
        # self.backbone = None
        self.backbone = HRNetV2_W18_small_v2_balance2(opts, n_input_dim=3) # [64, 100, 160, 240]
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
        cross_att_blocks = []
        self.num_branches = len(depths_ft_dims)

        # for i in range(self.num_branches - 1):
            # depth_convs.append( self.make_depth_convs(depths_ft_dims[i], depths_ft_dims[i+1]) )
    
        for i in range(self.num_branches - 3): # three aspp blocks, for three features.
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
        
        for i in range(1): # 1 cross-attention block
            cross_att_blocks.append( CrossAttentionBlock(opts, device, color_ft_dims[::-1][i], depths_ft_dims[::-1][i]) )

        # self.depth_convs0 = nn.ModuleList(depth_convs)
        self.deconv_blocks = nn.ModuleList(deconv_blocks)
        self.aspp_blocks = nn.ModuleList(aspp_blocks)
        self.cross_att_blocks0 = nn.ModuleList(cross_att_blocks)

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

        body_convs1 = []
        for i in range(1):
            dilation = (i+1) * 2 # the body convs are the preprocess for RGBD features.
            if i < 1: # the previous feature.
                body_convs1.append( self.make_body_convs( color_ft_dims[::-1][i], depths_ft_dims[::-1][i], dilation) )
            elif i < 4: # if i == 4. the location information's depth information, especially for shallow feature.
                body_convs1.append( self.make_body_convs2( color_ft_dims[::-1][i] + depths_ft_dims[::-1][i], dilation) )
            else:
                body_convs1.append( self.make_body_convs2( 64, dilation) )

        self.body_convs = nn.ModuleList(body_convs1)

    def make_body_convs(self, in_dim_rgb, in_dim_depth, dilation=2):
        # conv = nn.Sequential(
        #     nn.Conv2d(in_dim, in_dim * 2, 3, 1, dilation, dilation=2), # more global informations.
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_dim * 2, in_dim, 3, 1, 1),
        #     nn.LeakyReLU()
        # )
        conv = LocalEnhanceModule(self.opts, self.device, in_dim_rgb, in_dim_depth, dilate=dilation)
        return conv

    def make_body_convs2(self, in_dim, dilation=2):
        conv = LocalEnhanceModule2(self.opts, self.device, in_dim, dilate=dilation)
        return conv

    def make_depth_convs(self, in_dim, out_dim):
        conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, 1, 1), # 128
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2, padding=0) # keep the max information.
        )
        return conv

    def make_last_convs(self, in_dim):
        conv = nn.Sequential(
            nn.Conv2d(in_dim, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
            # nn.Tanh()
        )
        return conv

    def forward(self, input):
        rgbs, depths, masks = input
        # normalize (optional.)
        self.rgb_mean = self.rgb_mean.type_as(rgbs); self.rgb_std = self.rgb_std.type_as(rgbs);
        rgbs = (rgbs - self.rgb_mean) / self.rgb_std
        # crop in [-1,1],but the output layer may not localate in [-1,1].
        depths, mid, dis = normalize_face_depth(self.opts, depths, return_prop=True, if_clip=True, dis=None)
        # depths_1, mid, dis = normalize_face_depth(self.opts, depths, return_prop=True, if_clip=True, dis=None)
        
        # encoder. two encoders.
        c_ft0, c_ft1, c_ft2, c_ft3 = self.backbone( rgbs )
        d_ft0, d_ft1, d_ft2, d_ft3 = self.backbone2( depths ) # for the depth encoder, seperately encoding.
        # directly input the original depth maps.
        # d_ft_ = self.depth_convs0[0](depths) # no normalization for the depth maps.
        # d_ft0 = self.depth_convs0[1](d_ft_)
        # d_ft1 = self.depth_convs0[2](d_ft0)
        # d_ft2 = self.depth_convs0[3](d_ft1)
        # d_ft3 = self.depth_convs0[4](d_ft2)
        
        # for high-level features. (CAM)
        c_ft3_att, d_ft3_att = self.cross_att_blocks0[0](c_ft3, d_ft3) # [B, 240 & 120, 16, 16]
        ft3_att = torch.cat([c_ft3_att, d_ft3_att], dim=1) # the seperate attentioned layers.
        # ablation study.
        # ft3_att = torch.cat([c_ft3, d_ft3], dim=1) # the seperate attentioned layers.

        # ft3_contrast = self.body_convs2[0](torch.cat([c_ft3, d_ft3], dim=1)) # contrast features.
        # ft3_aspp = self.aspp_blocks[3](torch.cat([c_ft3, d_ft3], dim=1)) # [B, 240 & 120, 16, 16]
        
        ft3_up = self.deconv_blocks[0](ft3_att) # [B, 240, 32, 32]
        # c_ft3_up, d_ft3_up = ft3_up[:,:160], ft3_up[:, 160:] # seprate decoding features.
        depth0 = self.last_convs[0](ft3_up) # the low_resolution depth maps.
        
        # enhancement block. GAM
        ft_3_up_body = self.body_convs[0](ft3_att) # high-level no-need for contrast feature. # [B, 240 & 120, 16, 16]
        # ft_3_up_body = ft3_att.clone() # ablation study.

        ft2 = ft3_up + torch.cat([c_ft2, d_ft2], dim=1) # [B, 80, 128, 128], no adding from high level features.
        
        # c_ft2_att, d_ft2_att = self.cross_att_blocks[1](c_ft2, d_ft2) # [B, 180 & 60, 32, 32]
        # ft2_att = torch.cat([c_ft2_att, d_ft2_att], dim=1)
        # ft2_up = self.deconv_blocks[1](ft2_att) # [B, 160, 64, 64]
        # c_ft2_up, d_ft2_up = ft2_up[:,:100], ft2_up[:, 100:] # seprate feature.
        # depth1 = self.last_convs[1](ft2_up)

        # ft_2_up_body = self.body_convs0[1](ft2_up)
        
        # shallow features.
        ft2_aspp = self.aspp_blocks[2](ft2) # [B, 180 & 60, 32, 32]
        ft2_up = self.deconv_blocks[1](ft2_aspp) # [B, 160, 64, 64]
        depth1 = self.last_convs[1](ft2_up) # the 
        
        ft_2_up_body = ft2_up.clone()

        # c_ft1 = c_ft2_up + c_ft1
        # d_ft1 = d_ft2_up + d_ft1 # don't using high-level feature for drm.
        ft1 = ft2_up + torch.cat([c_ft1, d_ft1], dim=1) # [B, 80, 128, 128];
        
        # for shallow features, obtaining the high global information.(dilations).
        ft1_aspp = self.aspp_blocks[1](ft1) # [B, 160, 64，64]
        ft1_up = self.deconv_blocks[2](ft1_aspp) # [B, 80, 128, 128]
        depth2 = self.last_convs[2](ft1_up)
        
        ft_1_up_body = ft1_up.clone() # more local features needed here.

        ft0 = ft1_up + torch.cat([c_ft0, d_ft0], dim=1) # [B, 80, 128, 128];
        
        ft0_aspp = self.aspp_blocks[0](ft0) # [B, 80, 128, 128]
        ft0_up = self.deconv_blocks[3](ft0_aspp) # [B, 64, 256, 256]

        ft_0_up_body = ft0_up.clone() # totally using depth features (the local feature; 256 x 256;)
        
        depth3 = self.last_convs[3](ft0_up) # [256, 256]

        ft0_up = self.last_conv_mid(ft0_up) # [B, 64, 512, 512];
        ft0_up = self.deconv_blocks[4](ft0_up) # [B, 32, 512, 512]
        output = self.last_convs[4](ft0_up) # [512,512]

        # unnormalize & add the mask. 240 + 160 + 80 + 64;
        # Problem: the error appears in mid.
        # un_depths0 = unnormalize_depth(self.opts, depth0, mid, dis) * nn.UpsamplingNearest2d(size=(depth0.shape[2], depth0.shape[3]))(masks)
        # un_depths1 = unnormalize_depth(self.opts, depth1, mid, dis) * nn.UpsamplingNearest2d(size=(depth1.shape[2], depth1.shape[3]))(masks)
        # un_depths2 = unnormalize_depth(self.opts, depth2, mid, dis) * nn.UpsamplingNearest2d(size=(depth2.shape[2], depth2.shape[3]))(masks)
        # un_depths3 = unnormalize_depth(self.opts, depth3, mid, dis) * nn.UpsamplingNearest2d(size=(depth3.shape[2], depth3.shape[3]))(masks)
        # un_depths4 = unnormalize_depth(self.opts, output, mid, dis) * masks
        # need to unnormalize depth;
        un_depths0 = unnormalize_depth(self.opts, depth0, mid, dis) * F.interpolate(masks, size=depth0.size()[2:], mode='nearest')
        un_depths1 = unnormalize_depth(self.opts, depth1, mid, dis) * F.interpolate(masks, size=depth1.size()[2:], mode='nearest')
        un_depths2 = unnormalize_depth(self.opts, depth2, mid, dis) * F.interpolate(masks, size=depth2.size()[2:], mode='nearest')
        un_depths3 = unnormalize_depth(self.opts, depth3, mid, dis) * F.interpolate(masks, size=depth3.size()[2:], mode='nearest')
        un_depths4 = unnormalize_depth(self.opts, output, mid, dis) * masks
        # renormalize refined depth.
        # output, _, _ = normalize_face_depth(self.opts, un_depths4, return_prop=True, if_clip=False, dis=self.opts.z_bbox_len) # re
        # ft_0_up_body_ = self.backbone3(output) # encoding refined depth features.
        # un_depths[un_depths > self.z_far] = 0.0 # filtering.
        # un_depths[un_depths < self.z_near] = 0.0 # filtering method.
        # 240 + 160 + 80 + 64
        # un_depths = (output * self.depth_dis + self.depth_mid) * masks. 
        # un_depths = output * masks # no un-normalize.
        
        fts = [ft_3_up_body, ft_2_up_body, ft_1_up_body, ft_0_up_body] # only affect the ft0.
        output_depths = [un_depths0, un_depths1, un_depths2, un_depths3, un_depths4]

        return output_depths, fts


# from options.GeoTrainOptions import GeoTrainOptions
# import matplotlib.pyplot as plt
# from dataset.data_utils import save_samples_truncted_prob
# opts = GeoTrainOptions().parse()
# # train_dataset = GeoTrainDataloader(opts)
# ss = BodyDRM2(opts=opts, device='cuda:0').cuda()
# rgbs = torch.randn([3, 3, 512, 512]).cuda()
# depths = torch.randn([3, 1, 512, 512]).cuda()
# masks = torch.randn([3, 1, 512, 512]).cuda()
# a, b = ss([rgbs, depths, masks])
# print(a[2].shape, b[0].shape, b[1].shape, b[2].shape, b[3].shape)


class FaceSwinIR(nn.Module):
    
    def __init__(self, opts, device, num_in_ch=3, window_size=7, mlp_ratio=4, patch_size=1,
                 num_heads=[6,6,6,6], depths=[6,6,6,6], embed_dim=96, qk_scale=None, qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=1, img_range=1., upsampler='', resi_connection='1conv',
                 **kwargs):
        super(FaceSwinIR, self).__init__()
        self.opts = opts
        self.device = device
        num_in_ch  = num_in_ch
        num_out_ch = num_in_ch # only output depth maps.
        self.img_range = img_range
        
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.rgb_mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.window_size = window_size
        
        ####### 1. shallow feature extraction #########
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        
        ####### 2. deep feature extraction ########
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features =  embed_dim
        self.mlp_ratio = mlp_ratio
        
        # split the image into non-overlapping patches.
        self.patch_embed = PatchEmbed(
            img_size=(opts.face_load_size, opts.face_load_size), patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        ) 
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        self.patch_unembed = PatchUnEmbed(
            img_size=(opts.face_load_size, opts.face_load_size), patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=(opts.face_load_size, opts.face_load_size),
                         patch_size=patch_size,
                         resi_connection=resi_connection

                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        
        # last conv layer.
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        
        # lightweight SR.
        self.upsample = UpsampleOneStep(8, embed_dim, num_out_ch,
                                        (patches_resolution[0], patches_resolution[1]))
    
        self.apply(self._init_weights)        
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        assert mod_pad_h == 0 and mod_pad_w == 0, 'err_padding.,'
        # return x
    
    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)
        
        return x

    def forward(self, x):
        rgb_face = x.clone()
        self.check_image_size(rgb_face)
        
        # normalize rgb images.
        self.rgb_mean = self.rgb_mean.type_as(rgb_face)
        rgb_face = (rgb_face - self.rgb_mean) * self.img_range
        
        # pass the network.
        x = self.conv_first(rgb_face)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.upsample(x)
        
        x = x / self.img_range + self.rgb_mean
        
        return x

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        if self.upsample is not None:
            flops += self.upsample.flops()
            
        return flops


# from options.GeoTrainOptions import GeoTrainOptions
# import time
# opts = GeoTrainOptions().parse()
# model = FaceSwinIR(opts, 'cpu', window_size=8,
#                    img_range=1., depths=[6, 6, 6, 6], embed_dim=72, num_heads=[6, 6, 6, 6],
#                    mlp_ratio=2, resi_connection='1conv').to('cuda:0')

# print(model)
# x = torch.randn((12, 3, 64, 64)).to('cuda:0')
# o = model(x)
# num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(num_trainable_params)
# print(o.shape)


class BasicDepthRefine(nn.Module):
    
    def __init__(self, opts, device, output_dim=1):
        super(BasicDepthRefine, self).__init__()
        self.opts = opts
        self.device = device

        self.conv1_d = nn.Sequential(
            nn.Conv2d(1, 32, 7, 1, 3),
            nn.LeakyReLU()
            )
        # RGB, [512, 512, 3] -> [512, 512, 32]; (conv 7x7,padding=3)
        self.conv1_c = nn.Sequential(
            nn.Conv2d(3, 32, 7, 1, 3),
            nn.LeakyReLU()
            )

        # depth: [512, 512, 32] -> [256, 256, 64]; (conv 3x3, padding=1, stride=2)
        self.conv2_d = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU()
            )
        # RGB, [512, 512, 32] -> [256, 256, 64]; (conv 3x3, padding=1, stride=2)
        self.conv2_c = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU()
            )
        
        # depth [256, 256, 64] -> [256, 256, 64];
        self.conv3_d = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU()
            )

        # depth [256, 256, 64] -> [128, 128, 128]; (conv 3x3, padding=1, stride=2)
        self.conv4_d = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU()
            )
        # depth: [128, 128, 128] -> [128, 128, 128]
        self.conv5_d = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU()
            )
        self.conv6_d = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU()
            )
        # RGB, [256, 256, 64] -> [128, 128, 128]; (conv 3x3, padding=1, stride=2)
        self.conv4_c = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU()
            )
            
        self.diconv1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 2, dilation = 2),
            nn.LeakyReLU()
            )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 4, dilation = 4),
            nn.LeakyReLU()
            )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 8, dilation = 8),
            nn.LeakyReLU()
            )
        self.diconv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 16, dilation = 16),
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
            nn.LeakyReLU()
            )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.LeakyReLU()
            )
        self.conv8 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.LeakyReLU()
            )
        self.output = nn.Sequential(
            nn.Conv2d(16, output_dim, 3, 1, 1),
            )
        
        # two CBAM modules.
        self.cbam_block0 = ResidualCbamBlock(128, norm=None, cbam_reduction=8, act=nn.LeakyReLU())
        self.cbam_block1 = ResidualCbamBlock(128, norm=None, cbam_reduction=8, act=nn.LeakyReLU())
        self.tanh = nn.Tanh().to(device=device)
        
    def forward(self, input):
        # input: normal map, rgb, depth_raw -> depth_refine.
        rgbs, depth_raw, mask = input
        # x = torch.cat([rgbs, depth_raw], dim=1) # [B * N_v, 4 (rgb, depths), H, W]

        # guided fusion (rgb & depth);
        x_c = self.conv1_c(rgbs) # [h, w, 32]
        x_d = self.conv1_d(depth_raw) # [h, w, 32]
        x_d = x_d + x_c # [h, w, 32]
        res1 = x_d

        x_c = self.conv2_c(x_c) # [h / 2, w / 2, 64]
        x_d = self.conv2_d(x_d) # [h / 2, w / 2, 64]
        x_d = self.conv3_d(x_d) # [h / 2, w / 2, 64]
        x_d = x_d + x_c # [h / 2, w / 2, 64]
        res2 = x_d
        
        x_c = self.conv4_c(x_c) # [h / 4, w / 4, 128];
        x_d = self.conv4_d(x_d) # [h / 4, w / 4, 128]
        x_d = self.conv5_d(x_d)
        x_d = self.conv6_d(x_d)
        x_d = x_d + x_c # [h / 4, w / 4, 128];
        res3 = x_d

        # filter depth feature.
        x = self.diconv1(x_d)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        x = self.cbam_block0(x)
        x = self.cbam_block1(x)
        x = x + res3

        x = self.deconv1(x)
        x = x + res2
        x = self.conv7(x)
        x = self.deconv2(x)
        x = x + res1
        x = self.conv8(x)
        x = self.output(x)

        x = self.tanh(x) # [-1, 1]
        # force the unmask region to -1;
        # x = mask.expand_as(x) * x # the mask regions -> 0
        # x = x -(1 - mask) # force, the mask regions -> -1;
        
        # depth refinement, add depth_raw
        return x

class FaceRGBDSuperResolution(nn.Module):
    
    def __init__(self, opts, device):
        super(FaceRGBDSuperResolution, self).__init__()
        self.opts = opts
        self.device = device
        self.inputc = 4
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.inputc, 32, 7, 1, 3),
            nn.LeakyReLU()
        )
        
        self.conv2 = ResidualCbamBlock(32, norm=None, cbam_reduction=4, act=nn.LeakyReLU())
                
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU()
        )
        
        self.conv4 = ResidualCbamBlock(64, norm=None, cbam_reduction=8, act=nn.LeakyReLU())
        self.conv5 = ResidualCbamBlock(64, norm=None, cbam_reduction=8, act=nn.LeakyReLU())
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU()
        )
        
        self.conv7 = ResidualCbamBlock(128, norm=None, cbam_reduction=16, act=nn.LeakyReLU())
        self.conv8 = ResidualCbamBlock(128, norm=None, cbam_reduction=16, act=nn.LeakyReLU())
        # diconvs.
        self.diconv1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 2, dilation = 2),
            nn.LeakyReLU()
        )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 4, dilation = 4),
            nn.LeakyReLU()
        )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 8, dilation = 8),
            nn.LeakyReLU()
        )
        self.diconv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 16, dilation = 16),
            nn.LeakyReLU()
        )
        self.diconv5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 32, dilation = 32),
            nn.LeakyReLU()
        )
        
        self.conv9 = ResidualCbamBlock(128, norm=None, cbam_reduction=16, act=nn.LeakyReLU())
        self.conv10 = ResidualCbamBlock(128, norm=None, cbam_reduction=16, act=nn.LeakyReLU())
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.LeakyReLU()
        )
        
        self.conv11 = ResidualCbamBlock(64, norm=None, cbam_reduction=8, act=nn.LeakyReLU())
        self.conv12 = ResidualCbamBlock(64, norm=None, cbam_reduction=8, act=nn.LeakyReLU())

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.LeakyReLU()
        )
        
        self.conv13 = ResidualCbamBlock(32, norm=None, cbam_reduction=4, act=nn.LeakyReLU())
        self.conv14 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.LeakyReLU()
        )
        
        self.output = nn.Sequential(
            nn.Conv2d(16, self.inputc, 3, 1, 1),
        )
        
        # self.tanh = nn.Tanh().to(device=device)
        
    def forward(self, input):
        rgbs, depth_raw, mask = input
        
        x = torch.cat([rgbs, depth_raw], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        res1 = x
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        res2 = x
        
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        res3 = x
        x = self.diconv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        x = self.diconv5(x)
        x = res3 + x
        
        x = self.conv9(x)
        x = self.conv10(x)
        
        x = self.deconv1(x)
        x = res2 + x
        
        x = self.conv11(x)
        x = self.conv12(x)
        
        x = self.deconv2(x)
        x = res1 + x
        
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.output(x)
        
        return x

class TwoScaleDepthRefineModule(nn.Module):
    
    def __init__(self, opts, device):
        super(TwoScaleDepthRefineModule, self).__init__()
        self.opts = opts
        self.device = device
        self.num_views = opts.num_views
        
        # two-scale DRM.
        self.body_drm = BasicDepthRefine(opts, device).to(self.device)
        self.face_drm = BasicDepthRefine(opts, device).to(self.device)
    
    def crop_face_region(self, ft, face_bboxes, face_masks):
        # ft : B * N, C, H, W
        b, c, h, w = ft.shape
        ft = ft.view(-1, self.num_views, c, h, w)[:, 0] # [B, C, H, W]
        ups = nn.UpsamplingNearest2d(size=(h, w))
        
        fts_face = []
        for i in range(ft.shape[0]):
            bbox = face_bboxes[i]
            assert bbox[6] != 0, 'No face region here.'
            # crop the facial regions.
            h_h = int(bbox[3]*h/self.opts.load_size); h_l = int(bbox[1]*h/self.opts.load_size);  
            w_h = int(bbox[2]*w/self.opts.load_size); w_l = int(bbox[0]*w/self.opts.load_size);
            assert (h_h > h_l) and (w_h > w_l)
            ft_face = ups(ft[i, :, h_l:h_h, w_l:w_h][None, ...]) # [1, c, H, W]
            fts_face.append(ft_face)
        
        fts_face = torch.cat(fts_face, dim=0) * ups(face_masks) # [B, C, H, W], with mask (set the background of face to 0)
        return fts_face.detach()
    
    def forward(self, x, face_bbox=None, face_mid=None, face_dis=None):
        # the inputs.
        rgbs, depths, body_masks, face_masks = x
        # parse RGBDs.
        rgbds_body = torch.cat([rgbs, depths], dim=1) # [B * N, 4, H, W]
        # up-sampling the facial depths.
        rgbds_face = self.crop_face_region(rgbds_body, face_bbox, face_masks) # [B, 4, H, W]
        # renormalize the facial depths.
        rgb_face, d_face = rgbds_face[:, :-1], rgbds_face[:, -1:] # [B, 3, H, W], [B, 1, H, W]
        d_face = unnormalize_depth(self.opts, d_face, face_mid, face_dis) * face_masks # unnormalize_face_depth, masked with f_mask;
        d_face, face_mid, face_dis = normalize_face_depth(self.opts, d_face, if_clip=True, threshold=0.2, return_prop=True) # re-normalize face_depth
        # d_face *= face_masks # mask the back ground regions.
        
        # refine the depth maps.
        d_body_rf = self.body_drm([rgbs, depths, body_masks]) # [B * N, 1, H, W]
        d_face_rf = self.face_drm([rgb_face, d_face, face_masks]) # [B, 1, H, W], the up-sampled facial RGBD images.
        # rgb_face_rf, d_face_rf = rgbd_face_rf[:, :-1, ...], rgbd_face_rf[:, -1:, ...] # [B, 3, H, W], [B, 1, H, W]
        # masks the background of face.
        # rgb_face_rf *= face_masks
        
        return d_body_rf, d_face_rf, face_mid, face_dis
        
        