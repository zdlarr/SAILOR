"""
    The trans-nerf.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.Classifies.ImplicitFunction import SurfaceClassifier
from models.Classifies.SirenFunction import SirenNet
from c_lib.VoxelEncoding.freq_encoding import FreqEncoder
from c_lib.VoxelEncoding.sh_encoding import SHEncoder

from options.RenTrainOptions import RenTrainOptions
from utils_render.utils_render import trunc_exp, LaplaceDensity, LogisticDensity, MyLaplaceDensity, OccDensity, TempGrad, l2_normalize

from SRONet.aggregation import Transformer, Encoder, HydraEncoder
from models.networks import init_weights

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

class BlendMLP(nn.Module):
    
    def __init__(self):
        super(BlendMLP, self).__init__()
        self.l0 = nn.Sequential(
            nn.Linear(20*2, 64), # [40, 64]
            nn.ReLU(inplace=True)
        )
        self.l1 = nn.Sequential(
            nn.Linear(64, 64), # [64, 64]
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Sequential(
            nn.Linear(64, 32), # [64, 32];
            nn.ReLU(inplace=True)
        )
        self.l3 = nn.Sequential(
            nn.Linear(32+30, 64), # [62, 64];
            nn.ReLU(inplace=True)
        )
        self.l4 = nn.Sequential(
            nn.Linear(64, 32), # [62, 32];
            nn.ReLU(inplace=True)
        )
        self.l5 = nn.Sequential(
            nn.Linear(32, 3), # [32, 3];
            nn.Sigmoid()
        )
        
        # initialization each linear layer;
        self.l0.apply(weights_init)
        self.l1.apply(weights_init)
        self.l2.apply(weights_init)
        self.l3.apply(weights_init)
        self.l4.apply(weights_init)
        self.l5.apply(weights_init)
        
    def forward(self, neighbor_feats, ray_rgb_feats):
        # rgbs_feats: 40, ray_rgb_feats: 30;
        # use ray_rgb_feats to guide the neighbor views' pixel features;
        # parsing MLPs: part 1;
        x0 = self.l0( neighbor_feats )
        x1 = self.l1( x0 ) 
        x2 = self.l2( x1 )
        # part 2;
        x3 = self.l3( torch.cat( [x2, ray_rgb_feats], dim=-1 ) )
        x4 = self.l4( x3 )
        output = self.l5( x4 )
        
        return output

class TransNerf(nn.Module):
    
    def __init__(self, opts, device):
        super(TransNerf, self).__init__()
        # basic parameters.
        self._opts     = opts
        self._device   = device
        self._is_train = opts.is_train
        self._phase    = opts.phase
        
        self._batch_size = opts.batch_size
        self._num_views  = opts.num_views
        
        inputs_dim = 16 + 1 + 1 # depth feat + z + t-psdf
        inputs_dim_color = 16 + 8 + 3 + 3 # rgb feat, geo feat, rgbs, view dir.
        
        self.feats_net = SurfaceClassifier( # new inputs: p(x) + z + z" + im_feats, output the geo_feats and blending weights.
            filter_channels = [inputs_dim] + [opts.feat_net_hidden_dim] * (opts.num_layers_feat_net+1) + [opts.point_feats_dim],
            pe=False,
            no_residual=True, # no additional residual blocks in MLP;
            activation=nn.ReLU(inplace=True), # the ReLU activation
            geometric_init=False,
            last_op=nn.ReLU(inplace=True) # using softplus as activation funcs.
        )

        # hydra-attention, the cost completitation (O(NC)), a very light transformer.
        self._hydra_e = HydraEncoder(n_layers=2, n_input=inputs_dim_color, d_inner=32)
        init_weights( self._hydra_e )
        
        # output the occupancy values. [64, 64, 64, 1]
        self.sigma_net = SurfaceClassifier( # INPUT : points_feats, output sigma 1; 
            filter_channels = [opts.point_feats_dim] + [opts.sigma_net_hidden_dim] * (opts.num_layers_sigma_net+1) + [1],
            pe=False,
            no_residual=True,
            activation=nn.ReLU(inplace=True), # the ReLU activation 
            geometric_init=False,
            last_op=nn.Sigmoid()
        )
        
        self.rgb_net = nn.Sequential(
            nn.Linear(inputs_dim_color, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),
            nn.Sigmoid()
        )
        
        self.geo_ft_pca_net = nn.Sequential(
            nn.Linear(inputs_dim, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True)
        )

        if self._opts.adopting_upsampling:
            self._blended_mlp = BlendMLP()

        self.rgb_net.apply(weights_init)
        self.geo_ft_pca_net.apply(weights_init)
    
    def density(self, z, visi, im_feats_geo):
        # z, visi, rgbs and im_feats;
        inputs = torch.cat( [z, visi, im_feats_geo], dim=-1 ) # input dim : 2 + 16
        # parsing the geo_feat_mlp
        geo_feats = self.geo_ft_pca_net(inputs)

        # mean feature for geometries.
        encoded_feats = self.feats_net( inputs )
        encoded_feats_ = torch.mean( encoded_feats.view(self._batch_size, self._num_views, -1, encoded_feats.shape[-1]), 
                                    dim=1, keepdim=False ).view(-1, encoded_feats.shape[-1])

        occ = self.sigma_net(encoded_feats_)
    
        return occ, geo_feats
        
    def forward_density(self, z, visi, im_feats_geo):
        B, N_views, N_rays, N_points = z.shape[:4]
        z = z.view(-1, 1)
        visi = visi.view(-1, 1)
        im_feats_geo = im_feats_geo.view(-1, im_feats_geo.shape[-1])
        
        occ, geo_feat = self.density(z, visi, im_feats_geo)

        occ  = occ.view(B, N_rays, N_points, -1)
        geo_feat = geo_feat.view(B, N_views, N_rays, N_points, -1) # default 16;

        return occ, geo_feat
    
    def forward_color(self, rgb_feats, rgbs, geo_feats, dirs_l):
        B, N_views, N_rays, N_points, _ = dirs_l.shape 
        rgb_feats = rgb_feats.view(-1, rgb_feats.shape[-1])
        geo_feats = geo_feats.view(-1, geo_feats.shape[-1])
        rgbs = rgbs.view(-1, 3)
        dirs_l = dirs_l.view(-1, 3)

        inputs = torch.cat( [rgb_feats, rgbs, geo_feats, dirs_l], dim=-1 )
        
        encoded_feats = inputs.view(self._batch_size, N_views, -1, inputs.shape[-1]) \
                              .permute(0, 2, 1, 3).reshape(-1, N_views, inputs.shape[-1]) # [B*N_rays*N_points, N_view, C]

        encoded_feats = self._hydra_e( encoded_feats )[:,0] # hydra-attention for encoded features.
        
        color = self.rgb_net( encoded_feats )

        color = color.view(B, N_rays, N_points, -1) # [B, N_rays, 3].
        encoded_feats = encoded_feats.view(B, N_rays, N_points, -1) # [B, N_rays, 32].
        
        return color, encoded_feats

    def forward_density_(self, z, visi, im_feats_geo, in_img_flag):
        z = z.view(-1, 1); 
        visi = visi.view(-1, 1);
        im_feats_geo = im_feats_geo.view(-1, im_feats_geo.shape[-1])
        
        occ, _ = self.density( z, visi, im_feats_geo )
        if in_img_flag is not None:
            occ = occ * in_img_flag

        occ = occ.view(self._batch_size, 1, -1) # [B, 1, N_points].
        
        return occ.float()

    def predicted_fusion_weights(self, rgb_feats, dirs, coff, ray_rgb_feats):
        feats0 = torch.cat( [rgb_feats[0], dirs[0], coff[0]], dim=-1 )
        feats1 = torch.cat( [rgb_feats[1], dirs[1], coff[1]], dim=-1 )

        return self._blended_mlp( torch.cat( [feats0, feats1], dim=-1 ), ray_rgb_feats )