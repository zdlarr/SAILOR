"""
    The Rend Architecture.
"""

from random import random

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
# from torch.cuda.amp import autocast as autocast
import matplotlib.pyplot as plt
from depth_denoising.net import BodyDRM2

from models.basicNet import BasicNet
from models.utils import reshape_sample_tensor # for input tensor.
from models.utils import Depth2Normal
from models.modules.vgg import VGGLoss, VGGPerceptualLoss

# The model for rendering.
from utils_render.utils_render import batchify_rays
from c_lib.VoxelEncoding.ray_sampling import rgbdv_sampling
from c_lib.VoxelEncoding.udf_calculate import udf_calculating, udf_calculating_v2
from c_lib.VoxelEncoding.depth_normalization import DepthNormalizer
from upsampling.BasicRenNet import BasicRenNet
from utils_render.utils_render import normalize_rgbd, unnormalize_depth
from models.modules.ssim import SSIM # SSIM loss here.

import time

class SRONetUp(BasicNet):

    def __init__(self, opts, device):
        BasicNet.__init__(self, opts, device)
        self.batch_size = opts.batch_size
        self.num_views  = opts.num_views
        self.device     = device
        self.num_sampled_rays = self.opts.num_sampled_rays * opts.target_num_views
        
        self.render_net = BasicRenNet(opts, device).to(device)

        # utils for depth refinement, no need to divide by two here.
        self._depth_normalizer = DepthNormalizer(opts, divided2=False)
        # using the depth refinement module to preprocess the RGBD data.
        self.depth_refine = BodyDRM2(opts, device=device).to(device)
        self.depth2normal = Depth2Normal(opts, device).to(device) # for depth normalization.

        for _, param in self.depth_refine.named_parameters(): # dont update anymore.
            param.requires_grad = False

        if self.is_train:
            self.visual_names = ['rgbs', '_r_depths', 'masks', 'target_rgbs', 'target_depths', 'target_masks', '_depths', '_rgbs', '_r_normals']
        else:
            self.visual_names = ['_depths', '_rgbs']

        self.loss_names = ['color', 'vgg', 'weight'] # TODO: the loss names are to be determinated.

        # properties.
        self.rgb_mean = torch.Tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        self.rgb_std  = torch.Tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

        if self.is_train:
            # criterions; 
            self.criterionMSE      = torch.nn.MSELoss().to(device)
            self.criterionL1       = torch.nn.L1Loss().to(device)
            self.criterionSmoothL1 = torch.nn.SmoothL1Loss().to(device)
            self.criterionBCE      = torch.nn.BCELoss().to(device)
            self.criterionVGG      = VGGPerceptualLoss(resize=False).to(device)
            self.criterionSSIM     = SSIM().to(device)
            
            # partly training (fixed other part, only train upsampling part).
            for name, param in self.render_net.named_parameters():
                if 'blended_mlp' in name or 'filter_2d_high_res' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # optimizer
            if opts.optimizer_type == 'adam':
                self.optimizer_REND = torch.optim.Adam( filter(lambda p: p.requires_grad, self.render_net.parameters()),
                  lr=(opts.lr_render), betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay )
            elif opts.optimizer_type == 'rmsprop':
                self.optimizer_REND = torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.render_net.parameters()),
                    lr=(opts.lr_render), momentum=0, weight_decay=0)
            else:
                self.optimizer_REND = None
                raise 'Not support optimizer'
            
            if opts.lr_gamma != 1.0: # the learning rate scheduler.
                self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=self.optimizer_REND, gamma=opts.lr_gamma
                )
            else:
                self.lr_scheduler = None

            self.optimizers = [self.optimizer_REND]
            self.schedulers = [self.lr_scheduler]
            
            # gradscaler
            self._scaler       = torch.cuda.amp.GradScaler()

    def set_input(self, inputs):
        with torch.no_grad(): # feed the data to the render_architecture.
            # rgbd datas.
            self.rgbs   = inputs['rgbs'].to(self.device) # [B, N_v, 3, h, w]
            self.depths = inputs['depths'].to(self.device) # [B, N_v, 1, h, w]
            self.masks  = inputs['masks'].to(self.device) # [B, N_v, 1, h, w]
            # camera data, input views & target view.
            self.ks         = inputs['ks'].to(self.device) # [B, N_v, 3, 3]
            self.rts        = inputs['rts'].to(self.device) # [B, N_v, 3, 4]
            self.target_ks  = inputs['target_ks'].to(self.device) # [B, N_v, 3, 3]
            self.target_rts = inputs['target_rts'].to(self.device) # [B, N_v, 3, 4]
            # view info
            self.source_view_ids = inputs['source_view_ids'].to(self.device) # [B, N_v]
            
            self.camera_params = [self.ks, self.rts]
            self.target_camera_params = [self.target_ks, self.target_rts]

            # neighbor views idxs
            self.neighbor_views = inputs['neighbor_ids'].to(self.device) # [B, N_v_tar, 2]
            
            if self.is_train:
                # target info (rgb gt, depth gt, mask gt in target view for calculating loss.)
                self.target_rgbs   = inputs['target_rgbs'].to(self.device) # [B, N_v, 3, h, w]
                self.target_depths = inputs['target_depths'].to(self.device) # [B, N_v, 1, h, w]
                self.target_masks  = inputs['target_masks'].to(self.device) # [B, N_v, 1, h, w]

                # target rgbd and mask.
                self.target_rgbs_   = self.target_rgbs.view(-1, *self.target_rgbs.shape[2:])
                self.target_rgbs_   = F.unfold(self.target_rgbs_, (2,2), stride=2).view(-1, 12, self.opts.load_size, self.opts.load_size).contiguous()
                self.target_rgbs_   = self.target_rgbs_.view(self.batch_size, -1, *self.target_rgbs_.shape[1:]) # to [B, N_v, 12, h, w]

                self.target_depths_ = self.target_depths.view(-1, *self.target_depths.shape[2:])
                self.target_depths_ = F.interpolate(self.target_depths_, (self.opts.load_size, self.opts.load_size), mode='nearest')
                self.target_depths_ = self.target_depths_.view(self.batch_size, -1, *self.target_depths_.shape[1:]) # [B, N_v, 1, h, w]
                
                self.target_masks_ = self.target_masks.view(-1, *self.target_masks.shape[2:])
                self.target_masks_ = F.interpolate(self.target_masks_, (self.opts.load_size, self.opts.load_size), mode='nearest')
                self.target_masks_ = self.target_masks_.view(self.batch_size, -1, *self.target_masks_.shape[1:])
                
                self.target_view_ids = inputs['target_view_ids'].to(self.device) # [B, N_v]

    def setup_nets(self):
        # init parameters for the render-network and load weights for the rendernet.
        self._load_depth_refine_model()
        self.render_net = self.setup(self.render_net)[0]
        
    def save_nets(self, epoch):
        # Save the network
        self.save(self.render_net, epoch)
        
    def fix_nerf_weights(self): # without using the nerf weights for training.
        for _, param in self.render_net.named_parameters(): # dont update anymore.
            param.requires_grad = False
            
    def _load_depth_refine_model(self):
        model_path = './checkpoints_rend/SAILOR/latest_model_BodyDRM2.pth'
        self.depth_refine.load_state_dict( torch.load( model_path, map_location=str(self.device) ), strict=False )
        print('finished , loading depth refinement module')

    @torch.no_grad()
    def init(self):
        self._inputs = torch.cat([self.rgbs, self.depths, self.masks], dim=2) # [B, N_v, 6, ...]

    @torch.no_grad()
    def _depth_refine(self):
        prefix_shape = list(self._inputs.shape)

        inputs = self._inputs.view(-1, 5, self.opts.img_size, self.opts.img_size) # [B*Nv, 5, H, W]
        inputs_half_res = F.interpolate(inputs, (self.opts.load_size, self.opts.load_size), mode='nearest')
        rgbs, depths, masks = inputs_half_res[:,:3], inputs_half_res[:,3:4], inputs_half_res[:,4:]

        # perform depth denoising.
        depths_normalized, depth_mid, depth_dist, rgbs_normalized = normalize_rgbd( rgbs, depths, self._depth_normalizer )
        data = torch.cat( [rgbs_normalized, depths_normalized, masks], dim=1 )
        r_depths = self.depth_refine( data )
        self._r_depths = unnormalize_depth(r_depths, depth_mid, depth_dist) * masks
    
        # to original data.
        self._refined_depths = F.interpolate(self._r_depths, (self.opts.img_size, self.opts.img_size), mode='nearest')
        self._r_normals = self.depth2normal(self._r_depths, self.ks.view(-1,3,3)) # [B, 3, H, W]
        # plt.imshow(self._r_normals[0].permute(1,2,0).cpu().detach().numpy())
        # plt.show()

        # [RGB, Depth, Depth_refined, masks]
        inputs = torch.cat( [inputs[:,:3], self._refined_depths, inputs[:,4:]], dim=1 ) # [B, Nv, 5, H, W]
        self._inputs = inputs.view(prefix_shape)

    def forward(self): # only working for training processing.
        self.init() # build input data & random rays for rendering.
        self._depth_refine()
            
        # when batch_rays_idx is None, it's inference phase.
        results_nerf, batch_rays_idx = self.render_net(self._inputs, self.camera_params, self.neighbor_views, 
                                                       target_calibs=self.target_camera_params, target_masks=self.target_masks_)
        self._rgbs, self._depths, self._weight_c = results_nerf

        self._batch_rays_idx = batch_rays_idx;

    @torch.no_grad()
    def forward_build(self):
        self.init()
        self._depth_refine()
        render_net = self.render_net
        if isinstance(self.render_net, torch.nn.parallel.DistributedDataParallel):
            render_net = self.render_net.module
        
        render_net.forward_build(self._inputs, self.camera_params)

    @torch.no_grad() # in case the OOM problem here, no need calculate graph
    def forward_query(self, batch_rays=None):
        render_net = self.render_net
        if isinstance(self.render_net, torch.nn.parallel.DistributedDataParallel):
            render_net = self.render_net.module
            
        batch_rays_idx = None
        if batch_rays is None: # no given rays here.
            # 1. sample rays (from target masks, when training.) : [B, N_rays, 6(ori,dir)], [B, N_rays, 2(x,y)]
            batch_rays, batch_rays_idx = render_net.gen_rays(self.target_camera_params, target_masks=None)
            
        # inference in batches, batchify the rays.
        b_rgbs = []; b_depths = [];

        batch_rays_batchified = batchify_rays(batch_rays, self.opts.num_sampled_rays, self.opts.target_num_views);
        # batchifed running.
        for _, _batch_rays in enumerate(batch_rays_batchified): ### incase OOM problem.
            nerf_results = render_net.forward_query(batch_rays=_batch_rays.detach().contiguous(), target_calibs=self.target_camera_params)
            nerf_results = render_net.forward_upsampling_query(nerf_results, self.neighbor_views.detach(), self.target_camera_params)

            _rgbs, _depths, _ = nerf_results
            render_net.zero_grad(set_to_none=True)
            b_rgbs.append( _rgbs.contiguous().detach() )
            b_depths.append( _depths.contiguous().detach() )
        
        self._rgbs    = torch.cat(b_rgbs, dim=1).view(self.batch_size, self.opts.target_num_views, -1, 4, 3).contiguous();  # [B, 1, H*W, 3];
        self._depths  = torch.cat(b_depths, dim=1).view(self.batch_size, self.opts.target_num_views, -1, 1).contiguous(); # [B, 1, H*W, 1];
        self._batch_rays_idx = batch_rays_idx;
            
    @torch.no_grad() 
    def forward_query_occ(self, sampled_points):
        render_net = self.render_net
        if isinstance(self.render_net, torch.nn.parallel.DistributedDataParallel):
            render_net = self.render_net.module
            
        self._predicted_occ = render_net.forward_query_occ(sampled_points.detach()) # [B, N, 3] points -> [B, 1, N_p]

    def backward(self):
        # self.optimizer_REND.zero_grad(set_to_none=True)
        self.optimizer_REND.zero_grad()
        self.forward()
        self.backward_REND() # calculate the loss, and gradients.
        # gradient clipping, incase gradients boom.
        # nn.utils.clip_grad_norm_( self.render_net.parameters(), max_norm=20, norm_type=2 ) # the max gradients
        # self.optimizer_REND.step()
        self._scaler.step(self.optimizer_REND)
        
        # test all gradients, incase for nan or inf.
        for name, param in self.render_net.named_parameters():
            if param.requires_grad: # when have gradients here.
                grad_now = param.grad # get gradients.
                # check gradients, for nan.
                if grad_now == None:
                    print(name)
                elif grad_now.isnan().any() or grad_now.isinf().any():
                    print(name, grad_now.mean())
        
        self._scaler.update()

    def backward_REND(self):
        # TODO: loss backward.
        self.loss_color = self.loss_vgg = torch.tensor(0., device=self.device);
        # calculate the loss between the ground-truth color, depth and rendered results.
        lam_rgb    = self.opts.lam_rgb if self.opts.lam_rgb >= 0 else 1.0
        lam_vgg    = self.opts.lam_vgg if self.opts.lam_vgg >= 0 else 1.0

        # get the target colors and depth from the batch_rays_idx.
        if self._batch_rays_idx is None: # get all pixels' information.
            target_rgbs     = self.target_rgbs_.permute(0, 1, 3, 4, 2).view(self.batch_size, -1, 3)
            target_depths   = self.target_depths_.permute(0, 1, 3, 4, 2).view(self.batch_size, -1, 1)
            dlabels = None
        else: # get the rgbd from the target rgbds (using idxs.)
            with torch.no_grad(): # original rgbs are rays, target rgbs are in patches.
                _, target_rgbs, _, target_depths, dlabels = rgbdv_sampling(self.target_rgbs_,
                                                                           self.target_rgbs_, 
                                                                           self.target_depths_,
                                                                           self._batch_rays_idx,
                                                                           self.opts.ray_patch_sampling)

        if self.check_inf_nan( target_rgbs, target_depths, dlabels, self._rgbs, self._depths ): # check if the data is valid.
            print('meeting nan data or predictions') # when meeting the nan & inf values, 
        # the loss items for color and depths.
        patch_size = int( np.sqrt(self.opts.num_sampled_rays) ) 
        _rgbs  = self._rgbs.view(-1, patch_size * patch_size, 4, 3).view(-1, patch_size * patch_size, 12).permute(0,2,1)
        _rgbs  = _rgbs[:,[0,3,6,9,1,4,7,10,2,5,8,11]] # [B*Nv, 12, N_p]
        _rgbs  = F.fold(_rgbs, (patch_size * 2, patch_size * 2), (2,2), stride=2)

        self.loss_color  = self.criterionL1( _rgbs, target_rgbs ) * 0.4 + ( 1 - self.criterionSSIM( _rgbs, target_rgbs ) ) * 0.6
        self.loss_vgg    = self.criterionVGG( _rgbs, target_rgbs )
        self.loss_weight = self._weight_c.mean()
        # the loss weights control the color blending results.
        self.loss = lam_rgb * self.loss_color + lam_vgg * self.loss_vgg + self.loss_weight * 0.06
        
        self._scaler.scale(self.loss).backward()
        
    def check_inf_nan(self, target_rgbs, target_depths, dlabels, rgbs, depths): # incase for inf or nan values.
        if torch.isnan(target_rgbs).any() or torch.isinf(target_rgbs).any() or \
           torch.isnan(target_depths).any() or torch.isinf(target_depths).any() or \
           torch.isnan(dlabels).any() or torch.isinf(dlabels).any() or \
           torch.isnan(rgbs).any() or torch.isinf(rgbs).any() or \
           torch.isnan(depths).any() or torch.isinf(depths).any():
           return True
        
        return False

    def get_predict_properties(self):
        # get rgbs, depths, ray_sampled_xy.
        return self._rgbs, self._depths, self._batch_rays_idx
    
    def update_optimizer(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            lr = self.optimizer_REND.state_dict()['param_groups'][0]['lr']
            return lr
            
    def set_phase(self, phase='training'):
        render_net = self.render_net
        if isinstance(self.render_net, torch.nn.parallel.DistributedDataParallel):
            render_net = self.render_net.module
            
        render_net.opts.phase = phase

    def set_train(self):
        self._set_train(self.render_net)

    def set_eval(self):
        self._set_eval(self.render_net)

    def set_val(self, status=True):
        self.is_val = status

    def inference(self):
        BasicNet.inference()
    
    @torch.no_grad()
    def update_parameters(self, target_model, net_name='render'):
        # Update the our parameters with target_model's parameters.
        if net_name == 'render':
            net = self.render_net

        if isinstance(net, torch.nn.DataParallel) \
            or isinstance(net, torch.nn.parallel.DistributedDataParallel):
            net = net.module
        
        if isinstance(target_model, torch.nn.DataParallel) \
            or isinstance(target_model, torch.nn.parallel.DistributedDataParallel):
            target_model = target_model.module
        
        model_dict = target_model.state_dict()
        # update the state dicts.
        net.load_state_dict(model_dict, strict=True)
        return


if __name__ == '__main__':
    from options.RenTrainOptions import RenTrainOptions
    opts = RenTrainOptions().parse()
    sronet = SRONetUp(opts, 'cuda:0')
    print(sronet)