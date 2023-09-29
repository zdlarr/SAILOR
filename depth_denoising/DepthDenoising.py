'''
    Depth denoising Arch.
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

from models.basicNet import BasicNet
from models.utils import reshape_multiview_tensor, reshape_sample_tensor # for input tensor.
from models.utils import Depth2Normal
from models.modules.ssim import SSIM # SSIM loss here.
from models.modules.vgg import VGGLoss, VGGPerceptualLoss

from models.utils import DilateMask, DilateMask2
# The model for rendering.
from depth_denoising.net import BodyDRM2 as BodyDRM
from depth_denoising.net import BodyDRM3 as BodyDRM2
from depth_denoising.net import DepthRefineModule
from c_lib.VoxelEncoding.depth_normalization import DepthNormalizer

import time
import cv2

class DepthDenoising(BasicNet):

    def __init__(self, opts, device):
        BasicNet.__init__(self, opts, device)
        self.batch_size = opts.batch_size
        self.num_views  = opts.num_views
        self.device     = device

        # utils for depth refinement, no need to divide by two here.
        self._depth_normalizer = DepthNormalizer(opts, divided2=False)
        self.depth2normal = Depth2Normal(opts, device).to(device) # for depth normalization.

        # original version
        self.depth_refine = BodyDRM(opts, device=device).to(device)
        # lighter version.
        # self.depth_refine = BodyDRM2(opts, device=device).to(device)
        # unet-based depth-refinement network
        # self.depth_refine = DepthRefineModule(opts, device=device).to(device)

        self.visual_names = ['rgbs', '_r_depths', 'masks', '_r_normals']

        self.loss_names = ['depth', 'normal', '3d'] # TODO: the loss names are to be determinated.

        # properties.
        self.rgb_mean = torch.Tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        self.rgb_std  = torch.Tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

        if self.is_train:
            # criterions; 
            self.criterionMSE      = torch.nn.MSELoss().to(device)
            self.criterionL1       = torch.nn.L1Loss().to(device)
            self.criterionSmoothL1 = torch.nn.SmoothL1Loss().to(device)
            self.criterionBCE      = torch.nn.BCELoss().to(device)
            
            # optimizer
            if opts.optimizer_type == 'adam':
                self.optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, self.depth_refine.parameters()),
                  lr=(opts.lr_render), betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay )
                  
            elif opts.optimizer_type == 'rmsprop':
                self.optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.depth_refine.parameters()),
                    lr=(opts.lr_render), momentum=0, weight_decay=0)
            else:
                self.optimizer = None
                raise 'Not support optimizer'
            
            if opts.lr_gamma != 1.0: # the learning rate scheduler.
                self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=self.optimizer, gamma=opts.lr_gamma
                )
            else:
                self.lr_scheduler = None

            self.optimizers = [self.optimizer]
            self.schedulers = [self.lr_scheduler]

            self._scaler       = torch.cuda.amp.GradScaler()
            self.criterionSSIM = SSIM().to(device) # basic SSIM similarity.

    def set_input(self, inputs):
        with torch.no_grad():
            # rgbd datas.
            self.rgbs   = inputs['rgbs'].to(self.device) # [B, N_v, 3, h, w]
            self.depths = inputs['depths'].to(self.device) # [B, N_v, 1, h, w]
            self.masks  = inputs['masks'].to(self.device) # [B, N_v, 1, h, w]
            # camera data, input views & t
            self.ks         = inputs['ks'].to(self.device) # [B, N_v, 3, 3]
            self.rts        = inputs['rts'].to(self.device) # [B, N_v, 3, 4]
            
            self.camera_params = [self.ks, self.rts]
            
            if self.is_train:                          
                # get the target object's points.
                self.target_points = inputs['target_verts'].to(self.device) # [B, N_max, 3], the N_max default as 6e5
                self.gt_depths = inputs['gt_depths'].to(self.device)
                self.gt_depths = self.gt_depths.view(-1, *self.gt_depths.shape[2:]) # [B*N, 1, H, W]

    def setup_nets(self): # loading depth refinement modules.
        self.depth_refine = self.setup(self.depth_refine)[0]
        print('set up the depth refinement net successfully.')
        
    def save_nets(self, epoch):
        # Save the network.
        self.save(self.depth_refine, epoch)

    @torch.no_grad()
    def init(self):
        self._inputs = torch.cat([self.rgbs, self.depths, self.masks], dim=2) # [B, N_v, 6, ...]

    def forward(self): # only working for training processing.
        self.init() # build input data & random rays for rendering.
        self._depth_refine()

    def unnormalize_depth(self, depths, mids, dists):
        b,c,h,w = depths.shape
        mids_ = mids.view(-1); 
        dists_ = dists.view(-1);
        depths_un = depths.clone()
        # unormalize-depths, the encoder -> (d - mids) / (dists / 2.0)
        for i in range(b):
            depths_un[i] = depths[i] * dists_[i] + mids_[i] # d_un = d' * (dists / 2.0) + mids.
                        
        return depths_un

    def normalize_rgbd(self, rgbs, depths):
        depths_normalized, _, depth_mid, depth_dist = self._depth_normalizer( depths ) # to [-0.5, 0.5]
        rgbs_normalized    = (rgbs - self.rgb_mean.type_as(rgbs)) / self.rgb_std.type_as(rgbs) # standard normalization.
        return depths_normalized, depth_mid, depth_dist, rgbs_normalized
        
    def _depth_refine(self):
        inputs = self._inputs.view(-1, 5, self.opts.img_size, self.opts.img_size) # [B*Nv, 6, H, W]
        inputs_half_res = F.interpolate(inputs, (self.opts.load_size, self.opts.load_size), mode='nearest')
        rgbs, depths, masks = inputs_half_res[:,:3], inputs_half_res[:,3:4], inputs_half_res[:,4:]

        depths_normalized, depth_mid, depth_dist, rgbs_normalized = self.normalize_rgbd( rgbs, depths )
        data = torch.cat( [rgbs_normalized, depths_normalized, masks], dim=1 )
        
        r_depths = self.depth_refine( data ) # depth refinement network.
        
        self._r_depths = self.unnormalize_depth(r_depths, depth_mid, depth_dist) * masks # recover the original depths.
        self._r_normals = self.depth2normal(self._r_depths, self.ks.view(-1,3,3)) # [B, 3, H, W]
    
    def backward(self):
        # self.optimizer_REND.zero_grad(set_to_none=True)
        self.optimizer.zero_grad()
        self.forward()
        self.backward_depth() # calculate the loss, and gradients.
        self._scaler.step(self.optimizer)
        
        self._scaler.update()

    def backward_depth(self):
        # TODO: loss backward.
        self.loss_depth, self.loss_normal, self.loss_3d = self.extend_depth_error(self._r_depths, self.gt_depths)
        # self.loss_3d = torch.tensor(0.0).float().to(self.loss_normal.device)
        # loss balance weights.
        self.loss = self.loss_depth * 4.0 + self.loss_normal * 1.0 + self.loss_3d * 0.02
        self._scaler.scale(self.loss).backward()
        return

    def extend_depth_error(self, preds, labels):
        from pytorch3d.loss import chamfer_distance
        # neighbor_views : [B, N_t, 2], 
        # pred : [B*N_v, 1, h, w]; labels : [B*N_v, 1, h, w]
        nl_gt = self.depth2normal(labels, self.ks.view(-1,3,3))
        nl_pred = self.depth2normal(preds, self.ks.view(-1,3,3))
        # depth loss, two parts. MSE & L1.
        loss_depth = self.criterionMSE(preds, labels) + self.criterionL1(preds, labels)
        # normal loss.
        loss_normal_ssim = ( 1 - self.criterionSSIM(nl_gt, nl_pred) ) * 0.84 # the ssim loss.
        loss_normal_l1   = self.criterionSmoothL1(nl_gt, nl_pred) * 0.16 # smooth l1 loss.
        # nl_pred = F.normalize(nl_pred, dim=1)
        # nl_gt   = F.normalize(nl_gt, dim=1) # normalize the ground-truth vectors
        # nl_l1   = self.criterionL1(nl_pred, nl_gt) # step1, l2 loss here.
        # # [B, N_rays, 1] and 1, cos loss range in [-1, 1];
        # nl_cos = ( 1 - (nl_pred * nl_gt).sum(dim=1) ).mean() # step2, cos loss, l1 | 1 - N^T \cdot N |_1

        loss_3d = 0.0
        # ###################### Loss 3D ########################
        B = self.batch_size
        # fuse the multi-view depth maps.
        s_k = self.ks.view(-1, 3, 3); s_rt = self.rts.view(-1, 3, 4);
        # build the X-Y-Z grids.
        Y, X      = torch.meshgrid( torch.arange(0, self.opts.load_size, device=self.device, dtype=preds.dtype), 
                                    torch.arange(0, self.opts.load_size, device=self.device, dtype=preds.dtype) )
        # [B*N_v, 3, H, W]
        dst_grids = torch.stack( [X, Y, torch.ones_like(X)], dim=0)
        # [B*N_v, 3, H*W]
        dst_grids = torch.stack( [dst_grids] * preds.shape[0] ).view(preds.shape[0], 3, -1)
        # get global pcds.
        XYZ = (torch.inverse(s_k) @ dst_grids) * preds.view(-1,1,self.opts.load_size*self.opts.load_size) # [B,3,H*W] * [B,1,H*W] -> [B,3,H*W]
        XYZ = torch.inverse(s_rt[:,:3,:3]) @ (XYZ - s_rt[:,:3,-1:]) # [B,3,3] * [B,3,H*W] -> [B,3,H*W]
        # build the global pcds.
        XYZ = XYZ.view(B, self.num_views, 3, -1) # [B, Nv, 3, H*W]
        preds_ = preds.view(B, self.num_views, -1) # [B, 4, H, W]

        for bid in range(B):
            nodes = []
            for vid in range(self.num_views):
                xyz = XYZ[bid, vid] # [3, N_p]
                valid = preds_[bid, vid] > 0.01 # [1, N_p]
                xyz_valid = xyz[:, valid] # [3, N_p']
                if xyz_valid.numel() != 0:
                    nodes.append( xyz_valid )
            if nodes != []:
                nodes = torch.cat( nodes, dim=-1 ) # [3, N']
                target_verts = self.target_points[bid] # [N', 3]
                # 3D loss for fused 3d points and gt points.
                loss_3d += chamfer_distance( nodes.permute(1,0)[None], target_verts[None] )[0]
        ##################### Loss 3D ########################

        return loss_depth, loss_normal_ssim + loss_normal_l1, loss_3d
        # return loss_depth, loss_normal_ssim + loss_normal_l1

    def update_optimizer(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            return lr

    def set_train(self):
        self._set_train(self.depth_refine)

    def set_eval(self):
        self._set_eval(self.depth_refine)

    def set_val(self, status=True):
        self.is_val = status

    def inference(self):
        BasicNet.inference()

    @torch.no_grad()
    def update_parameters(self, target_model):
        # Update the our parameters with target_model's parameters.
        net = self.depth_refine

        if isinstance(net, torch.nn.parallel.DistributedDataParallel):
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
    depth_denoiser = DepthDenoising(opts, 'cuda:0')
    print(depth_denoiser)