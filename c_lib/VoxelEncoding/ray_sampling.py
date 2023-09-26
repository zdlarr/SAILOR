
import numpy as np
import sys, os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

from c_lib.VoxelEncoding.dist import VoxelEncoding # Voxel encoding library.
from torch.autograd import Function, gradcheck, Variable, grad
import matplotlib.pyplot as plt
# Sampling rays in the masks regions.

class _ray_sampling(Function):
    
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, ks, rs, ts, load_size=512, num_rays_perv=1024, masks=None, in_patch=False, sampled_in_bbox=False, body_bbox=None):
        # ks : [B, N_v, 3, 3], rs : [B, N_v, 3, 3], ts : [B, N_v, 3, 1]
        # masks : [B*N_v, 1, h, w]
        gpu_id = int(str(ks.device).split(':')[-1]) # 0 or 1
        batch_size = ks.shape[0]
        num_views  = ks.shape[1]
        
        # get all the rays ori_dir. [B, N_v, H, W, 6]
        ray_ori_dir = VoxelEncoding.rays_calculating(ks, rs, ts, load_size, gpu_id) # ~0.5ms
        sampled_xy  = None # the sampled xy idx : [B, N_v, N_rays, 2]
        
        if masks is None or sampled_in_bbox: # sample rays randomly per pixels.
            # get the rays intersected with the bboxs.
            ray_ori_dir_ = ray_ori_dir.view(-1, load_size * load_size, 6) # [B*N_v, h, w, 6]
            sampled_xy = []
            
            for i in range(batch_size * num_views):
                bid = i // num_views
                rays_o = ray_ori_dir_[i,:,:3]; rays_d = ray_ori_dir_[i,:,3:] # [B*N_v, h*w, 3]
                # judge if intersect with bbox.
                tmin = (body_bbox[bid,:,:1].T - rays_o) / rays_d
                tmax = (body_bbox[bid,:,1:].T - rays_o) / rays_d
                
                t1 = torch.minimum(tmin, tmax)
                t2 = torch.maximum(tmin, tmax)
                
                near = torch.max(t1, dim=-1)[0]
                far  = torch.min(t2, dim=-1)[0]
                masks_at_box = torch.gt(far, near).int().view(load_size, load_size) # mark whether the rays loads in body masks.
                # get the rays.
                # plt.imshow(masks_at_box.view(512, 512).cpu().detach().numpy())
                # plt.show()
                # plt.savefig('./rays.png')
                # exit()
                h_idx, w_idx = torch.where(masks_at_box != 0)
                if h_idx.size()[0] == 0:
                    h_idx, w_idx = torch.where(masks_at_box == 0)
                    
                rand_idx = torch.randint(0, h_idx.size()[0], size=[num_rays_perv]) # [num_rays]
                sampled_x = w_idx[rand_idx] # get the sampled idx.
                sampled_y = h_idx[rand_idx] # get the sampled idx.
                sampled_xy.append( torch.stack([sampled_x, sampled_y], dim=-1) ) # [N_rays, 2]
            
            sampled_xy = torch.stack(sampled_xy, dim=0)
            sampled_xy = sampled_xy.view(batch_size, num_views, num_rays_perv, 2)

            # rand_idx_x = torch.randint(0, load_size, size=[batch_size, num_views, num_rays_perv], 
            #                            device=ks.device) # [0, 512]
            # rand_idx_y = torch.randint(0, load_size, size=[batch_size, num_views, num_rays_perv], 
            #                            device=ks.device) # [0, 512]
            
            # sampled_xy = torch.stack([rand_idx_x, rand_idx_y], dim=-1) # [B, N_v, N_rays, 2]

        else: # sample rays randomly inside per mask.
            masks = masks.view(-1, load_size, load_size) # to shape: [B*N_v, h,w]
            sampled_xy = []
            
            # cannot perform sampling in parallel. ~ 2ms for b=3,n_v=3
            for i in range(batch_size * num_views): # per N_v * B.
                h_idx, w_idx = torch.where(masks[i] != 0) # get the non-empty pixels.
                # when meeting the non masks here, merely not happened.
                if h_idx.size()[0] == 0:
                    h_idx, w_idx = torch.where(masks[i] == 0) # resampling rays.
                
                if not in_patch: # not sampling in patchies.
                    # get random idx.
                    rand_idx = torch.randint(0, h_idx.size()[0], size=[num_rays_perv]) # [num_rays]
                    sampled_x = w_idx[rand_idx] # get the sampled idx.
                    sampled_y = h_idx[rand_idx] # get the sampled idx.
                    sampled_xy.append( torch.stack([sampled_x, sampled_y], dim=-1) ) # [N_rays, 2]
                else: # sampling in patches, not in parallel.
                    # get a center point randomly
                    patch_size = int( np.sqrt(num_rays_perv) ) // 2 # e.g. 16;
                    rand_idx = torch.randint(0, h_idx.size()[0], size=[1]) # [1 idx.]
                    # range in [ps, load_size - ps]
                    sampled_x = torch.clamp( w_idx[rand_idx], patch_size, load_size - patch_size )[0]
                    sampled_y = torch.clamp( h_idx[rand_idx], patch_size, load_size - patch_size )[0]
                    # notice that the torch meshgrid !!! return y and x (different from numpy)
                    ys, xs = torch.meshgrid( torch.arange(sampled_y-patch_size, sampled_y+patch_size, device=ks.device), 
                                             torch.arange(sampled_x-patch_size, sampled_x+patch_size, device=ks.device))
                    xs = xs.reshape(-1); ys = ys.reshape(-1); # [P**2]
                    sampled_xy.append( torch.stack([xs, ys], dim=-1) ) # [N_rays, 2]
            
            sampled_xy = torch.stack(sampled_xy, dim=0)
            sampled_xy = sampled_xy.view(batch_size, num_views, num_rays_perv, 2) # [B, N_v, N_rays, 2]
            
        sampled_rays = VoxelEncoding.rays_selecting(ray_ori_dir, sampled_xy.int(), gpu_id) # [B, N_v, N_rays, 6]
        
        return sampled_rays, sampled_xy # default type (long, for sampled_xy)
        
        
    @staticmethod
    @custom_bwd
    def backward(ctx, grad0, grad1):
        return None, None, None, None, None, None, None
        

ray_sampling = _ray_sampling.apply



class _rgbdv_sampling(Function):
    
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rgbs, normals, depths, sampled_xy, in_patch=False):
        # rgbs : [B, N_v, 3 or 12, h, w], depths: [B, N_v, 1, h,w]
        # sampled_xy [B, N_v, N_rays, 2]
        gpu_id = int(str(rgbs.device).split(':')[-1]) # 0 or 1
        rgb_channels = rgbs.shape[-3]
        
        rgbs   = rgbs.permute(0,1,3,4,2) # [B, N_v, h, w, 3 or 12]
        normals = normals.permute(0,1,3,4,2) # [[B, N_v, h, w, 3]
        depths = depths.permute(0,1,3,4,2) # [B, N_v, h, w, 1]

        rgbs_, normals_, depths_, labels = VoxelEncoding.rgbdv_selecting(rgbs, normals, depths, sampled_xy.int(), gpu_id);
        
        normals_ = normals_.view(normals_.shape[0], -1, 3) # [B, N_rays, 3]
        depths_ = depths_.view(depths_.shape[0], -1, 1) # [B, N_rays, 1]
        labels  = labels.view(labels.shape[0], -1, 1) # [B, N_rays, 1]
        
        # patched in batches.
        if in_patch: # rendering in patches.
            num_sampled_rays = sampled_xy.shape[-2] # e.g., 1024
            patch_size = int( np.sqrt(num_sampled_rays) )
            rgbs_ori   = rgbs_.view(rgbs_.shape[0], -1, rgb_channels) # [B, N_rays, 3 or 12]
            
            if rgb_channels == 12:
                rgbs_ = rgbs_.view(-1, num_sampled_rays, rgb_channels).permute(0,2,1) # [b*n_v,12,n_rays]
                rgbs_ = F.fold(rgbs_, (patch_size*2, patch_size*2), (2,2), stride=2) # [b*n_v,3,h,w]
            else:
                rgbs_   = rgbs_.view(-1, patch_size, patch_size, rgb_channels).permute(0,3,1,2) # [B*N_v, 3, h, w]
            # depths_ = depths_.view(-1, patch_size, patch_size, 1).permute(0,3,1,2) # [B*N_v, 1, h, w]
            # labels  = labels.view(-1, patch_size, patch_size, 1).permute(0,3,1,2) # [B*N_v, 1, h, w]
        else:
            rgbs_   = rgbs_.view(rgbs_.shape[0], -1, rgb_channels) # [B, N_rays, 3 or 12]
            rgbs_ori = rgbs_.clone()
        
        return rgbs_ori, rgbs_, normals_, depths_, labels
        
    @staticmethod
    @custom_bwd
    def backward(ctx, grad0, grad1, grad2):
        return None, None, None, None

rgbdv_sampling = _rgbdv_sampling.apply