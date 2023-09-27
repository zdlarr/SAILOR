
import torch
import torch.nn as nn
import torch.nn.functional as F
from c_lib.VoxelEncoding.dist import VoxelEncoding # Voxel encoding library.
from c_lib.VoxelEncoding.ray_sampling import ray_sampling
from utils_render.util import make_dir
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 
from PIL import Image
from skimage.measure import compare_psnr, compare_ssim

import os
import numpy as np
import open3d as o3d
# from snapshot_smpl.smpl import Smpl
import cv2
import time
import random
from c_lib.VoxelEncoding.depth_normalization import depth_normalizer as _depth_normalizer
from utils_render.mesh_util import save_obj_mesh, reconstruction, create_grids, eval_grid_octree
import mcubes

# 0. metric-utils; PSNR in masks
def cal_psnr(im1, im2, mask, in_masks=True):
    mask[mask < 0.5] = 0; mask[mask >= 0.5] = 1;
    mask = mask[..., None] # [H, W, 1];
    num_none_zeros_pixels = mask.sum()
    # sum(diff**2) / num_pixels, the pixels' difference;
    if in_masks:
        mse = (np.abs(im1*mask - im2*mask) ** 2).sum() / num_none_zeros_pixels
    else:
        mse = (np.abs(im1*mask - im2*mask) ** 2).mean() # only calculate the 
    psnr = 10 * np.log10(255 * 255 / mse)
    return psnr

# 0. basic utils.
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def save_rgb_dsr_results(opts, visuals, local_rank, subject_name, epoch, idx, phase='training'):
    image_dir = os.path.join(opts.model_dir, phase)
    make_dir(image_dir)
    save_path_rgb    = os.path.join(image_dir, 'rgb_{}_rank{}_epoch{}_idx_dsr{}.jpg'.format(subject_name, local_rank, epoch, idx))
    
    if opts.is_train:
        assert opts.batch_size == 1, 'validating batch size must be 1.'
        
        input_rgbs, target_rgbs, blend_high_freq, \
        predicted_rgbs, rgbs_refine, input_normals = \
            visuals['rgbs'], visuals['target_rgbs'], visuals['_blend_high_fq'], \
            visuals['_rgbs'], visuals['_rgbs_refine'], visuals['_r_normals'] ;
        
        input_rgbs = input_rgbs.view(-1, opts.num_views, 3, opts.img_size, opts.img_size)
        target_rgbs = target_rgbs.view(-1, 3, opts.img_size, opts.img_size)
        input_normals = F.interpolate(input_normals.view(-1, 3, opts.load_size, opts.load_size), (opts.img_size,opts.img_size), mode='bilinear')
        input_normals = input_normals.view(-1, opts.num_views, 3, opts.img_size, opts.img_size)
        
        # rgbs_warpped = (rgbs_wp_left + rgbs_wp_right) / 2.0 # [1k, 1k]
        rgbs_warpped   = torch.clamp( blend_high_freq, 0, 1.0 ) # clamp to [0,1]
        predicted_rgbs = predicted_rgbs.view(-1, opts.load_size, opts.load_size, 3).permute(0, 3, 1, 2)
        predicted_rgbs = F.interpolate(predicted_rgbs, (opts.img_size, opts.img_size), mode='bilinear')
        rgbs_refine    = torch.clamp( rgbs_refine.view(-1, 3, opts.img_size, opts.img_size), 0.0, 1.0 )
        
        predicted_rgbs_ = predicted_rgbs[0].permute(1,2,0).detach().cpu().numpy() * 255.0
        rgbs_warpped_   = rgbs_warpped[0].permute(1,2,0).detach().cpu().numpy() * 255.0
        rgbs_refine_    = rgbs_refine[0].permute(1,2,0).detach().cpu().numpy() * 255.0
        
        _rgbs = []; _depths = [];
        for i in range(opts.num_views):
            _rgbs.append( input_rgbs[0,i].permute(1,2,0).detach().cpu().numpy() * 255.0 ) # [H,W,3]
            # _input_depth = input_depths[0,i].detach().cpu().numpy() # [H,W]
            # _input_depth = np.clip( (_input_depth + 1)*0.5, 0, 1)
            # _input_depth = np.stack([_input_depth]*3, axis=-1) # [H,W,3]
            _depths.append( input_normals[0,i].permute(1,2,0).detach().cpu().numpy() * 255.0 )
            
        # [H,W*3,3]
        _rgbs   = np.concatenate(_rgbs, axis=1)
        _depths = np.concatenate( _depths, axis=1 )
        
        # get predicted results.
        _target_rgbs   = target_rgbs[0].permute(1,2,0).detach().cpu().numpy() * 255.0
        # _target_depths = target_depths[0,0].detach().cpu().numpy()
        
        # _target_depths = np.clip( (_target_depths + 1)*0.5, 0, 1)
        # _target_depths = np.stack([_target_depths]*3, axis=-1) * 255 # [H,W,3]
        
        # rgbs = []; depths = []; _rgbs = []; _depths = [];
        # for i in range(opts.batch_size): # the 
        #     rgbs.append( predicted_rgbs_[i] )
        #     depths.append( predicted_depths_[i] )
        #     _rgbs.append( target_rgbs[i,0].permute(1,2,0).detach().cpu().numpy() * 255.0 )
        #     _depths.append( target_depths[i,0,0].detach().cpu().numpy() * 10000.0 )

        # rgbs    = np.concatenate(rgbs, axis=1) # [H, W*B, 3]
        # depths  = np.concatenate(depths, axis=1) # [H, W*B]
        # _rgbs   = np.concatenate(_rgbs, axis=1) # [H, W*B, 3]
        # _depths = np.concatenate(_depths, axis=1) # [H, W*B]
        
        _rgbs   = np.concatenate([_rgbs, _target_rgbs, predicted_rgbs_], axis=1)
        _depths = np.concatenate([_depths, rgbs_warpped_, rgbs_refine_], axis=1)
        output  = np.concatenate([_rgbs, _depths], axis=0)
        
        Image.fromarray( output.astype(np.uint8) ).save(save_path_rgb)

def save_rgb_depths_depthdenoising(opts, visuals, local_rank, subject_name, epoch, idx, phase='training'):
    image_dir = os.path.join(opts.model_dir, phase)
    make_dir(image_dir)

    input_rgbs, input_masks, predicted_depths, predicted_normals = \
            visuals['rgbs'], visuals['masks'], visuals['_r_depths'], visuals['_r_normals']

    input_rgbs = input_rgbs.view(-1, opts.num_views, 3, opts.img_size, opts.img_size)
    input_masks  = input_masks.view(-1, opts.num_views, 1, opts.img_size, opts.img_size) # [B,1,H,W]
    predicted_depths = predicted_depths.view(-1, opts.num_views, 1, opts.img_size, opts.img_size);
    predicted_normals = predicted_normals.view(-1, opts.num_views, 3, opts.img_size, opts.img_size)
    predicted_depths, _, _, _ = _depth_normalizer(predicted_depths[:, :, 0], opts.valid_depth_min, opts.valid_depth_max)

    _rgbs = []; _depths = []; _normals = []
    for i in range(opts.num_views):
        _rgbs.append( input_rgbs[0,i].permute(1,2,0).detach().cpu().numpy() * 255.0 ) # [H,W,3]
        _input_depth = predicted_depths[0,i].detach().cpu().numpy() # [H,W]
        _input_depth = np.clip( (_input_depth + 1)*0.5, 0, 1)
        _input_depth = np.stack([_input_depth]*3, axis=-1) # [H,W,3]
        _depths.append( _input_depth * 255.0 )
        _normals.append( predicted_normals[0,i].permute(1,2,0).detach().cpu().numpy() * 255.0 ) # [H,W,3]

    _rgbs   = np.concatenate(_rgbs, axis=1)
    _depths = np.concatenate( _depths, axis=1 )
    _normals = np.concatenate( _normals, axis=1 )

    output  = np.concatenate([_rgbs, _depths, _normals], axis=0)

    save_path_rgb = os.path.join( image_dir, 'rgb_{}_rank{}_epoch{}_idx{}.jpg'.format(subject_name, local_rank, epoch, idx) )

    Image.fromarray( output.astype(np.uint8) ).save(save_path_rgb)

def save_rgb_depths_1k(opts, visuals, local_rank, subject_name, epoch, idx, phase='training'):
    image_dir = os.path.join(opts.model_dir, phase)
    make_dir(image_dir)
    
    if opts.is_train:
        assert opts.batch_size == 1, 'validating batch size must be 1.'
        
        input_rgbs, input_depths, input_masks, \
        target_rgbs, target_depths, target_masks, \
        predicted_rgbs, predicted_depths = \
            visuals['rgbs'], visuals['_r_depths'], visuals['masks'], \
            visuals['target_rgbs'], visuals['target_depths'], visuals['target_masks'], \
            visuals['_rgbs'], visuals['_depths'];
        
        # inputs resize, only when resolution 1K.
        input_rgbs = input_rgbs.view(-1, opts.num_views, 3, opts.img_size, opts.img_size)
        input_depths = F.interpolate(input_depths, (opts.img_size,opts.img_size), mode='nearest')
        input_depths = input_depths.view(-1, opts.num_views, 1, opts.img_size, opts.img_size)
        target_rgbs = target_rgbs.view(-1, 3, opts.img_size, opts.img_size)
        target_depths = target_depths.view(-1, 1, opts.img_size, opts.img_size)
        target_masks  = target_masks.view(-1, 1, opts.img_size, opts.img_size) # [B,1,H,W]
            
        predicted_rgbs  = predicted_rgbs.view(-1, opts.load_size * opts.load_size, 4, 3).view(-1, opts.load_size * opts.load_size, 12).permute(0,2,1) # [B*Nv, 12, N_p]
        predicted_rgbs = predicted_rgbs[:,[0,3,6,9,1,4,7,10,2,5,8,11]]
        # predicted_rgbs  = predicted_rgbs.reshape(-1, opts.load_size, opts.load_size, 2, 2, 3)
        # predicted_rgbs  = predicted_rgbs.view(-1, opts.load_size * opts.load_size, 4, 3)
        predicted_rgbs = F.fold(predicted_rgbs, (opts.img_size, opts.img_size), (2,2), stride=2).permute(0,2,3,1) # [b*n_v,3,h,w]
        predicted_rgbs = torch.clamp(predicted_rgbs, 0, 1) # [1, 3, H, W];
        # target_rgbs : [1, 3, H, W];
        
        predicted_depths = predicted_depths.view(-1, 1, opts.load_size, opts.load_size);
        predicted_depths = F.interpolate(predicted_depths, (opts.img_size,opts.img_size), mode='nearest')
        
        predicted_depths, _, _, _ = _depth_normalizer(predicted_depths, opts.valid_depth_min, opts.valid_depth_max)
        # the predicted depths normalization:
        # predicted_depths = (predicted_depths - opts.z_size) / opts.z_bbox_len * target_masks[:,0];
        # predicted_depths = (predicted_depths - opts.z_size) / opts.z_bbox_len;
        # input_depths,[B,N,1,H,W]
        input_depths, _, _, _ = _depth_normalizer(input_depths[:, :, 0], opts.valid_depth_min, opts.valid_depth_max)
        # normalize the target depths.
        target_depths, _, _, _ = _depth_normalizer(target_depths, opts.valid_depth_min, opts.valid_depth_max)
        
        predicted_rgbs_   = predicted_rgbs[0].detach().cpu().numpy() * 255.0 # [H,W,3]
        predicted_depths_ = predicted_depths[0,0].detach().cpu().numpy()
        target_masks      = target_masks[0,0].detach().cpu().numpy() # [H,W]
        
        _rgbs = []; _depths = [];
        for i in range(opts.num_views):
            _rgbs.append( input_rgbs[0,i].permute(1,2,0).detach().cpu().numpy() * 255.0 ) # [H,W,3]
            _input_depth = input_depths[0,i].detach().cpu().numpy() # [H,W]
            _input_depth = np.clip( (_input_depth + 1)*0.5, 0, 1)
            _input_depth = np.stack([_input_depth]*3, axis=-1) # [H,W,3]
            _depths.append( _input_depth * 255.0 )
            
        # [H,W*3,3]
        _rgbs   = np.concatenate(_rgbs, axis=1)
        _depths = np.concatenate( _depths, axis=1 )
        
        # get predicted results.
        _target_rgbs   = target_rgbs[0].permute(1,2,0).detach().cpu().numpy() * 255.0
        _target_depths = target_depths[0,0].detach().cpu().numpy()
        
        _target_depths = np.clip( (_target_depths + 1)*0.5, 0, 1)
        _target_depths = np.stack([_target_depths]*3, axis=-1) * 255 # [H,W,3]
        
        predicted_depths_ = np.clip( (predicted_depths_ + 1)*0.5, 0, 1)
        predicted_depths_ = np.stack([predicted_depths_]*3, axis=-1) * 255 # [H,W,3]
        
        # rgbs = []; depths = []; _rgbs = []; _depths = [];
        # for i in range(opts.batch_size): # the 
        #     rgbs.append( predicted_rgbs_[i] )
        #     depths.append( predicted_depths_[i] )
        #     _rgbs.append( target_rgbs[i,0].permute(1,2,0).detach().cpu().numpy() * 255.0 )
        #     _depths.append( target_depths[i,0,0].detach().cpu().numpy() * 10000.0 )

        # rgbs    = np.concatenate(rgbs, axis=1) # [H, W*B, 3]
        # depths  = np.concatenate(depths, axis=1) # [H, W*B]
        # _rgbs   = np.concatenate(_rgbs, axis=1) # [H, W*B, 3]
        # _depths = np.concatenate(_depths, axis=1) # [H, W*B]

        _rgbs   = np.concatenate([_rgbs, _target_rgbs, predicted_rgbs_], axis=1)
        _depths = np.concatenate([_depths, _target_depths, predicted_depths_], axis=1)
        output  = np.concatenate([_rgbs, _depths], axis=0)

        # PSNR values;
        psnr = cal_psnr( predicted_rgbs_, _target_rgbs, target_masks, in_masks=False ) # target_rgbs & target_rgbs are weighted by masks
        psnr = round(psnr, 2)
        # Image.fromarray( rgbs.astype(np.uint8) ).save(save_path_rgb)
        # Image.fromarray( _rgbs.astype(np.uint8) ).save(save_path_rgb_)
        # cv2.imwrite( save_path_depth, depths.astype(np.uint16) )
        # cv2.imwrite( save_path_depth_, _depths.astype(np.uint16) )
        save_path_rgb = os.path.join( image_dir, 'rgb_{}_rank{}_epoch{}_idx{}_psnr_{}.jpg'.format(subject_name, local_rank, epoch, idx, psnr) )

        Image.fromarray( output.astype(np.uint8) ).save(save_path_rgb)
        
def save_rgb_depths(opts, visuals, local_rank, subject_name, epoch, idx, phase='training'):
    image_dir = os.path.join(opts.model_dir, phase)
    make_dir(image_dir)
    save_path_rgb    = os.path.join(image_dir, 'rgb_{}_rank{}_epoch{}_idx{}.jpg'.format(subject_name, local_rank, epoch, idx))

    if opts.is_train:
        assert opts.batch_size == 1, 'validating batch size must be 1.'
        
        input_rgbs, input_depths, input_masks, \
        target_rgbs, target_depths, target_masks, \
        predicted_rgbs, predicted_depths, input_normals = \
            visuals['rgbs'], visuals['_r_depths'], visuals['masks'], \
            visuals['target_rgbs'], visuals['target_depths'], visuals['target_masks'], \
            visuals['_rgbs'], visuals['_depths'], visuals['_r_normals'];
        
        # inputs resize, only when resolution 1K.
        input_rgbs = input_rgbs.view(-1, opts.num_views, 3, opts.img_size, opts.img_size)
        input_normals = input_normals.view(-1, opts.num_views, 3, opts.img_size, opts.img_size)
        target_rgbs = target_rgbs.view(-1, 3, opts.img_size, opts.img_size)
        target_depths = target_depths.view(-1, 1, opts.img_size, opts.img_size)
            
        predicted_rgbs  = predicted_rgbs.view(-1, opts.img_size, opts.img_size, 3)
        predicted_depths = predicted_depths.view(-1, 1, opts.load_size, opts.load_size);
        predicted_depths, _, _, _ = _depth_normalizer(predicted_depths, opts.valid_depth_min, opts.valid_depth_max)
        # input_depths,[B,N,1,H,W]
        # normalize the target depths.
        target_depths, _, _, _ = _depth_normalizer(target_depths, opts.valid_depth_min, opts.valid_depth_max)
        
        predicted_rgbs_   = predicted_rgbs[0].detach().cpu().numpy() * 255.0
        predicted_depths_ = predicted_depths[0,0].detach().cpu().numpy()
        
        _rgbs = []; _depths = [];
        for i in range(opts.num_views):
            _rgbs.append( input_rgbs[0,i].permute(1,2,0).detach().cpu().numpy() * 255.0 ) # [H,W,3]
            # _depths.append( _input_depth * 255.0 )
            _depths.append( input_normals[0,i].permute(1,2,0).detach().cpu().numpy() * 255.0 )
            
        # [H,W*3,3]
        _rgbs   = np.concatenate(_rgbs, axis=1)
        _depths = np.concatenate( _depths, axis=1 )
        
        # get predicted results.
        _target_rgbs   = target_rgbs[0].permute(1,2,0).detach().cpu().numpy() * 255.0
        _target_depths = target_depths[0,0].detach().cpu().numpy()
        
        _target_depths = np.clip( (_target_depths + 1)*0.5, 0, 1)
        _target_depths = np.stack([_target_depths]*3, axis=-1) * 255 # [H,W,3]
        
        predicted_depths_ = np.clip( (predicted_depths_ + 1)*0.5, 0, 1)
        predicted_depths_ = np.stack([predicted_depths_]*3, axis=-1) * 255 # [H,W,3]
        
        # rgbs = []; depths = []; _rgbs = []; _depths = [];
        # for i in range(opts.batch_size): # the 
        #     rgbs.append( predicted_rgbs_[i] )
        #     depths.append( predicted_depths_[i] )
        #     _rgbs.append( target_rgbs[i,0].permute(1,2,0).detach().cpu().numpy() * 255.0 )
        #     _depths.append( target_depths[i,0,0].detach().cpu().numpy() * 10000.0 )

        # rgbs    = np.concatenate(rgbs, axis=1) # [H, W*B, 3]
        # depths  = np.concatenate(depths, axis=1) # [H, W*B]
        # _rgbs   = np.concatenate(_rgbs, axis=1) # [H, W*B, 3]
        # _depths = np.concatenate(_depths, axis=1) # [H, W*B]

        _rgbs   = np.concatenate([_rgbs, _target_rgbs, predicted_rgbs_], axis=1)
        _depths = np.concatenate([_depths, _target_depths, predicted_depths_], axis=1)
        output  = np.concatenate([_rgbs, _depths], axis=0)
        
        # Image.fromarray( rgbs.astype(np.uint8) ).save(save_path_rgb)
        # Image.fromarray( _rgbs.astype(np.uint8) ).save(save_path_rgb_)
        # cv2.imwrite( save_path_depth, depths.astype(np.uint16) )
        # cv2.imwrite( save_path_depth_, _depths.astype(np.uint16) )
        Image.fromarray( output.astype(np.uint8) ).save(save_path_rgb)

# e.g., get features from other four neighborviews
def warp_func(o_d, t_d, o_k, o_rt, t_ks, t_rts, t_feats, get_im=False, d_thresh=0.01, warp_d_thresh=0.03):
    # o_d : [B*1, 1, H, W], o_k(rt): [B*1, 3, 3(4)]
    # t_d : [B*N_nei, 1, H, W], t_feats : [B*N_nei, 3, HH, WW]
    # given source depth and k,rt matrix, warp target features given target k & rt matrix.
    # print(o_d.shape, t_d.shape, o_k.shape, t_ks.shape, t_feats.shape)
    B, _, H, W = o_d.shape # target depth only contain one view.
    HH, WW = t_feats.shape[-2:]
    num_neighbors_views = t_d.shape[0] // B
    
    ys, xs = torch.meshgrid( torch.arange(0, H, device=o_d.device, dtype=o_d.dtype),  # [H, W]
                             torch.arange(0, W, device=o_d.device, dtype=o_d.dtype) )
    ys, xs = ys.reshape(-1), xs.reshape(-1) # [h*w]
    # transform to the camera coordinate.
    xyz  = torch.stack( [xs, ys, torch.ones_like(xs)], dim=0 ) # [3, H*W]
    xyz  = xyz.view(1,3,-1).repeat(B,1,1) # [B,3,H*W]
    XYZ  = (torch.inverse(o_k) @ xyz) * o_d.view(B,1,-1) # [B,3,H*W] * [B,1,H*W] -> [B,3,H*W]
    # transform to world positions, if depth is 0 he re, the returned pos is the camera's position.
    XYZ  = torch.inverse(o_rt[:,:3,:3]) @ (XYZ - o_rt[:,:3,-1:]) # [B,3,3] * [B,3,H*W] -> [B,3,H*W]
    XYZ_ = XYZ.view(B,1,3,-1).repeat(1,num_neighbors_views,1,1).view(-1,3,H*W) # [B*N_v,3,H*W]
    # project to other views.
    xyz_t = t_rts[...,:3] @ XYZ_ + t_rts[...,-1:] # [B*N_v,3,H*W]
    p_d   = xyz_t[:,-1:,:].clone() # [B*N_v, 1, H*W]
    # set the invalid depth to -1.
    invalid  = p_d <= d_thresh # [B*N_v, 1, H*W]
    p_d_ori = p_d.clone() # original depths.
    p_d[invalid] = 1
    
    xyz_t     /= p_d # [x / d, y / d, 1.];
    pts_screen = t_ks @ xyz_t # [B*Nv, 3, h*w]
    # transform to [-1,1], unvalid regions are masked as -2;
    grids      = (pts_screen[:,:2] - H // 2) / (H // 2) # [-1, 1] [B*Nv, 2, h*w], H==W here
    grids[invalid.repeat(1,2,1)] = -1
    grids      = grids.permute(0,2,1)[:,:,None] # [B*Nv, h*w,1,2]
    
    # get visible masks.
    # step0. get the target depth values
    t_d_warp = F.grid_sample(t_d, grids, mode='nearest', align_corners=True)[..., 0] # [b*Nv, 1, h*w]
    in_visi_labels  = (t_d_warp - p_d_ori).abs() > warp_d_thresh # [B*N_v, 1, H * W]
    grids[in_visi_labels.permute(0, 2, 1)] = -1 # mark the invisible regions as -2;
    grids_maps = grids[:,:,0].permute(0,2,1).view(-1,2,H,W) # [B*Nv, 2, h, w]
    grids_maps = F.interpolate(grids_maps, (HH, WW), mode='bilinear') # [B*Nv, 2, h, w]
    
    if get_im:
        grids = grids_maps.view(B*num_neighbors_views, 2, -1).permute(0,2,1)[:,:,None]
        # the warpped features, all regions are valid now, then warpping.
        t_feats = torch.cat( [t_feats, torch.ones_like(t_feats[:,:1,...])], dim=1 )
        warped_feats = F.grid_sample(t_feats, grids, mode='bilinear', align_corners=True)[..., 0] # [B*Nv, c, h*w]
        warped_feats = warped_feats.view(t_ks.shape[0], -1, HH, WW) # B*Nnei, C+1. H, W
        mask = warped_feats[:, -1:].clone() # valid regions.
        warped_feats = warped_feats[:, :-1] * mask
        return grids_maps, warped_feats
    
    # cv2.imwrite( './visi_mask.png', ( (1 - in_visi_labels.view(-1, 1, H, W).int()[0,0].detach().cpu().numpy()) * 255).astype(np.uint8) )
    # cv2.imwrite( './warped_feats.png', ( warped_feats[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)[..., ::-1] )
    # exit()
    # return warped_feats, t_d_ori.view(-1, 1, H, W), in_visi_labels.view(-1, 1, H, W).int(), grids # [B*Nv, h*w, 1, 2]
    return grids_maps


def warp_func_features( feats, grids, B, Nv ):
    _,C,H,W = feats.shape
    grids_ = grids.view(B*Nv, 2, -1).permute(0,2,1)[:,:,None] # [BNv,h*w,1,2]
    t_feats = torch.cat( [feats, torch.ones_like(feats[:,:1,...])], dim=1 )
    warped_feats = F.grid_sample(t_feats, grids_, mode='bilinear', align_corners=True)[..., 0] # [B*Nv, c+1, h*w]
    warped_feats = warped_feats.view(B*Nv, -1, H, W) # B*Nv, C+1. H, W
    mask = warped_feats[:, -1:].clone() # valid regions.
    warped_feats = warped_feats[:, :-1] * mask
    return warped_feats

def proj_persp(points, K, R, T, p_depths, n_views, depth_thres=0.01, min_d=0.1, max_d=4.0, res=512):
    # points: [B, 3, N_point]
    # p_depths: [B, 1, N_point]
    if p_depths is not None:
        invalid_d = (p_depths < min_d) | (p_depths > max_d) # [B', 1, N_point]
        points = points[:, None].repeat(1, n_views, 1, 1).view(-1, *points.shape[-2:])
        
    pts = R @ points + T # [B*N_v, 3, N]
    # To screen coordinate
    ori_pts = pts.clone() # [B*N_v, 3, N]
    depth = pts[:, -1:, :].clone() # [B*N_v, 1, N]
    invalid = depth < depth_thres # invalid depth values.
    depth_ = depth.clone() # get depth values.
    depth_[invalid] = 1 # assign with 1, don't divide by this depth value.
    
    if p_depths is not None:
        invalid_d = invalid_d[:, None].repeat(1, n_views, 1, 1).view(-1, 1, invalid_d.shape[-1]) # [BNv, 1, N_rays]
        depth[invalid_d] = -2
    
    pts /= depth_ # [x / d, y / d, 1.];
    pts_screen = K @ pts # [f1 * x / d + f_x, f2 * y / d + f_y]
    pts_screen = pts_screen[:, :2, :] # [B, 2, N_p]
    # in dataset, the intrinsic matrix has been normalized to [f1 / fx, f2 / fy, 1], -> [-1,1]
    grids = (pts_screen - res // 2) / (res // 2) # to [-1,1] [B, 2, N]
    # invalid_ = invalid.repeat(1,2,1)[..., None].repeat(1,1,1,4).view(-1, 2, grids.shape[-1])
    grids[invalid.repeat(1,2,1)] = -2 # invalid with -1.
    if p_depths is not None:
        grids[invalid_d.repeat(1,2,1)] = -2;
    
    return ori_pts, grids, depth


# density properties: beta_init, beta_min;
class Density(nn.Module):
    def __init__(self, params_init={}):
        super().__init__()
        for p in params_init:
            param = nn.Parameter(torch.tensor(params_init[p]))
            setattr(self, p, param)

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)

# density
#  {
#   params_init{
#     beta = 0.1
#   }
#   beta_min = 0.0001
#  }

# density func: alpha * (0.5 + 0.5 * udf)
class LaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, params_init={}, beta_min=0.0001, device='cuda:0'):
        super().__init__(params_init=params_init)
        self.beta_min = torch.tensor(beta_min, device=device)

    def density_func(self, sdf, beta=None):
        # sdf = 1/beta * ();
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta
    
# my density func, regard as a smooth function for volume rendering.
class MyLaplaceDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, opts, device='cuda:0'):
        super(MyLaplaceDensity, self).__init__()
        self.register_parameter( 'beta', nn.Parameter(torch.tensor(opts.beta_init)) )  # default as 0
        self.beta_min = torch.tensor(opts.beta_min, device=device) # default as 1 / 10000.0
        # self.beta_max = torch.tensor(opts.beta_max, device=device) # default as 1 / 1000.0

    def density_func(self, occ, beta=None):
        # transform occ to robust sdf.
        # inside : -0.5, outside 0.5;
        
        _sdf = occ - 0.5 # [-0.5, +0.5]
        if beta is None:
            beta = self.get_beta() # the beta, such as 0.01

        alpha = 1 / beta # alpha <= 1/ beta; the default alpha can be set as 100 as max alpha;
        # for sdf <= 0, density = 0.5 * exp( sdf / beta ) / beta;
        # for sdf >  0, density = (1 - 0.5 * exp( - sdf / beta )) / beta;
        # 0.5 - 0.5 * (exp(-sdf/beta) - 1) -> 1 - 0.5 * exp(  - sdf / beta ); ( >0);
        # 0.5 + 0.5 * (exp(sdf / beta) - 1) ->  0.5 * exp(  sdf / beta ); ( <= 0)
        return alpha * (0.5 - 0.5 * _sdf.sign() * torch.expm1(-_sdf.abs() / beta))
    
    def forward(self, x):
        return self.density_func(x)

    def get_beta(self):
        # limit the beta. [0.0001, 0.001];
        beta = self.beta_min + self.beta.abs()
        return beta
    
# occ to density functions.
class OccDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, opts, device='cuda:0'):
        super(OccDensity, self).__init__()
        self.register_parameter( 'beta', nn.Parameter(torch.tensor(opts.beta_init)) )  # default as 0.1
        self.beta_min = torch.tensor(opts.beta_min, device=device) # default as 0.0001

    def density_func(self, occ, beta=None):
        if beta is None:
            beta = self.get_beta()

        # alpha = 1 / beta, the density = alpha * occ
        return occ / beta
    
    def forward(self, x):
        return self.density_func(x)

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta
    

class TempGrad(object):
    
    def __enter__(self):
        self.prev = torch.is_grad_enabled() # get previous gradients.
        torch.set_grad_enabled(True) # temp gradient enable.
    
    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_grad_enabled(self.prev) # recover the previous states for gradients.


# when UDF -> 0, the density -> +inf;
class LogisticDensity(nn.Module): # density value: s * e^{-s*udf} / (1+e^{-s*udf})^2
    
    def __init__(self, opts, device):
        super(LogisticDensity, self).__init__()
        # the paramter is initialized with 0.01, reload model will update the s.
        self.register_parameter('s', nn.Parameter(torch.tensor(opts.s_init)) )
        self.s_min = torch.tensor(opts.s_min, device=device)
    
    def density_func(self, udf):
        s = self.get_s()
        # print('s_min:', self.s_min.cpu(), ', s_now:', self.s.cpu())
        # during training, the s gradually becomes larger.
        return s * torch.exp(-s*udf)  / (1 + torch.exp(-s*udf))**2
    
    def forward(self, x):
        return self.density_func(x)

    def get_s(self):
        s = self.s_min + self.s.abs()
        return s


# 1. the batchify rays volume rendering functions.
@torch.no_grad()
def generate_rays(ks, rs, ts, res, masks=None, body_bbox=None, num_sampled_rays=1024, in_patch=False, in_bbox=False, device='cuda:0'):
    
    if num_sampled_rays is None: # when inference, get all rays for sampling.
        gpu_id = int(str(device).split(':')[-1]) # 0 or 1
        # sampled rays. [B, N_v, H, W, 6]
        ray_ori_dir = VoxelEncoding.rays_calculating(ks, rs, ts, res, gpu_id) # ~0.5ms
        ray_ori_dir = ray_ori_dir.view(ks.shape[0], -1, ray_ori_dir.shape[-1]) # [B, -1, 6]
        return ray_ori_dir, None # no need for xy.
    
    # sampling rays inside the masks, and sampling indeed num_rays. ~ 2ms.
    sampled_rays, sampled_xy = ray_sampling(ks, rs, ts, res, num_sampled_rays, masks, in_patch, in_bbox, body_bbox)
    sampled_rays = sampled_rays.view(sampled_rays.shape[0], -1, 6) # [B, N_rays, 6]
    
    return sampled_rays, sampled_xy


@torch.no_grad()
def batchify_rays(batch_rays, num_sampled_rays, target_num_views):
    # batch_rays: [B, N_rays, 6]
    batch_rays_batchified = []
    num_total_rays = batch_rays.shape[1] # suppose as N_v * H * W;
    num_batch_rays = num_sampled_rays * target_num_views # when inference, num of the target views is 1;
    num_batches = num_total_rays // num_batch_rays; # get the num of rays, may not enough here.
    # _batch_rays = batch_rays.view(batch_rays.shape[0], num_batch_rays, num_batches, 6).permute(2,0,1,3); # [N_b, B, N_rays, 6];

    for i in range(num_batches):
        batch_rays_batchified.append( batch_rays[:, i*num_batch_rays:(i+1)*num_batch_rays, ].contiguous() )
    # when the batchied rays are not in sequtial for all rays.
    left_rays = num_total_rays - num_batch_rays * num_batches
    if left_rays > 0:
        batch_rays_batchified.append( batch_rays[:, num_batch_rays*num_batches:, ].contiguous() )
    
    # [[B, N_rays*1, 6], ...]
    # for i in range(num_batches):
    #     batch_rays_batchified.append( _batch_rays[i] )
    
    return batch_rays_batchified

def high_res_rgbs_images_mean(rgbs_fold):
    # rgbs fold data, rgbs_fold : [B, N_v, 12, h, w]
    rgbs_chanel = rgbs_fold.shape[2] // 3
    # get the rgbs mean data in channels.
    rgbs_mean   = torch.stack( [rgbs_fold[:,:, :rgbs_chanel].mean(dim=2), 
                                rgbs_fold[:,:, rgbs_chanel:rgbs_chanel*2].mean(dim=2), 
                                rgbs_fold[:,:, rgbs_chanel*2:].mean(dim=2)], dim=2 )
    return rgbs_mean


def high_res_rgbs_rays_mean(rgbs_fold):
    # rgbs fold data, rgbs_fold : rays [B, N_rays, 12]
    rgbs_chanel = rgbs_fold.shape[-1] // 3
    # get the rgbs mean data in channels.
    rgbs_mean   = torch.stack( [rgbs_fold[..., :rgbs_chanel].mean(dim=-1), 
                                rgbs_fold[..., rgbs_chanel:rgbs_chanel*2].mean(dim=-1), 
                                rgbs_fold[..., rgbs_chanel*2:].mean(dim=-1)], dim=-1 )
    return rgbs_mean


def l2_normalize(x):
    # x : [..., 3].
    normalized_x = x / ( x.norm(2, dim=-1).unsqueeze(-1) + 1e-5 )
    return normalized_x

# 2. normalizer or other properties.
def rgb_normalizer(rgb_tensors, mask_tensors, mid=0.5, dist=0.5):
    # rgbs: [B, 3, H, W]; masks : [B, 1, H, W];
    tmp = (rgb_tensors - mid) / dist; # [-1, 1], rgb normalization.
    tmp *= mask_tensors # mask the back ground to 0.
    return tmp

def depth_normalizer(depth_tensors, mid, dist):
    # depth tensors: [B, N_v, N_rays, N_sampled, 1]
    # mid, dist : [B, N_v, ....]
    # step 1. expanding.
    if len(depth_tensors.shape) == 5:
        mid = mid[..., None];
        mid_expand  = mid.expand_as(depth_tensors)
        if type(dist) == torch.Tensor:
            dist = dist[..., None]; # [B, N_v, 1, 1, 1]
            assert len(depth_tensors.shape) == len(mid.shape) == len(dist.shape), "unsupported invalid shape."
            dist_expand = dist.expand_as(depth_tensors)
            return (depth_tensors - mid_expand) / (dist_expand / 2.0)
        else:
            return (depth_tensors - mid_expand) / dist # the dist is float value.
        
    else: # [B*N_v, N_p, 1]
        b, n_v = mid.shape[:2]
        depth_tensors = depth_tensors.view(b, n_v, -1, 1)
        mid_expand  = mid.expand_as(depth_tensors)
        if type(dist) == torch.Tensor:
            dist_expand = dist.expand_as(depth_tensors)
            depth_normalized = (depth_tensors - mid_expand) / (dist_expand / 2.0)
        else:
            depth_normalized = (depth_tensors - mid_expand) / dist
            
        # reshape to [BN_v, N_points, 1]
        depth_normalized = depth_normalized.view(b*n_v, -1, 1)
        return depth_normalized

def normalize_rgbd(rgbs, depths, depth_normalizer):
    rgb_mean = torch.Tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    rgb_std  = torch.Tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    
    depths_normalized, _, depth_mid, depth_dist = depth_normalizer( depths ) # to [-0.5, 0.5]
    rgbs_normalized    = (rgbs - rgb_mean.type_as(rgbs)) / rgb_std.type_as(rgbs) # standard normalization.
    return depths_normalized, depth_mid, depth_dist, rgbs_normalized


def unnormalize_depth(depths, mids, dists):
    b,c,h,w = depths.shape
    mids_ = mids.view(-1); 
    dists_ = dists.view(-1);
    depths_un = depths.clone()
    # unormalize-depths, the encoder -> (d - mids) / (dists / 2.0)
    for i in range(b):
        depths_un[i] = depths[i] * dists_[i] + mids_[i] # d_un = d' * (dists / 2.0) + mids.
                    
    return depths_un

def z_normalizer(z, mid, dist):
    b, n_v = mid.shape[:2]
    z = z.view(b, n_v, -1, 1)
    mid_expand  = mid.expand_as(z)
    dist_expand = dist.expand_as(z)
    z_normalized = (z - mid_expand) / (dist_expand / 2.0)
    z_normalized = z_normalized.view(b*n_v, -1, 1)
    return z_normalized


class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))


trunc_exp = _trunc_exp.apply


### other funcs:
def to_pcd_mesh(tsdf_vol, color_vol, bbox, voxel_size):
    tsdf  = tsdf_vol[0].cpu().numpy()
    color = color_vol[0].cpu().numpy()
    vol_bbox   = bbox[0, :, 0].cpu().numpy()
    voxel_size = voxel_size[0, 0].cpu().numpy() # the voxel's size;
    pc = get_point_cloud(tsdf, color, vol_bbox, voxel_size)
    verts, faces, norms, colors = get_mesh(tsdf, color, vol_bbox, voxel_size)
    pcwrite('./pc_all0.ply', pc)
    meshwrite('./mesh_all0.ply', verts, faces, norms, colors)

def get_point_cloud(tsdf_vol, color_vol, vol_bbox, voxel_size):
    """Extract a point cloud from the voxel volume.
    """
    from skimage import measure

    # Marching cubes
    verts = measure.marching_cubes_lewiner(tsdf_vol, level=0)[0] # verts is float values.
    verts_ind = np.round(verts).astype(int)
    verts = verts * voxel_size + vol_bbox
    # Get vertex colors
    rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    colors_b = np.floor( rgb_vals / 256 )
    colors_g = np.floor((rgb_vals - colors_b * 256) / 256)
    colors_r = rgb_vals - colors_b*256 - colors_g*256
    colors = np.floor(np.asarray([colors_b, colors_g, colors_r])).T
    colors = colors.astype(np.uint8)

    pc = np.hstack([verts, colors])
    return pc

def get_mesh(tsdf_vol, color_vol, vol_bbox, voxel_size):
    """Compute a mesh from the voxel volume using marching cubes.
    """
    from skimage import measure

    # Marching cubes
    verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol, level=0)
    verts_ind = np.round(verts).astype(int)
    verts = verts * voxel_size + vol_bbox
    # Get vertex colors
    rgb_vals = color_vol[verts_ind[:,0], verts_ind[:,1], verts_ind[:,2]]
    colors_b = np.floor(rgb_vals/256)
    colors_g = np.floor((rgb_vals-colors_b*256)/256)
    colors_r = rgb_vals-colors_b*256-colors_g*256
    colors = np.floor(np.asarray([colors_r,colors_g,colors_b])).T
    colors = colors.astype(np.uint8)
    
    return verts, faces, norms, colors

def pcwrite(filename, xyzrgb):
    """Save a point cloud to a polygon .ply file.
    """
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:].astype(np.uint8)

    # Write header
    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write("%f %f %f %d %d %d\n"%(
        xyz[i, 0], xyz[i, 1], xyz[i, 2],
        rgb[i, 0], rgb[i, 1], rgb[i, 2],
            ))
        
def meshwrite(filename, verts, faces, norms, colors):
    """Save a 3D mesh to a polygon .ply file.
    """
    # Write header
    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n"%(faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write("%f %f %f %f %f %f %d %d %d\n"%(
        verts[i,0], verts[i,1], verts[i,2],
        norms[i,0], norms[i,1], norms[i,2],
        colors[i,0], colors[i,1], colors[i,2],
        ))

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n"%(faces[i,0], faces[i,1], faces[i,2]))

    ply_file.close()

def draw_octree_nodes(nodes, indexs, _ray_ori, _ray_dir, sampled_depths, _idx, _sampled_idx):
    import open3d as o3d
    mesh_all = []
    # print(indexs, nodes.shape)
    # exit()

    # nodes0 = nodes[:indexs[3]].cpu().numpy() # [3, N]
    # nodes0 = nodes[indexs[1]:indexs[2]].cpu().numpy() # [3, N]
    # nodes0 = nodes[indexs[0]:indexs[1]].cpu().numpy() #[3, N]
    nodes0 = nodes[:].cpu().numpy() # [3, N]
    # nodes0 = nodes[indexs[1]:].cpu().numpy()
    # nodes0 = nodes[:indexs[1]].cpu().numpy()
    _idx = _idx.cpu().numpy()[0, 200]
    _sampled_idx = _sampled_idx.cpu().numpy()[0, 200]

    tmp = 0
    all_points = []
    all_lines  = []
    all_colors = []
    # basic_color = [20/255,70/255, 77/255]; highlighted_color = [100/255,140/255, 147/255]
    basic_color = [50/255,100/255, 107/255]; highlighted_color = [100/255,140/255, 147/255]
    # basic_color = [80/255,80/255,80/255]; highlighted_color = [120/255,120/255,120/255];
    new_color = [0,0,1]
    for i in range(nodes0.shape[0]):
        node = nodes0[i] # 5.x,y,z,size,b
        # if node[-2] == 1 and node[-1] == 0 and i in _sampled_idx: # batch id ==0 && valid.
        # view all the points for sampled_idx != 0;
        if node[-2] == 1 and node[-1] == 0: # batch id ==0 && valid.
            points = []
            for k in range(-1,3,2):
                for m in range(-1,3,2):
                    for n in range(-1,3,2):
                        points.append([node[0]+k*node[3]/2, node[1]+m*node[3]/2, node[2]+n*node[3]/2])
            points = np.array(points) # [8,3]
            all_points.append(points)
            lines = np.array([[0,1], [2,3], [4,5], [6,7], [0,4], [1,5],[2,6],[3,7],[0,2],[1,3],[4,6],[5,7]]) + tmp*8 # [12,2]
            all_lines.append(lines)
            tmp +=1

            # if i in [_sampled_idx[-1]]: # the intersected voxels
            #     all_colors.append(np.stack([np.array(new_color)]* lines.shape[0], axis=0))
            #     continue;
                
            if i in range(indexs[1],nodes0.shape[0]):
                all_colors.append(np.stack([np.array(highlighted_color)]* lines.shape[0], axis=0))
            else:
                all_colors.append(np.stack([np.array(basic_color)]* lines.shape[0], axis=0))

    all_points = np.concatenate(all_points, axis=0)
    all_lines  = np.concatenate(all_lines, axis=0)
    # all_colors = np.array([[0.4,0.4,0.4] for i in range(all_lines.shape[0])])
    all_colors = np.concatenate(all_colors, axis=0)
    # print(all_points.shape, all_lines.shape, all_colors.shape)
    # octree.
    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector(all_lines)
    lines_pcd.colors = o3d.utility.Vector3dVector(all_colors)
    lines_pcd.points = o3d.utility.Vector3dVector(all_points)
    # mesh
    mesh = o3d.io.read_triangle_mesh('./mesh_all0.ply', False)
    pcds = o3d.io.read_point_cloud('./pc_all0.ply')
    # pcds = o3d.io.read_triangle_mesh('./realtime_pifu_3571.obj', False)
    # pcds = o3d.io.read_point_cloud('/home/yons/my_Rendering/HumanRendering/depths_input/22_0401_data1_v3_all.ply')
    # pcds = o3d.io.read_point_cloud('/home/yons/my_Rendering/HumanRendering/depths_input/FRAME3587_all.ply')

    # draw rays.
    ray_ori = _ray_ori[0, 200, :].cpu().detach().numpy() # [3]
    ray_dir = _ray_dir[0, 200, :].cpu().detach().numpy() # [3]
    ray_len = 3
    ray_end = ray_ori + ray_len * ray_dir
    line_ray = o3d.geometry.LineSet()
    line_ray.lines = o3d.utility.Vector2iVector(np.array([[0,1]]))
    line_ray.colors = o3d.utility.Vector3dVector(np.array([[0.05,0.05,0.05]]))
    line_ray.points = o3d.utility.Vector3dVector(np.stack([ray_ori, ray_end], axis=0))

    # draw sampled points, the sampled points.
    sampled_z = sampled_depths[0, 200].cpu().numpy()
    sampled_points = ray_ori[:, None] + ray_dir[:, None] * sampled_z
    sampled_pcds = o3d.geometry.PointCloud()
    sampled_points = o3d.utility.Vector3dVector(sampled_points.T)
    sampled_pcds.points = sampled_points
    
    o3d.visualization.draw_geometries([lines_pcd, line_ray, sampled_pcds, pcds])

    # o3d.visualization.draw_geometries([lines_pcd, line_ray, sampled_pcds])
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # render_opt = vis.get_render_option()
    # render_opt.point_size = 0.2
    # render_opt.background_color = np.array([239/255, 244/255, 244/255])
    # # render_opt.background_color = np.array([219/255, 224/255, 244/255])
    # vis.add_geometry(lines_pcd)
    # # vis.add_geometry(pcds)
    # vis.add_geometry(line_ray)
    # vis.add_geometry(sampled_pcds)
    # # o3d.visualization.draw_geometries([lines_pcd, pcds])
    # vis.run()

########################## for geometries ########################
    
def gen_mesh(opts, model, val_data, device, epoch, use_octree=True):
    # get bbox for generate data.
    bbox_min = np.array( opts.bbox_min ) # [1, 3]
    bbox_max = np.array( opts.bbox_max ) # [1, 3]
    # get the path.
    mesh_dir = os.path.join(opts.model_dir, 'val_results')
    make_dir(mesh_dir)
    geo_save_path = os.path.join(mesh_dir, '{}_epoch{}.obj'.format(val_data['name'][0], epoch))
    
    t0 = time.time()
    with torch.no_grad():
        verts, faces  = reconstruction(opts, model, device, opts.num_views, opts.resolution, 
                                       bbox_min, bbox_max, use_octree=use_octree)
    t1 = time.time()
    print( 'Time for predict sdf field: {}'.format(t1 - t0) )

    save_obj_mesh(geo_save_path, verts, faces)


def eval_func(opts, model, num_views, points, device):
    points = np.expand_dims(points, axis=0) # [1, 3, N_Points.]
    if not opts.is_train: # especially for the front-face,
        points = np.concatenate([points] * num_views, axis=0) # [N_view * 1, 3, N_points]; {front_view; other 3 views};
    else:
        points = np.concatenate([points] * num_views, axis=0) # [N_view * 1, 3, N_points]; {front_view; other 3 views};

    samples = torch.from_numpy(points).to(device=device).float() # [N1, 3, N2].
    model.forward_query_occ(samples) # [[B, 1, N_points] * Stages.] -> [1, N_points]
    pred = model._predicted_occ[0] # select the first one.
    return pred.detach().cpu().numpy()
    

def reconstruction(opts, model, device, num_views, resolution, bbox_min, bbox_max, 
                   use_octree=True, num_sample_batch=10000):  
    coords, mat = create_grids(resolution, bbox_min, bbox_max)
    res_XYZ = (resolution, resolution, resolution)
    if use_octree:
        sdf = eval_grid_octree(opts, model, device, num_views, coords, eval_func, num_samples=num_sample_batch)
    else:
        raise("no support.")
    
    verts, faces = mcubes.marching_cubes(sdf, 0.5)
    verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
    verts = verts.T
    return verts, faces

