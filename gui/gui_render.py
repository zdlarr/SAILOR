
import glfw
import OpenGL.GL as gl

import collections
import os, sys, time, math
from py3nvml import py3nvml
import warnings
import imgui
from imgui.integrations.glfw import GlfwRenderer
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
from torch2trt import TRTModule
import matplotlib.pyplot as plt
from matplotlib import cm

from c_lib.FastNerf.dist import InferNerf
from c_lib.VoxelEncoding.voxel_encoding_helper import OcTreeOptions, sampling_multi_view_feats, integrate_depths, volume_render, volume_render_infer, volume_render_torch
from utils_render.utils_render import rgb_normalizer, z_normalizer, draw_octree_nodes, to_pcd_mesh, generate_rays, proj_persp
from gui.RenderGUIDataset import RenTestDataloader
from gui.gui_load_data import RawDataProcessor, BGMatting
# normalization;
from c_lib.VoxelEncoding.depth_normalization import DepthNormalizer
from c_lib.VoxelEncoding.dist import VoxelEncoding # Voxel encoding library.
# implicit fast occupancy field building, for building volumes;
from implicit_seg.functional import Seg3dLossless
import torch.multiprocessing as mp
import kornia
from utils_render.camera import Camera

# from utils_render.k4a_capturing import K4aCapturer
import threading

warnings.filterwarnings('ignore')


class FastNerf(nn.Module):
    # given the nerf's input data : [point's feature, output's results], in data parallel;
    # surface rendering for hydra attention;
    
    def __init__(self, num_views, num_sampled_points, params_density, params_color):
        super(FastNerf, self).__init__()
        # the parameters are saved in two GPU-card;
        self._geo_views, self._tex_views = num_views
        # parameters and batch_size;
        self.params_density = [params_density]
        self.params_color   = [params_color]
        self.batch_size     = 1
        self.num_sampled_points = num_sampled_points

    def forward(self, dist, z, psdf, geo_feats, rgbs, rgb_feats, ray_dirs):
        # 1, nviews, n_rays, 8, 1;
        device_Id = int(str(z.device).split(':')[1])
        num_rays = z.shape[0] // (self.batch_size * self._geo_views * self.num_sampled_points);

        density, color, feats_color = \
            InferNerf.infer_nerf(dist,
                                 z.view(-1,1), psdf.view(-1,1), geo_feats.view(-1,geo_feats.shape[-1]),
                                 rgbs.view(-1,3), rgb_feats.view(-1,rgb_feats.shape[-1]), ray_dirs.view(-1,3),
                                 self.params_density[device_Id], self.params_color[device_Id],
                                 self.batch_size, num_rays, self.num_sampled_points, self._geo_views, self._tex_views,
                                 device_Id)

        return density, color, feats_color


class FastBlend(nn.Module):
    
    def __init__(self, params_blend):
        super(FastBlend, self).__init__()
        self.params_blend = params_blend
    
    def forward(self, n_feats, rgb_feats):
        device_Id = int(str(n_feats.device).split(':')[1])
        
        return InferNerf.infer_blending(n_feats.view(-1, n_feats.shape[-1]), rgb_feats.view(-1, rgb_feats.shape[-1]), 
                                        self.params_blend[device_Id], device_Id)


class DataPrefecher(nn.Module):
    # cuda stream to preload next data, fetech & depth refine here;
    
    def __init__(self, rank, dataset, drm, first_fid, num_frames, resolution, shared_depths_buf, barriers, num_gpus, opts):
        super(DataPrefecher, self).__init__()
        self.dataset = dataset
        self.drm     = drm
        self.opts    = opts
        self.device  = 'cuda:0'
        self.rank    = rank
        self.num_gpus = num_gpus
        self.shared_depths_buf = shared_depths_buf
        self.barriers = barriers
        self.ns      = opts.num_views // num_gpus
        
        self.first_fid  = first_fid
        self.num_frames = num_frames
        self.resolution = resolution # default as 1k;
        self.depth_normalizer   = DepthNormalizer(opts, divided2=False)
        self.depth_normalizer_r = DepthNormalizer(opts)
        self.rgb_mean = torch.Tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        self.rgb_std  = torch.Tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        self._integrate_depths = integrate_depths.to(self.device)
        self.batch_size = self.opts.batch_size

        # preload first two data here;
        self.batch_data_cur = {};
        self.batch_data = {};
        self.stream0 = torch.cuda.Stream(self.device)
        
        with torch.cuda.stream(self.stream0):
            self.preload(self.first_fid, self.rank)
            self.batch_data_cur = self.batch_data.copy() # copy to cur data;
            self.preload(self.first_fid+1 if num_frames > 1 else self.first_fid, self.rank)
        
        torch.cuda.current_stream().wait_stream(self.stream0)
        # c the depth-data of current frame;
        self.sync_depth_data()
        # get the point-cloud's center positions.
        self.target_center = self.get_target_center(self.batch_data_cur)
    
    def get_target_center(self, output_data):
        inputs_data = [ output_data['rgbs_ds'], output_data['depths_ds'], output_data['masks_ds'], \
                        output_data['ks'], output_data['rts'] ]
        return self._integrate_depths.get_target_center(inputs_data, self.batch_size)[0]
    
    def sync_depth_data(self):
        # step 1. gpu depth to cpu; [2, 1, 512, 512], write to CPU shared mem;
        if self.num_gpus > 1:
            self.shared_depths_buf[self.rank*self.ns:(self.rank+1)*self.ns] *= 0;
            self.shared_depths_buf[self.rank*self.ns:(self.rank+1)*self.ns] += self.batch_data_cur['depths_ds'].cpu()
            self.barriers.wait() # sync data !!!;
            # cpu->gpu, sync cpu to the depth-data;
            self.batch_data_cur['depths_ds'] = self.shared_depths_buf.to(self.device, non_blocking=True)
        
        # normalize depth data;
        self.batch_data_cur['depths_ds_n'], self.batch_data_cur['depths_mid'], self.batch_data_cur['depths_dist'] \
            = self.depth_normalize( self.batch_data_cur['depths_ds'] )
    
    @torch.no_grad()
    def depth_refinement(self, rgbs, depths, masks):
        b = rgbs.shape[0]
        depths_n, _, mids, dists = self.depth_normalizer( depths )
        rgbs_n = (rgbs - self.rgb_mean.type_as(rgbs)) / self.rgb_std.type_as(rgbs)
        data = torch.cat( [rgbs_n, depths_n, masks], dim=1 )

        r_depths = self.drm( data ).float() # depth refinement model;

        r_depths = ( r_depths * dists.view(b,1,1,1) + mids.view(b,1,1,1) ) * masks

        del rgbs_n, data, depths_n, mids, dists

        return r_depths

    @torch.no_grad()
    def depth_normalize(self, depths):
        b = depths.shape[0]
        depths_n, _, mids, dists = self.depth_normalizer_r( depths )
        return depths_n, mids, dists
        
    def preload(self, frame_id, rank):
        # preload the data from the dataset;
        batch_data = self.dataset[frame_id % self.num_frames]; # current batch
        # CPU data to GPU;
        self.batch_data['name'] = batch_data['name']
        for item in ['rgbs', 'depths', 'masks', 'ks', 'rts']:
            self.batch_data[item] = batch_data[item].to(self.device, non_blocking=True)[0]
        
        ## preprocess the data from dataloader;
        # downsampled the images;
        self.batch_data['rgbs_ds']   = F.interpolate( self.batch_data['rgbs'], (self.resolution // 2, self.resolution // 2), mode='bilinear' )
        self.batch_data['depths_ds'] = F.interpolate( self.batch_data['depths'], (self.resolution // 2, self.resolution // 2), mode='nearest' )
        self.batch_data['masks_ds']  = F.interpolate( self.batch_data['masks'], (self.resolution // 2, self.resolution // 2), mode='nearest' )
        # data normalize;
        self.batch_data['rgbs_n']    = rgb_normalizer( self.batch_data['rgbs'], self.batch_data['masks'] )
        self.batch_data['rgbs_ds_n'] = rgb_normalizer( self.batch_data['rgbs_ds'], self.batch_data['masks_ds'] )
        # depth refinement and normalize;
        d_dir = os.path.join(batch_data['dir'][0], 'refined_depth')
        
        # when all four images exist, load the images from disk.
        # name0 = '{}.png'.format(rank*self.ns); name1 = '{}.png'.format((rank+1)*self.ns - 1)
        # if not ( os.path.exists(os.path.join(d_dir, batch_data['name'][0], name0)) and os.path.exists(os.path.join(d_dir, batch_data['name'][0], name1)) ):
        if True: # without saving depth images.
            # save the refined depth maps;
            # os.makedirs( os.path.join(d_dir, batch_data['name'][0]), exist_ok=True )
            self.batch_data['depths_ds'] = self.depth_refinement( self.batch_data['rgbs_ds'][rank*self.ns:(rank+1)*self.ns],
                                                                  self.batch_data['depths_ds'][rank*self.ns:(rank+1)*self.ns], 
                                                                  self.batch_data['masks_ds'][rank*self.ns:(rank+1)*self.ns] )
            torch.cuda.synchronize()
                                                               
            # for i in [int(rank*self.ns), int((rank+1)*self.ns -1)]: # save the output depths;
            #     cv2.imwrite(os.path.join(d_dir, batch_data['name'][0], '{}.png'.format(i)), 
            #                 (self.batch_data['depths_ds'][i%self.ns,0].cpu().detach().numpy() * 10000).astype(np.uint16))
            
        else:
            r_depths = []
            sub_dir = os.path.join(d_dir, batch_data['name'][0])
            for i in [int(rank*self.ns), int((rank+1)*self.ns - 1)]: # loading all the r depths data without depth sync;
                r_d = cv2.imread( os.path.join(sub_dir, '{}.png'.format(i)), -1 ) / 10000
                r_depths.append( torch.as_tensor( r_d )[None, None, ...] )
            self.batch_data['depths_ds'] = torch.cat(r_depths, dim=0).float() # [4,1,H,W], still in CPU here;
            self.barriers.wait()
        
        # cams, rotataion and trans matrix;
        self.batch_data['rs'] = self.batch_data['rts'][...,:3].contiguous()
        self.batch_data['ts'] = self.batch_data['rts'][...,-1:].contiguous()

    def next(self, nxt_frame_id):
        b_cur = self.batch_data_cur.copy()
        torch.cuda.current_stream().wait_stream(self.stream0)
        self.batch_data_cur = self.batch_data.copy()
        self.sync_depth_data() # sync depth data on CPU;
        b_nxt = self.batch_data_cur.copy()
        with torch.cuda.stream(self.stream0):
            self.preload(nxt_frame_id, self.rank)
            
        return b_cur, b_nxt

class PrefecherBuilder(nn.Module):
    
    def __init__(self, rank, hrnets, unets, batch_data_cur, params_density, shared_rgb_fts, shared_depth_fts, barrier, num_gpus, opts):
        super(PrefecherBuilder, self).__init__()
        # id information;
        self._rank  = rank
        self._num_gpus = num_gpus
        self._shared_rgb_fts   = shared_rgb_fts
        self._shared_depth_fts = shared_depth_fts
        self._barrier = barrier
        # modules;
        self._hrnet = hrnets.eval()
        self._unets = unets.eval()
        self._device = 'cuda:0'
        self._device_Id = int(str(self._device).split(':')[1])
        self._opts   = opts
        self._params_density = params_density
        # some variables;
        self.batch_size = self._opts.batch_size; # default batch size;
        self.ns      = opts.num_views // num_gpus
        
        self._integrate_depths = integrate_depths.to(self._device)
        b_min = torch.tensor([-1.0, -1.0, -1.0]).float()
        b_max = torch.tensor([ 1.0,  1.0,  1.0]).float()
        # realtime-pifu engine;
        self._reconEngine = Seg3dLossless(
                            query_func=self.query_func,
                            b_min=b_min.unsqueeze(0).numpy(),
                            b_max=b_max.unsqueeze(0).numpy(),
                            resolutions=[16+1, 32+1, 64+1, 128+1],
                            balance_value=0.5,
                            use_cuda_impl=True,
                            faster=True).to(self._device)
        # octree-structure: 
        self.octree_optioner = OcTreeOptions(opts.octree_level, opts.batch_size, opts.octree_rate, 
                                             opts.volume_dim, opts.num_max_hits_voxel, self._device)
        # all the output data, for cuda streams;
        self._output_data = {}
        
        self.stream0 = torch.cuda.Stream(self._device)
        
        # build octree variables;
        self.init_octree_variables()
        
        with torch.cuda.stream(self.stream0):
            self.pre_encoding(batch_data_cur, self._rank)
        
        torch.cuda.current_stream().wait_stream(self.stream0) # wait first process
        
    def sync_rgbd_features(self):
        if self._num_gpus != 2:
            return
        # step 1. gpu depth to cpu; [2, 1, 512, 512]; 
        self._shared_rgb_fts[self._rank*self.ns:(self._rank+1)*self.ns] *= 0
        self._shared_rgb_fts[self._rank*self.ns:(self._rank+1)*self.ns] += self._output_data['rgb_feats'].cpu() # CPU sync;
        self._barrier.wait() # sync data !!!;
        # cpu->gpu, concat; [4, 16, 128, 128], sync half of the data;
        if self._rank == 0:
            self._output_data['rgb_feats'] = \
                torch.cat( [self._output_data['rgb_feats'], self._shared_rgb_fts[2:4].to(self._device, non_blocking=True)], dim=0) # [4, 16, ...]
        else:
            self._output_data['rgb_feats'] = \
                torch.cat( [self._shared_rgb_fts[0:2].to(self._device, non_blocking=True), self._output_data['rgb_feats']], dim=0) # [4, 16, ...]

        # step 2. geo features;
        self._shared_depth_fts[self._rank*self.ns:(self._rank+1)*self.ns] *= 0
        self._shared_depth_fts[self._rank*self.ns:(self._rank+1)*self.ns] += self._output_data['geo_feats'].cpu() # CPU sync;
        self._barrier.wait() # sync data !!!;
        # cpu->gpu, concat; [4, 16, 128, 128], sync half of the data;
        if self._rank == 0:
            self._output_data['geo_feats'] = \
                torch.cat( [self._output_data['geo_feats'], self._shared_depth_fts[2:4].to(self._device, non_blocking=True)], dim=0) # [4, 16, ...]
        else:
            self._output_data['geo_feats'] = \
                torch.cat( [self._shared_depth_fts[0:2].to(self._device, non_blocking=True), self._output_data['geo_feats']], dim=0) # [4, 16, ...]
    
    def init_octree_variables(self):
        self._occupied_volumes = [] # save the indexs.
        self._num_occ_voxels   = []
        self._volume_dims      = [self._opts.volume_dim] * 3
        
        for i in range(self._opts.octree_level):
            vol_dim = [self._volume_dims[k] // self._opts.octree_rate**i for k in range(3)]
            o_volume = torch.empty([self.batch_size, *vol_dim], dtype=torch.int32, device=self._device) # index -1 is invalid.
            self._occupied_volumes.append( o_volume )
            self._num_occ_voxels.append( torch.empty([self.batch_size,1], dtype=torch.int32, device=self._device) )
    
    def fusion_occ_volume(self, v0, v1):
        # new volume; 
        fiter_idx = v0 < 0.001;
        min_occ_th = 0.3 if self._opts.rend_full_body else 0.45
        v0[v0 <= min_occ_th] = 0; v0[v0 >= 0.999] = 0 # set v0 to [0,1] volume;
        v0[(v0 < 0.999) & (v0 > min_occ_th)] = 1
        # old volume;
        v1[v1 > -1] = 1; v1[v1 <= -1] = 0; # set v1 to [0,1] volume;
        v1 += v0.int() # fusion;
        v1[v1 > 0] = 1; v1[v1 <= 0] = 0;
        v1[fiter_idx] = 0;
        self._num_occ_voxels[0] = v1.sum().unsqueeze(0).int()
        v1[v1 == 0] = -1; # update with [-1, 1]'s volume;
        self._occupied_volumes[0] = v1.int()
        
    @torch.no_grad()
    def query_func(self, points, batch_data):
        # get properties;
        ks, rs, ts, geo_feats = batch_data['ks'], batch_data['rs'], batch_data['ts'], batch_data['geo_feats']
        depths_mid, depths_dist = batch_data['depths_mid'], batch_data['depths_dist'] # get mid & depths;
        
        r_depths = batch_data['depths_ds']
        pts = points.permute(0,2,1)[:, [2,1,0],:]
        _, proj_xy, proj_z = proj_persp(pts, ks, rs, ts, None, self._opts.num_views)
        proj_xy = proj_xy.permute(0,2,1).view( proj_z.shape[0], -1, 1, 2 ); # [B*Nv, N_pts, 1, 2];
        # sample features; feats, z, diff_z ;
        sampled_geo_feats = F.grid_sample( geo_feats, proj_xy, align_corners=True, mode='bilinear')[..., 0] # [B*N, C, N_points.]
        sampled_geo_feats = sampled_geo_feats.permute(0,2,1).contiguous() # geo_feats;
        sampled_depths = F.grid_sample(r_depths, proj_xy, align_corners=True, mode='bilinear')[..., 0]
        t_psdf = torch.clamp(proj_z - sampled_depths, -0.01, 0.01) * (1.0 / 0.01);
        t_psdf = t_psdf.permute(0,2,1).contiguous() # t_psdf;
        # normalize z_; proj_z: [B*Nv, 1, N_p];
        z_normalized = z_normalizer( proj_z.permute(0,2,1), depths_mid, depths_dist )
        # pass the density network;
        occ = InferNerf.infer_density(z_normalized.view(-1,1), t_psdf.view(-1,1), sampled_geo_feats.view(-1,sampled_geo_feats.shape[-1]), 
                                      self._params_density, 1, z_normalized.shape[1], 4, self._device_Id)
        return occ
    
    @torch.no_grad()
    def pre_encoding(self, batch_data, rank):
        inputs_hrnet_rgbs = batch_data['rgbs_ds_n'][:self._opts.num_views][rank*self.ns : (rank+1) * self.ns]
        inputs_unet_depths = batch_data['depths_ds_n'][rank*self.ns : (rank+1) * self.ns]
        inputs_unet_rgbs  = batch_data['rgbs_n'][:self._opts.num_views]
        
        geo_feats, rgb_feats = self._hrnet( inputs_hrnet_rgbs, inputs_unet_depths )
        rgb_high_res_feats   = self._unets( inputs_unet_rgbs )
        torch.cuda.synchronize()

        # half of the features;
        self._output_data = batch_data.copy() # copy original data;
        self._output_data['geo_feats'] = geo_feats # [2, 16, 128, 128]
        self._output_data['rgb_feats'] = rgb_feats # [2, 16, 128, 128]
        self._output_data['rgb_high_res_feats'] = rgb_high_res_feats # [4, 16, 1024, 1024]
    
    @torch.no_grad()
    def forward_build_octree(self, output_data):
        # two-level Tree;
        inputs_data = [ output_data['rgbs_ds'], output_data['depths_ds'], output_data['masks_ds'], \
                        output_data['ks'], output_data['rts'] ]
        input_volumes = [self._occupied_volumes, self._num_occ_voxels]
        
        self._occupied_volumes, self._num_occ_voxels, _, _, self._vol_origin, self._vol_bbox, self._voxel_size = \
            self._integrate_depths(inputs_data, input_volumes, self.batch_size, 
                                   _level_volumes=self._opts.octree_level, _volume_res_rate=self._opts.octree_rate,
                                   _tsdf_th_low=self._opts.tsdf_th_low, _tsdf_th_high=self._opts.tsdf_th_high,
                                   build_octree=False)

        if self._opts.support_post_fusion: # used for post-fusion.
            # forward_build occupancy volumes.
            new_b_min = self._vol_bbox[:,[2,1,0],0].unsqueeze(1) # [BZ, 1, 3]
            new_b_max = self._vol_bbox[:,[2,1,0],1].unsqueeze(1) # [BZ, 1, 3]
            self._reconEngine.update_bmin_bmax( new_b_min, new_b_max )
            output_volume = self._reconEngine( batch_data=output_data )
            # forward the volume, [B,N,N,N];
            output_volume = F.interpolate( output_volume, (self._opts.volume_dim, self._opts.volume_dim, self._opts.volume_dim) )[0]
            # volume fusion operations and build multi_res volumes;
            self.fusion_occ_volume(output_volume, self._occupied_volumes[0])

        VoxelEncoding.build_multi_res_volumes(self._occupied_volumes, self._num_occ_voxels, 
                                              self._opts.octree_level, self._opts.octree_rate, self._device_Id)
        # import mcubes
        # verts, faces = mcubes.marching_cubes(self._occupied_volumes[0][0].cpu().detach().numpy(), 0.5)
        # mcubes.export_obj(verts, faces, './test.obj')
        self.octree_optioner.init_octree(self._occupied_volumes, self._num_occ_voxels, self._voxel_size, self._vol_bbox)
        torch.cuda.synchronize()
    
    def next(self, nxt_batch_data):
        torch.cuda.current_stream().wait_stream(self.stream0)
        self.sync_rgbd_features() # sync output_data for rgbd-features;
        b_cur = self._output_data.copy()
        # build octree data, after obtaining the volume data;
        self.forward_build_octree(b_cur)
        # start loading nxt batch_data;
        with torch.cuda.stream(self.stream0):
            self.pre_encoding(nxt_batch_data, self._rank)

        return b_cur, self.octree_optioner


class Render:

    def __init__(self, opts,
                 drm_path, hrnet_path, unet_path, nerf_path,
                 window_h=600, window_w=600,
                 num_gpus=2):
        self._opts   = opts
        self._num_gpus = num_gpus
        
        ##### define the common features
        self._drm_path   = drm_path
        self._hrnet_path = hrnet_path
        self._unet_path  = unet_path
        self._nerf_path  = nerf_path
        self._win_w      = window_w
        self._win_h      = window_h
        self.default_res = opts.img_size
        self.origin_res  = opts.load_size
        self.ns          = opts.num_views // num_gpus

        self._num_samplings = self._opts.num_sampled_points_coarse
        self._num_views  = self._opts.num_views
        # sampling rays parallelly;
        self.num_sampled_rays = self.origin_res ** 2 // num_gpus
        # shared_memory data, on CPU's data;

        self._render_process     = []
        self._shared_r_ds        = torch.full([self._num_views, 1, self.origin_res, self.origin_res], 0.0).share_memory_()
        self._shared_rgb_fts     = torch.full([self._num_views, 16, 128, 128], 0.0).share_memory_()
        self._shared_depth_fts   = torch.full([self._num_views, 16, 128, 128], 0.0).share_memory_()
        self._shared_o_c_details = torch.full([1, self.origin_res, self.origin_res, 4, 3], 0.0).share_memory_()
        self._shared_o_d         = torch.full([1, self.origin_res, self.origin_res, 1], 0.0).share_memory_()
        self._queue_rgbs_depths  = mp.Queue(maxsize=100) # queue for rgbs & depths data
        # self._queue_o            = mp.Queue(maxsize=100)
        self._queue_o            = collections.deque(maxlen=100)
        self._barrier            = mp.Barrier(num_gpus)
        self._lock               = mp.Lock()
        # cams;
        self._cam_center         = torch.full([3], 0.0).share_memory_()
        self._cam_dir            = torch.full([3], 0.0).share_memory_()
        self._cam_t              = torch.full([1], 0.0).share_memory_() # init the cam phase;
        self._cam_sint           = torch.full([1], 0.5).share_memory_() # init the cam phase;
        self._camera             = Camera( self._win_h, self._win_w, num_gpus ) # init camera object.

        self._cam_auto_ply       = torch.full([1], 0).share_memory_() # not auto ply;
        self._rendering_rgb      = torch.full([1], 1).share_memory_() # if rendering rgbs.
        self._show_sub_pixel     = torch.full([1], 0).share_memory_() # if show subpixels' featuree;
        self._rend_latency       = torch.full([1], 0.0).share_memory_() # record the rendering speed;
        self._cam_fix_y_axis     = torch.full([1], 0).share_memory_() # whether fix Y axis of rendering.
        self._cam_only_rotate_one_axis = torch.full([1], 1).share_memory_() # whether only rotate one axis in a time.
        self._cur_frame_id       = torch.full( [1], 0 ).int().share_memory_() # record the current frame's id.

        if opts.is_online_demo: # pre-define the data for rendering.
            self._fin_cam_data_preparing = torch.full([1], 0).int().share_memory_() # judge whether cam's data has been loaded.
            # new shared data {high res, low res.}
            self._data_high_res = torch.full( [self._num_views, 3, self.default_res, self.default_res], 0.0 ).share_memory_()
            self._data_low_res  = torch.full( [self._num_views, 2, self.origin_res, self.origin_res], 0.0 ).share_memory_()
            self._shared_kc     = torch.full( [self._num_views, 3, 3], 0.0 ).share_memory_()
            # queue for saving captured data.
            # self._queue_frames  = mp.Queue(maxsize=10)
            self._queue_frames = collections.deque(maxlen=10)

            # the background rgb for rendering.
            self._bg_rgbs      = torch.full( [self._num_views, 3, opts.rgb_raw_height, opts.rgb_raw_width], 0.0).share_memory_()
            # the curr, nxt, nxt-nxt data for loading.
            self._input_rgbs   = torch.full( [3, self._num_views, 3, opts.rgb_raw_height, opts.rgb_raw_width], 0.0).share_memory_()
            self._input_depths = torch.full( [3, self._num_views, 1, opts.depth_raw_height, opts.depth_raw_width], 0.0).share_memory_()

        # output data;
        self._o_mask  = torch.full( [1, 1, self.origin_res, self.origin_res], 0.0 ).share_memory_()
        bg_color_v    = 0.0
        self._o_bg    = torch.full( [1, 3, self._win_h, self._win_w], bg_color_v ).share_memory_()

        self._r_bg    = torch.full( [1], 0.0).share_memory_()
        self._g_bg    = torch.full( [1], 0.0).share_memory_()
        self._b_bg    = torch.full( [1], 0.0).share_memory_()
        # default rendering results;
        self._o_color = np.full( [self._win_h, self._win_w, 3], bg_color_v * 255, dtype=np.uint8 )
        print('Build shared memory data ... finished ')
        
        # init min-max depths;
        if self._opts.rend_full_body:
            self._depth_min = self._opts.valid_depth_min
            self._depth_max = self._opts.valid_depth_max
        else: # for portrait rendering.
            self._depth_min = 0.3
            self._depth_max = 0.5

        self._gauss = kornia.filters.GaussianBlur2d((3,3), (1,1)) # gaussian blur kernel.
        self._cmap  = cm.get_cmap('inferno')

        # self._max_pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self._max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) # eroding.
        # init opengl-module;
        self.init_gl()
        print('Initialize opengl data ... finished ')
        
        py3nvml.nvmlInit()
    
    def start_process(self):
        ##### start rendering process #####
        for gpu_id in range(self._num_gpus):
            p = mp.Process(target=self.render_frame, 
                           args=(gpu_id, 
                                 self._shared_r_ds, self._shared_rgb_fts, self._shared_depth_fts, self._shared_o_c_details, self._shared_o_d,
                                 self._barrier)
                          )
            self._render_process.append(p)
        # aggregate results' process;
        p_agg = threading.Thread(target=self.agg_results, args=())
        self._render_process.append(p_agg)

        # kinect capturing process.
        if self._opts.is_online_demo:
            # p_kinect = threading.Thread(target=self.k4a_connection, args=())
            p_kinect = threading.Thread(target=self.kinect_capturing, args=())
            self._render_process.append( p_kinect )

        for p in self._render_process:
            p.start()
        return
    
    def sync_rays_cpu(self, rank, barrier, shared_o_c_details, shared_o_d):
        # write frames to CPU;
        if self._num_gpus > 1:
            shared_o_c_details[:,rank::self._num_gpus] *= 0;
            shared_o_c_details[:,rank::self._num_gpus] += self._o_c_detail.view(1,self.origin_res//self._num_gpus,self.origin_res,4,3).cpu()
            barrier.wait()
            shared_o_d[:,rank::self._num_gpus] *= 0;
            shared_o_d[:,rank::self._num_gpus] += self._o_d.view(1,self.origin_res//self._num_gpus,self.origin_res,1).cpu()
            barrier.wait() # sync;
        
            if rank == 0: # only process 0, sync the queue, push the shared data to shared memory.
                self._queue_rgbs_depths.put(
                    { 'c' : shared_o_c_details, 'd': shared_o_d}
                )
        else: # push the data (rgbs & depths) to rgbs_depths queue.
            self._queue_rgbs_depths.put(
                {
                  'c' : self._o_c_detail.view(1, self.origin_res, self.origin_res, 4, 3).cpu(), 
                  'd' : self._o_d.view(1, self.origin_res, self.origin_res, 1).cpu()
                }
            )
    
    def render_frame(self, rank,
                     shared_r_ds, shared_rgb_fts, shared_depth_fts, shared_o_c_details, shared_o_d,
                     barrier):
        self.current_frame_id = 0
        os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
        
        # define the data for rendering, half of the data;
        self.rays_noise_c = torch.full( [1, self.num_sampled_rays, self._num_samplings], 0.5).cuda()
        self._o_c         = torch.full( [1, self.num_sampled_rays, 3], 0.0 ).cuda()
        self._o_c_feat    = torch.full( [1, self.num_sampled_rays, 30], 0.0 ).cuda()
        self._o_d         = torch.full( [1, self.num_sampled_rays, 1], 0.0 ).cuda()
        self._o_c_detail  = torch.full( [1, self.num_sampled_rays, 4, 3], 0.0 ).cuda()
        
        # loading dataset here, load all data to CPU first;
        if not self._opts.is_online_demo:
            print('Loading dataset in rank {}....'.format(rank))
            self.dataset = list(RenTestDataloader(self._opts, phase=self._opts.phase).get_iter())
            self.num_frames = len(self.dataset)
            print('Num frames of the captured data, in rank {} = {}'.format(rank, self.num_frames))
            if self.num_frames < 3: # static-data.
                frames_step = 0; step = 0; rot_step = 30 # rotation step-size;
            else: # dynamic-data.
                frames_step = 2; step = 1; rot_step = self.num_frames
        else: # online-data.
            frames_step = 2; step = 1; rot_step = 5
        
        ################################### load_models ###################################
        # half precision's cnn model;
        self._body_drm = TRTModule()
        self._hrnets   = TRTModule()
        self._unet     = TRTModule()
        # trt modules
        self.load_net_parameters( self._drm_path, self._hrnet_path, self._unet_path, self._nerf_path )
        self._fast_nerf  = FastNerf([self._num_views]*2, self._num_samplings, self.params_density_cuda, self.params_color_cuda)
        self._fast_blend = FastBlend(self.params_blend_cuda)
        print('networks loading finished.')

        # data prefecher & drm
        if self._opts.is_online_demo:
            while not self._fin_cam_data_preparing[0]:
                # print('Not finishing cam data preparing or cam synchronize...')
                pass
                
            self._data_prefecher = RawDataProcessor(rank, self._body_drm, 
                                                    self._bg_rgbs, self._input_rgbs, self._input_depths, self.default_res, 
                                                    self._data_high_res, self._data_low_res, self._shared_kc, 
                                                    barrier, self._num_gpus, self._opts)
        else:
            self._data_prefecher = DataPrefecher(rank, self.dataset, self._body_drm, self.current_frame_id, 
                                                 self.num_frames, self.default_res, shared_r_ds, barrier, self._num_gpus, self._opts);
        # data encoder
        self._encoder_prefecher = PrefecherBuilder(rank, self._hrnets, self._unet, self._data_prefecher.batch_data_cur, 
                                                   self.params_density_cuda, shared_rgb_fts, shared_depth_fts, barrier, self._num_gpus, self._opts)
        # init target view's cam data;
        self.init_camera_new( self._data_prefecher.batch_data_cur ) # init camera tar k, rt
        self._camera.initialize_K_RT( self.K_, self.RT_.cpu(), self._data_prefecher.target_center )

        barrier.wait()
        self._camera.set_fin_flag( rank ) # finish load initialized cameras.
        ######################################################################################
        
        while True:
            nxt_frame_id = self.current_frame_id + frames_step # get the next rendered framed_id;
            # self.update_cameras(self._cam_t[0] * 360.0)
            if self._cam_auto_ply[0] == 1 and rank == 0: # rotation degree along the up vector.
                self._cam_t[0] = self._cam_t[0] + 1 / rot_step # updated start dirs.
                self._cam_sint[0] = (torch.sin(self._cam_t[0]) + 1.0) / 2.0
    
            self.update_cameras_new()

            # load data to GPU;
            if not self._opts.is_online_demo:
                batch_data, batch_data_nxt = self._data_prefecher.next( nxt_frame_id )
            else:
                batch_data, batch_data_nxt = self._data_prefecher.next( [self._input_rgbs[-1], self._input_depths[-1]] )

            st = time.time()
            self.render( rank, batch_data, batch_data_nxt )
            ed = time.time()
            if rank == 0:
                t_diff = ed - st
                print(1. / t_diff, t_diff * 1000.0)
                self._rend_latency[0] = t_diff * 1000.0

            # sync to cpu;
            self.sync_rays_cpu( rank, barrier, shared_o_c_details, shared_o_d )
            self.current_frame_id += step

            barrier.wait()
    
    def init_gl(self):
        gl.glEnable(gl.GL_MULTISAMPLE)
        
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, self._win_w, self._win_h, 0,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)

        self.readFboId = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.readFboId)
        gl.glFramebufferTexture2D(gl.GL_READ_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                  gl.GL_TEXTURE_2D, self.tex, 0)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
    
    def load_net_parameters(self, drm_path, hrnet_path, unet_path, nerf_path):
        # loading TRT modules;
        self._body_drm.load_state_dict( torch.load(drm_path) )
        self._hrnets.load_state_dict( torch.load(hrnet_path) )
        self._unet.load_state_dict( torch.load(unet_path) )

        # loading nerf's MLP;
        state_dict_nerf = torch.load( nerf_path )
        state_dict_density = {}; state_dict_color = {}; state_dict_blend = {};
        params_density = []; params_color = []; params_blend = [];
        for k, v in state_dict_nerf.items():
            module_names = k.split('.')
            key_ = '.'.join(module_names[1:])
            if module_names[1] == 'feats_net' or module_names[1] == 'sigma_net' or module_names[1] == 'geo_ft_pca_net':
                state_dict_density[key_] = v
                params_density.append(v.view(-1))
            elif module_names[1] == '_hydra_e' or module_names[1] == 'rgb_net':
                state_dict_color[key_] = v
                params_color.append(v.view(-1))
            elif module_names[1] == '_blended_mlp':
                state_dict_blend[key_] = v
                params_blend.append(v.view(-1))
            else:
                continue
        
        params_density = torch.cat( params_density, dim=0).half() # 22521;
        params_color   = torch.cat( params_color, dim=0).half() # 15807 (11644 + 4163);
        params_blend   = torch.cat( params_blend, dim=0).half() # 15075;
        
        # net parameters to CUDA
        self.params_density_cuda = params_density.cuda()
        self.params_color_cuda   = params_color.cuda()
        self.params_blend_cuda   = params_blend.cuda()
    
    @torch.no_grad()
    def ray_intersect(self, ray_oris, ray_dirs, num_points, octree_opt):
        B, num_rays = ray_oris.shape[:2]
        # intersect;
        idx_c, min_depths_c, max_depths_c, ray_total_valid_len  = \
            octree_opt.ray_intersection(ray_oris, ray_dirs, instersect_start_level=self._opts.instersect_start_level,
                                        th_early_stop=self._opts.early_stop_th, only_voxel_traversal=True, is_training=False, 
                                        opt_num_intersected_voxels=-1)
        # sampling;
        sampled_idx_c, sampled_depths_c = \
            octree_opt.ray_voxels_points_sampling_coarse(idx_c, min_depths_c, max_depths_c, self.rays_noise_c, ray_total_valid_len, 
                                                         self._opts.instersect_start_level, num_points)
        sorted_depths, sampled_sort_idxs = octree_opt.sort_sampling(sampled_depths_c)
        # draw_octree_nodes(octree_opt.get_all_nodes(), self._octree_nodes_index, 
                        #   self._ray_ori, self._ray_dir, sampled_depths_c,
                        #   idx_c, sampled_idx_c)
        torch.cuda.synchronize()                        
        
        return sampled_idx_c, sampled_depths_c, sorted_depths, sampled_sort_idxs
    
    @torch.no_grad()
    def forward_nerf(self, ray_ori, ray_dir, sorted_dists, batch_data ):
        ks, rs, ts, geo_feats, rgb_feats = batch_data['ks'], batch_data['rs'], batch_data['ts'], \
                                           batch_data['geo_feats'], batch_data['rgb_feats']
        depths_mid, depths_dist = batch_data['depths_mid'], batch_data['depths_dist'] # get mid & depths;
        rgbs_n, r_depths = batch_data['rgbs_n'], batch_data['depths_ds']
        # filter rays.
        valid_rays = torch.prod( sorted_dists != -1, dim=-1 ) # [B,N_rays]
        self._o_c.fill_(0.0); self._o_d.fill_(0.0); self._o_c_feat.fill_(0.0); # reset all the buffer data;
        valid_rays_idx = torch.where( valid_rays[0] == True )[0]
        sorted_dists = sorted_dists[:, valid_rays_idx]
        rays_points  = ray_ori[:,valid_rays_idx][:,:,None] + ray_dir[:,valid_rays_idx][:,:,None] * sorted_dists[...,None] # [1, N_rays', N_p, 3]
        if rays_points.numel() == 0: # judge whether the rays are valid.
            return False

        ray_dir_cam  = self._ray_dir_cam[:, :, valid_rays_idx]
        # projection
        _, proj_xy, proj_z = proj_persp( rays_points.view(1,-1,3).permute(0,2,1),
                                         ks, rs, ts, sorted_dists.view(1,-1,1).permute(0,2,1), self._opts.num_views )
        proj_xy = proj_xy.permute(0,2,1).view( proj_z.shape[0], -1, 1, 2 );  # [B*Nv, N_p, 2 and 1]
        proj_z_normalized = z_normalizer( proj_z.permute(0,2,1), depths_mid, depths_dist ) # [B*Nv, Np, 1]
        # sampled features;
        sampled_geo_feats = F.grid_sample( geo_feats, proj_xy, align_corners=True, mode='bilinear' )[..., 0] # [B*N, C, N_points.]
        sampled_geo_feats = sampled_geo_feats.permute(0,2,1).contiguous() # geo_feats;
        sampled_depths = F.grid_sample( r_depths, proj_xy, align_corners=True, mode='bilinear' )[..., 0]
        t_psdf = torch.clamp( proj_z - sampled_depths, -0.01, 0.01 ) * (1.0 / 0.01);
        t_psdf = t_psdf.permute(0,2,1).contiguous() # t_psdf;
        sampled_rgb_feats = F.grid_sample( rgb_feats, proj_xy, align_corners=True, mode='bilinear' )[..., 0] # [B*N, C, N_points.]
        sampled_rgb_feats = sampled_rgb_feats.permute(0,2,1).contiguous() # geo_feats;
        sampled_rgbs    = F.grid_sample( rgbs_n, proj_xy, align_corners=True, mode='bilinear' )[..., 0]
        sampled_rgbs    = sampled_rgbs.permute(0,2,1).contiguous()
        # forward-nerf, using multi-process
        o_d, o_c, o_fts = self._fast_nerf( sorted_dists.view(-1,1), proj_z_normalized.view(-1,1), 
                                           t_psdf.view(-1,1), sampled_geo_feats.view(-1,16), 
                                           sampled_rgbs.view(-1,3), sampled_rgb_feats.view(-1,16), ray_dir_cam.view(-1,3) )
        self._o_c[:, valid_rays_idx] = o_c
        self._o_d[:, valid_rays_idx] = o_d
        self._o_c_feat[:, valid_rays_idx] = o_fts
        return True
    
    @torch.no_grad()
    def rays_interpolate(self, ray_dirs, num_rays):
        ks = self.tar_K.view(-1,3,3); rs = self.tar_R.view(-1,3,3);
        dirs = ray_dirs.view(-1,num_rays,3).permute(0,2,1)
        # pts: K * (R * dirs) -> [xt,yt,t]
        pts = ks @ (rs @ dirs)
        pts /= pts[:, -1:].clone(); pts = pts[:, :2]
        
        ptsx = pts[:,:1]; ptsy = pts[:,1:]
        ptsx_e = ptsx + 0.5; ptsy_e = ptsy + 0.5;
        ptsx_ey = torch.cat( [ptsx_e, ptsy], dim=1 ) # [B*Nv, 2, N_p]
        ptsxy_e = torch.cat( [ptsx, ptsy_e], dim=1 ) # [B*Nv, 2, N_p]
        ptsx_ey_e = torch.cat( [ptsx_e, ptsy_e], dim=1 ) # [B*Nv, 2, N_p]
        pts_interpolate = torch.stack( [pts, ptsx_ey, ptsxy_e, ptsx_ey_e], dim=-1 ).view(-1, 2, num_rays * 4) # [B*Nv, 2, N_p, 4]

        # dirs : R^{-1} * (K^{-1} @ [x,y,1])
        pts_interpolate_ep = torch.cat( [pts_interpolate, torch.ones_like(pts_interpolate[:,:1])], dim=1 ) # [B*Nv, 3, N_p*4]
        dirs_ep = (torch.inverse(rs) @ ( torch.inverse(ks) @ pts_interpolate_ep )).permute(0,2,1).contiguous() # [B*Nv, N_r*4, 3]
        dirs_ep_n = F.normalize( dirs_ep, dim=-1 ).view(1, -1, *dirs_ep.shape[-2:]).view(1,-1,3) # [B, Nv*Nr*4, 3]
        
        return dirs_ep_n
        
    @torch.no_grad()
    def forward_upsampling(self, ray_ori, ray_dir, batch_data):
        ks, rs, ts = batch_data['ks'], batch_data['rs'], batch_data['ts']
        rgb_high_res_fts, rgbs, depths = batch_data['rgb_high_res_feats'], batch_data['rgbs'], batch_data['depths_ds']
        #filter rays and data;
        valid_rays = (self._o_d > self._opts.valid_depth_min) & (self._o_d < self._opts.valid_depth_max) # [B, N_rays, 1]
        self._o_c_detail.fill_(0.0)
        valid_rays_idx = torch.where( valid_rays[0,:,0] == True )[0]
        num_sampled_rays = valid_rays_idx.numel()
        
        if num_sampled_rays == 0: # none valid rays;
            return 
            
        ray_ori = ray_ori[:, valid_rays_idx]; ray_dir = ray_dir[:, valid_rays_idx];
        o_d = self._o_d[:, valid_rays_idx]; o_c_fts = self._o_c_feat[:, valid_rays_idx];
        o_c = self._o_c[:, valid_rays_idx];
        # ray reusing;
        ray_dirs_ep = self.rays_interpolate( ray_dir, num_sampled_rays )
        # repeat features;
        ray_ori = ray_ori[:,:,None].repeat(1,1,4,1).view(1,-1,3)
        o_d     = o_d[:,:,None].repeat(1,1,4,1).view(1,-1,1) # [B, 4*N_rays, 1]
        o_c     = o_c[:,:,None].repeat(1,1,4,1).view(-1,3) # coarse rgb values;
        o_c_fts = o_c_fts[:,:,None].repeat(1,1,4,1).view(-1,o_c_fts.shape[-1]) # shared four features;
        
        # obtain the surface points;
        surface_pos = ray_ori + ray_dirs_ep * o_d;
        _, proj_xy, proj_z = proj_persp(surface_pos.permute(0,2,1), 
                                        ks, rs, ts, o_d.permute(0,2,1), self._opts.num_views)
        proj_xy = proj_xy.permute(0,2,1).view( 1, self._opts.num_views, -1, 4, 2 );
        proj_z  = proj_z.permute(0,2,1).view( 1, self._opts.num_views, -1, 4, 1 ); # [B, N_rays, 2 , 1]
        # sample neighbor view's features;
        rgbs_details  = self.sample_neighbor( rgb_high_res_fts, rgbs, depths[:self._opts.num_views], 
                                              ray_dirs_ep, rs, o_c, o_c_fts,
                                              proj_xy[:,:self._opts.num_views], proj_z[:,:self._opts.num_views],
                                            )
        self._o_c_detail[:, valid_rays_idx] = rgbs_details # [N_rays,4,3]
    
    @torch.no_grad()
    def sample_neighbor(self, rgbs_fts, rgbs, depths, dirs, rs, ray_c_rgbs, ray_rgbs_fts, proj_xy, proj_z ):
        n_nei = self.nei_Ids.shape[-1]
        feats = [];
        
        nei_ids = self.nei_Ids[0] # [2];
        # select feats & rgbs ...
        select_rgbs_fts = rgbs_fts[nei_ids] # [2, C, H, W]
        select_rgbs     = rgbs[nei_ids] # [2,3,H, W]
        select_depths   = depths[nei_ids] # [2, 1, h, w]
        select_dirs     = dirs.repeat(n_nei,1,1) # [2, n_rays, 3]
        select_rs       = rs[nei_ids] # [2, 3, 3]
        xy = proj_xy[0,nei_ids]; z = proj_z[0,nei_ids]
        xy_ = xy.view(n_nei,-1,1,2); z_ = z.view(n_nei,-1,1);
        # project to obtain features;
        sam_feats_rgbs = F.grid_sample( select_rgbs_fts, xy_, align_corners=True, mode='bilinear' )[...,0].permute(0,2,1)
        sam_rgbs       = F.grid_sample( select_rgbs, xy_, align_corners=True, mode='nearest' )[...,0].permute(0,2,1)
        sam_depths     = F.grid_sample( select_depths, xy_, align_corners=True, mode='nearest' )[...,0].permute(0,2,1)
        sam_dirs       = (select_rs @ select_dirs.permute(0,2,1)).permute(0,2,1)
        coff = torch.exp( -1.0 * 200 * (z_ - sam_depths) ** 2) # gaussian weights;
        feats = []
        for i in range(n_nei):
            feats.append( torch.cat( [sam_feats_rgbs[i], sam_dirs[i], coff[i]], -1 ) )
        
        weights = self._fast_blend( torch.cat( feats,-1 ), ray_rgbs_fts )
        final_c = torch.clamp( weights[:,0:1] * sam_rgbs[0] + weights[:,1:2] * sam_rgbs[1] + weights[:,-1:] * ray_c_rgbs, 0, 1 )
        final_c = final_c.view(-1, 4, 3)
        return final_c
        
    @torch.no_grad()
    def parse_rays(self, batch_rays, rs):
        self._ray_ori = batch_rays[..., :3].contiguous()
        self._ray_dir = batch_rays[..., 3:].contiguous() # [B, N_rays, 3]
        # transform the rays dir to camera's space. dir_local = R @ dir_global. [B*N_v, 3, 3] @ [B*N_v, 3, N]
        ray_dir_expand = self._ray_dir.permute(0, 2, 1)[:,None].repeat(1,self._opts.num_views,1,1)\
                        .view(self._opts.batch_size*self._opts.num_views, 3, -1)
        self._ray_dir_cam_ = (rs @ ray_dir_expand).permute(0,2,1).contiguous() # [B*N_v, N_rays, 3]
        self._ray_dir_cam  = self._ray_dir_cam_.view(self._opts.batch_size, self._opts.num_views, 
                                                     self.num_sampled_rays, 1, 3)\
                                                    .repeat(1,1,1,self._num_samplings,1) # [B, N_v, N_rays, N_sampled, 3]
    
    def agg_results(self):
        while True:
            if self._queue_rgbs_depths.empty():
                continue
            results = self._queue_rgbs_depths.get()

            # rgbs; [1,h,w,4,3]
            if self._show_sub_pixel[0] == 0: # get rgb results;
                # rgb results;
                if self._rendering_rgb[0] == 1:
                    rgb = results['c'].view(1, -1, 12).permute(0,2,1) # [B,12,N_rays]
                    rgb = rgb[:, [0,3,6,9,1,4,7,10,2,5,8,11]]
                    rgb = F.fold(rgb, (self.default_res, self.default_res), (2,2), stride=2) # [b,3,h',w'];
                    rgb = F.interpolate( rgb, (self._win_h, self._win_w), mode='bilinear' ) # [b,3,h',w']
                    
                    # depths;
                    depth = results['d'].view(1,1,self.origin_res, self.origin_res)
                    self._o_mask *= 0.0
                    self._o_mask[(depth > self._opts.valid_depth_min) & (depth < self._opts.valid_depth_max)] += 1
                    self._o_mask = self._gauss(-self._max_pool(-self._o_mask)) # erode the mask;
                    # self._o_mask = self._gauss(self._o_mask) # erode the mask;
                    mask = F.interpolate( self._o_mask, (self._win_h, self._win_w), mode='bilinear' ) # [B,1,H',W']
                        
                    data = rgb * mask + self._o_bg * (1-mask) # [B,3,H',W'] here for rgb images;
                else:
                    depth = results['d'].view(1,1,self.origin_res, self.origin_res)
                    depth = F.interpolate( depth, (self._win_h, self._win_w), mode='nearest' )[0,0].numpy()
                    # depth = np.clip((depth - self._depth_min) / (self._depth_max - self._depth_min), 0, 1.0)
                    # depth_min = depth[depth > 0.3].min()
                    depth = np.clip((depth - 0.4) / (depth.max() - 0.4), 0, 1.0)
                    data = torch.from_numpy(self._cmap(depth))[...,:3].permute(2,0,1)[None] # mapping to rgb-data.

            else: # [1, h,w,4,3]
                rgb = results['c'].permute(0,4,1,3,2).reshape(1,3,self.origin_res,2,2,self.origin_res) \
                                  .permute(0,1,3,2,4,5).reshape(1,3,self.default_res,self.default_res)
                data = F.interpolate( rgb, (self._win_h, self._win_w), mode='bilinear' ) # [b,3,h',w']
            
            o_color = (torch.flip( data, (2,) )[0].permute(1,2,0).contiguous() * 255).to(torch.uint8).numpy() # [h,w,3]
            # self._queue_o.put_nowait(o_color) # queue_o
            self._queue_o.append(o_color)

            # update new frame.
            if self._opts.is_online_demo and (self._queue_frames.__len__() > 0):
                # finish rendering one frame, load the latest data.
                frame_data = self._queue_frames.popleft()
                self.update_latest_frame_data( frame_data['rgbs'], frame_data['depths'] )

    def results_to_opengl(self):
        if self._queue_o.__len__() > 0:
            self._o_color *= 0
            self._o_color += self._queue_o.popleft() # update color tensor;

        ptr = self._o_color.data
            
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self._win_w, self._win_h,
                           gl.GL_RGB, gl.GL_UNSIGNED_BYTE, ptr)
        
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.readFboId)
        gl.glBlitFramebuffer(0, 0, self._win_w, self._win_h,
                             0, 0, self._win_w, self._win_h,
                             gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)

        gl.glFlush()
    
    @torch.no_grad()
    def render(self, rank, batch_data, batch_data_nxt):
        # A stream rendering system;
        encoded_data, octree_opts = self._encoder_prefecher.next(batch_data_nxt)
        # generate rays, [B, N_rays // 2, 6] and obtain the rays for curr rank;
        ray_ori_dirs = VoxelEncoding.rays_calculating_parallel(self.tar_K, self.tar_R, self.tar_T, self.origin_res, rank, self._num_gpus) \
                                    .view(1,-1,6) # seperate rays for multi-processing, sampled half rays for rendering;
        self.parse_rays( ray_ori_dirs, batch_data['rs'] )
        # ray-voxel intersection
        _,_, sorted_dists, _ = self.ray_intersect( self._ray_ori, self._ray_dir, self._num_samplings, octree_opts )
        # forward nerf;
        flag = self.forward_nerf( self._ray_ori, self._ray_dir, sorted_dists, encoded_data )
        if self._opts.adopting_upsampling and flag:
            self.forward_upsampling( self._ray_ori, self._ray_dir, encoded_data )
        torch.cuda.synchronize()

    @torch.no_grad()
    def init_camera_new(self, fsv_batch_data):
        self.ks, self.rts = fsv_batch_data['ks'], fsv_batch_data['rts']
        k, rt = self.ks[0], self.rts[0]
        self.K_  =  k
        self.RT_ = rt

    def get_neighbor_views_id(self, s_rts, t_rts):
        s_cpos = []; t_cpos = [];

        for vid in range(0, self._opts.num_views):
            r = s_rts[vid,:3,:3]; t = s_rts[vid,:3,-1:];
            # c_pos = - torch.inverse(r) @ t # [3,1]
            c_pos = - r.T @ t # [3,1]
            s_cpos.append( F.normalize(c_pos, dim=0) )

        t_rt = t_rts[0,0]; # [3, 4]
        r = t_rt[:3,:3]; t = t_rt[:3,-1:]
        # c_pos = - torch.inverse(r) @ t # [3,1]
        c_pos  = - r.T @ t
        t_cpos.append( F.normalize(c_pos, dim=0) ) # suppose target pos is [0,0,0]
        
        # select neighbor views.
        neighbor_ids = []
        for t_pos in t_cpos:
            distances = [];
            for s_pos in s_cpos:
                cos_distance = (t_pos * s_pos).sum() # [-1, 1]
                distances.append(cos_distance)
            v, idx = torch.stack(distances, dim=0).topk(2, dim=0, largest=True) # [2]
            if not self._opts.rend_full_body: # only select nearest view;
                idx[1] = idx[0] # top 1 views;
            neighbor_ids.append(idx)
        
        self.nei_Ids = torch.stack( neighbor_ids, dim=0 ) # [N_t, 2]

    @torch.no_grad()
    def update_cameras_new(self):
        # get camera's K, RT from the cam.
        self.tar_K = self.K_[None, None].float() # [1,1,3,3]
        self.tar_RT = self._camera.get_w2c()[None, None].float()
        self.tar_R = self.tar_RT[..., :3].contiguous() # [1,1,3,3]
        self.tar_T = self.tar_RT[..., -1:].contiguous() # [1,1,3,1]

        self.get_neighbor_views_id( self.rts[:self._num_views], self.tar_RT )

    ############################ Utils for online-demo data ################################
    @torch.no_grad()
    def update_initial_frame_data(self, bg_rgbs, rgbs, depths):
        # bg_rgbs: [N_views,3,H,W]; rgbs: [N_views,3,H,W] * 2, depths: [N_views,1,H,W] * 2
        rgbs_frame0, rgbs_frame1 = rgbs
        depth_frame0, depth_frame1 = depths
        # copy the background rgbs.
        self._bg_rgbs.copy_( torch.tensor(bg_rgbs) )
        
        # copy captured rgb & depth data.
        self._input_rgbs[0].copy_( torch.tensor(rgbs_frame0) )
        self._input_rgbs[1].copy_( torch.tensor(rgbs_frame1) )

        self._input_depths[0].copy_( torch.tensor(depth_frame0) )
        self._input_depths[1].copy_( torch.tensor(depth_frame1) )
        
        # update the flag
        self._fin_cam_data_preparing *= 0 # not finished loading images, stop rendering process.

    @torch.no_grad()
    def update_latest_frame_data(self, rgb, depth):
        # copy the latest data to rgb&depths buffers.
        self._input_rgbs[-1].copy_( torch.tensor(rgb) ) # [N_views, 3, H, W]
        self._input_depths[-1].copy_( torch.tensor(depth) ) # [N_views ,1, H, W]
        # finishing data loading.
        self._fin_cam_data_preparing *= 0
        self._fin_cam_data_preparing += 1
    
    ######################## Utils For kinect capturing #############################
    def k4a_connection(self):
        pass

    def kinect_capturing(self):
        pass

def glfw_mouse_button_callback(window, button, action, mods):
    if (imgui.get_io().want_capture_mouse):
        return
    
    rend: Render = glfw.get_window_user_pointer(window)

    if not rend._camera.get_fin_flag(): # not finised load krt.
        return

    x, y = glfw.get_cursor_pos(window)

    if action == glfw.PRESS:
        
        CONTROL = mods & glfw.MOD_CONTROL
        rend._camera.begin_drag(
            x, y, button == glfw.MOUSE_BUTTON_MIDDLE, button == glfw.MOUSE_BUTTON_LEFT, CONTROL or rend._cam_fix_y_axis, 
            button == glfw.MOUSE_BUTTON_RIGHT
        )

    elif action == glfw.RELEASE:
        rend._camera.end_drag()

def glfw_cursor_pos_callback(window, x, y):
    rend: Render = glfw.get_window_user_pointer(window)
    
    camera = rend._camera
    if not camera.get_fin_flag(): # not finised load krt.
        return
    
    camera.drag_update_new(x, y, rend._opts.rend_full_body, rend._opts.is_online_demo, rend._cam_only_rotate_one_axis)

def glfw_scroll_callback(window, xoffset, yoffset):
    if (imgui.get_io().want_capture_mouse):
        return
        
    rend: Render = glfw.get_window_user_pointer(window)
    camera = rend._camera
    if not camera.get_fin_flag(): # not finised load krt.
        return

    speed_factor = 1e-1
    movement = -speed_factor if yoffset < 0 else speed_factor
    camera.move( movement )

def glfw_key_callback(window, key, scancode, action, mods):
    if (imgui.get_io().want_capture_keyboard):
        return
    
    rend: Render = glfw.get_window_user_pointer(window)
    camera = rend._camera
    if not camera.get_fin_flag(): # not finised load krt.
        return

    if action == glfw.PRESS and key == glfw.KEY_R:
        camera.reset_RT(camera.get_w2c_init())

def glfw_bind_callback(window, rend):
    glfw.set_window_user_pointer(window, rend)
    glfw.set_cursor_pos_callback(window, glfw_cursor_pos_callback)
    glfw.set_mouse_button_callback(window, glfw_mouse_button_callback)
    glfw.set_key_callback(window, glfw_key_callback)
    glfw.set_scroll_callback(window, glfw_scroll_callback)

def main(opts):
    window = impl_glfw_init(opts)  # prepare gl bindings
    impl = imgui_init(window)  # prepare imgui related init
    font = imgui_load_font(impl, opts.font_path, 14)  # prepare gui font
    
    rend = Render(opts, opts.drm_path, opts.hrnet_path, opts.unet_path, opts.nerf_path,
                  opts.window_h, opts.window_w, opts.num_gpus)
    rend.start_process() # start rendering processes on GPUS;

    glfw_bind_callback(window, rend)

    while not glfw.window_should_close(window):
        rend.results_to_opengl()  # render network output to main frame buffer

        draw_imgui(font, rend)  # defines GUI elements
        impl.render(imgui.get_draw_data())  # render actual GUI elements
        impl.process_inputs()  # keyboard and mouse inputs for imgui update

        glfw.swap_buffers(window)
        glfw.poll_events()  # process pending events, keyboard and stuff

    impl.shutdown()
    glfw.terminate()

def imgui_init(window):
    imgui.create_context()
    impl = GlfwRenderer(window)  # show window when network's already prepared
    return impl

def imgui_load_font(impl, filepath, fontsize):
    io = imgui.get_io()
    font = io.fonts.add_font_from_file_ttf(
        filepath, fontsize,
    )
    impl.refresh_font_texture()
    return font

def draw_imgui(font, rend):
    imgui.new_frame()
    # auto-change the rend gui;

    with imgui.font(font):
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("System", True):
                clicked_quit, selected_quit = imgui.menu_item("Quit", 'Ctrl+Q', False, True)
                if clicked_quit:
                    exit(1)
                imgui.end_menu()
            imgui.end_main_menu_bar()

        imgui.begin("Rend Backend: GPU_0, GPU_1.")
        # silders & buttens;
        if imgui.collapsing_header("Render. Latency: %.3f" % rend._rend_latency[0].numpy() + "ms." , flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            auto_changed, rend._cam_auto_ply[0]  = imgui.checkbox('Auto Rot', rend._cam_auto_ply[0])
            auto_changed, rend._rendering_rgb[0] = imgui.checkbox('RGB / Depth', rend._rendering_rgb[0])
            _, rend._show_sub_pixel[0] = imgui.checkbox('Show SubPixels', rend._show_sub_pixel[0])
            
            # auto update cameras.
            if rend._cam_auto_ply[0]:
                if rend._camera.get_fin_flag(): # not finised load krt.
                    rend._camera.yaw( rend._cam_sint[0] * 180.0 - 90.0, rend._opts.rend_full_body, rend._opts.is_online_demo ) # [-90 ~ 90 degree]
                    # rend._camera.yaw( rend._cam_sint[0] * 360 - 180.0, rend._opts.rend_full_body, rend._opts.is_online_demo ) # [-180 ~ 180 degree]
                    # rend._camera.yaw( rend._cam_sint[0] * 18 - 9.0, rend._opts.rend_full_body, rend._opts.is_online_demo ) # [-9 ~ 9 degree]
            else:
                rend._camera.record_static_start_dir()

            changed, rgb_bg = imgui.color_edit3("BG color", rend._r_bg, rend._g_bg, rend._b_bg)
            rend._r_bg, rend._g_bg, rend._b_bg = rgb_bg[0], rgb_bg[1], rgb_bg[2]
            rend._o_bg *= 0
            rend._o_bg[:,0] += rend._r_bg; rend._o_bg[:,1] += rend._g_bg; rend._o_bg[:,2] += rend._b_bg; 
            
        if imgui.collapsing_header('Camera', flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            new_center = torch.tensor(imgui.input_float3('Center', *rend._camera.center.numpy())[1])
            new_front  = torch.tensor(imgui.input_float3('Front', *rend._camera.v_front.numpy())[1])
            new_up     = torch.tensor(imgui.input_float3('Up', *rend._camera.v_world_up.numpy())[1])
            
            if rend._camera.get_fin_flag():
                rend._camera.center *= 0;     rend._camera.center     += new_center;
                rend._camera.v_front *= 0;    rend._camera.v_front    += new_front;
                rend._camera.v_world_up *= 0; rend._camera.v_world_up += new_up;

                rend._camera.update_trans()
            
            _, tmp_sint = imgui.slider_float("Rotation", rend._cam_sint[0], 0, 1.0) # located in [0,1]
            # rend._cam_t[0] = np.arcsin(tmp_sint * 2 - 1)

            auto_changed, rend._cam_fix_y_axis[0] = imgui.checkbox('Fix Y Axis', rend._cam_fix_y_axis[0])
            auto_changed, rend._cam_only_rotate_one_axis[0] = imgui.checkbox('Only One Axis', rend._cam_only_rotate_one_axis[0])

            if imgui.button("Reset Camera (or key: R)"):
                if rend._camera.get_fin_flag(): # not finised load krt.
                    rend._camera.reset_RT(rend._camera.get_w2c_init())
        
        imgui.end()
    
    imgui.render()

def impl_glfw_init(opts):
    window_name = "SAILOR"
    width, height = opts.window_w, opts.window_h
    
    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    # OS X supports only forward-compatible core profiles from 3.2
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(
        int(width + opts.gui_w), int(height), window_name, None, None
    )
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window


if __name__ == '__main__':
    from gui.RenTestOptions import RenTestOptions

    opts = RenTestOptions().parse()
    rend_model_dir = './accelerated_models'
    # rendering dir.
    opts.ren_data_root = './test_data/static_data0'
    # opts.ren_data_root = './test_data/dynamic_data0'

    opts.num_gpus = 2 # [2 as default,1 (slow)]

    if opts.num_gpus == 2: # batch-size is 2 for each trt-module.
        drm_path   = os.path.join( rend_model_dir, 'depth_refine_trt_parallel_v0.pth' )
        # drm_path   = os.path.join( rend_model_dir, 'depth_refine_trt_parallel_v1.pth' ) # lighter version.
        hrnet_path = os.path.join( rend_model_dir, 'hrunet_trt_parallel.pth' )
    elif opts.num_gpus == 1: # batch-size is 4 for each trt-module.
        drm_path   = os.path.join( rend_model_dir, 'depth_refine_trt_parallel_v0_big.pth' )
        # drm_path   = os.path.join( rend_model_dir, 'depth_refine_trt_parallel_v1_big.pth' ) # lighter version.
        hrnet_path = os.path.join( rend_model_dir, 'hrunet_trt_parallel_big.pth' )

    unet_path  = os.path.join( rend_model_dir, 'high_res_unet_trt.pth' )
    nerf_path  = './checkpoints_rend/SAILOR/latest_model_BasicRenNet.pth'
    # bgmatting_path = os.path.join( rend_model_dir, 'matting_trt_parallel.pth' ) # tensort matting.
    
    opts.window_w = 600
    opts.window_h = 600
    opts.gui_w    = 220
    opts.drm_path   = drm_path
    opts.hrnet_path = hrnet_path
    opts.unet_path  = unet_path
    opts.nerf_path  = nerf_path
    # opts.bgmatting_path = bgmatting_path
    opts.bgmatting_path = ''
    opts.font_path  = './data/Caskaydia Cove Nerd Font Complete.ttf'
    
    opts.is_online_demo   = False # offline-demo.

    ####### for processing online data. #######
    opts.rgb_raw_width    = 2560
    opts.rgb_raw_height   = 1440
    opts.depth_raw_height = 1024
    opts.depth_raw_width  = 1024
    opts.cameras_xml_path = '' # the camera parameters' xml file.

    opts.bbox_thres       = 180 # body.
    # opts.bbox_thres       = 500 # portrait
    opts.NUM_MAX_FRAMES   = 5000
    
    # options.
    opts.is_undistort     = True
    opts.is_vertical      = True
    opts.using_drm        = True
    opts.rot_degree       = 1 # dataset.
    # opts.rot_degree       = 3
    opts.num_distort_params = 8 # online demo, all 8 distortion parameters.
    opts.num_distort_params = 5 # dataset.

    torch.set_num_threads(1)
    
    # multi-processing rendering pipeline here for rendering
    main(opts)