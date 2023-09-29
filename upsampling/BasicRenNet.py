
import torch
import torch.nn as nn
import torch.nn.functional as F

from options.RenTrainOptions import RenTrainOptions
from c_lib.VoxelEncoding.voxel_encoding_helper import OcTreeOptions, sampling_multi_view_feats, integrate_depths, volume_render, volume_render_infer, volume_render_torch
from c_lib.VoxelEncoding.freq_encoding import FreqEncoder
from c_lib.VoxelEncoding.depth_normalization import DepthNormalizer

from SRONet.nerf import TransNerf
from utils_render.utils_render import Density, rgb_normalizer, depth_normalizer, draw_octree_nodes, to_pcd_mesh, generate_rays, proj_persp, z_normalizer
from models.utils import reshape_multiview_tensor, DilateMask, DilateMask2, project3D, index2D
from depth_denoising.net import HRNetUNet, FeatureNet
from models.networks import init_weights
from models.utils import grid_sample
from c_lib.VoxelEncoding.dist import VoxelEncoding # Voxel encoding library.

from implicit_seg.functional import Seg3dTopk, Seg3dLossless

import matplotlib.pyplot as plt

class BasicRenNet(nn.Module):

    def __init__(self, opts, device):
        super(BasicRenNet, self).__init__() # suppose device is a string.
        
        self.opts = opts
        self.device = device
        self._device_Id = int(str(self.device).split(':')[1])

        self.batch_size = opts.batch_size
        self.num_views  = opts.num_views
        self.is_train   = opts.is_train
        self.target_num_views = opts.target_num_views
        
        self.num_sampled_rays = opts.num_sampled_rays * opts.target_num_views # sampling N(e.g., 1024) rays.
        self.num_sampled_rays_pv = opts.num_sampled_rays
        
        # 1. Image Encoder, for RGB, D seperately encoding.
        self.filter_2d = self.get_filter()
        self.filter_2d_high_res = FeatureNet() # the feature extraction network used for high-res image encoding.

        # 2. get the octree options.
        self.octree_optioner = OcTreeOptions(opts.octree_level, opts.batch_size, opts.octree_rate, 
                                             opts.volume_dim, opts.num_max_hits_voxel, device)

        # 3. get the nerf function (sigmas & colors);
        self._nerf_func = TransNerf(opts, device)

        # 4. other properties (freq encoder, depth noramlizer, feature sampler)
        self._depth_normalizer = DepthNormalizer(opts)
        self._sampling_multi_view_feats = sampling_multi_view_feats.to(device)
        self._integrate_depths = integrate_depths.to(device)
        self._volume_render_torch = volume_render_torch.to(device)
        self._dilate_mask_func = DilateMask2(ksize=opts.dilate_ksize).to(device)
        self.project3d         = project3D(opts, proj_type=self.opts.project_mode).to(self.device)
        
        # 5. init variables.
        self.init_octree_variables()
        self._vol_origin = self._voxel_size = self._vol_bbox = None;
        self._total_num_occupied_voxels = self._octree_nodes_index = None;
        self._Ks = self._RTs = self._Rs = self._Ts = None;
        self._ray_ori = self._ray_dir = self._ray_dir_cam = self._ray_dir_cam_ = self._ray_dir_world = None;

        # for sampling points.
        self._num_samplings = self.opts.num_sampled_points_coarse
        self.rays_noise_c = torch.full([self.batch_size, self.num_sampled_rays, self._num_samplings], 0.5, dtype=torch.float32, device=self.device)

        # 6. using real-time pifu during inference.
        if not self.is_train:
            self.b_min = torch.tensor([-1.0, -1.0, -1.0]).float()
            self.b_max = torch.tensor([ 1.0,  1.0,  1.0]).float()
            self._reconEngine = Seg3dLossless(
                                    query_func=self.query_func,
                                    b_min=self.b_min.unsqueeze(0).numpy(),
                                    b_max=self.b_max.unsqueeze(0).numpy(),
                                    resolutions=[16+1, 32+1, 64+1, 128+1],
                                    balance_value=0.5,
                                    use_cuda_impl=True,
                                    faster=True
                                ).to(self.device)
        
    def init_octree_variables(self):
        self._occupied_volumes = [] # save the indexs.
        self._num_occ_voxels   = []
        self._volume_dims      = [self.opts.volume_dim] * 3
        
        for i in range(self.opts.octree_level):
            vol_dim = [self._volume_dims[k] // self.opts.octree_rate**i for k in range(3)]
            o_volume = torch.empty([self.batch_size, *vol_dim], dtype=torch.int32, device=self.device) # index -1 is invalid.
            self._occupied_volumes.append( o_volume )
            self._num_occ_voxels.append( torch.empty([self.batch_size,1], dtype=torch.int32, device=self.device) )

    def parse_input(self, images):
        # parse images into rgb, depth, masks, preprocess the depth maps.
        images = reshape_multiview_tensor(images)[0] # # [B,N,K,H,W] to [B*N, ...]
        # high-res data
        self._rgbs_high_res   = images[:, :3, ...].clone() # [N, 3, H, W]
        self._depths_high_res = images[:, 3:4, ...].clone() # [N, 1, H, W]
        self._masks_high_res  = images[:, -1:, ...].clone() # [N, 1, H, W]
        self._rgbs_high_res_n = rgb_normalizer(self._rgbs_high_res, self._masks_high_res)
        # low-res data.
        self._rgbs   = F.interpolate( self._rgbs_high_res, (self.opts.load_size, self.opts.load_size), mode='bilinear' )
        self._depths = F.interpolate(self._depths_high_res, (self.opts.load_size, self.opts.load_size), mode='nearest')
        self._masks  = F.interpolate(self._masks_high_res, (self.opts.load_size, self.opts.load_size), mode='nearest')

        # normalize the depth maps (CUDA) : [-1, 1] ~ 1.5ms
        self._rgbs_normalized = rgb_normalizer(self._rgbs, self._masks)
        self._depths_normalized, _, self._depth_mid, self._depth_dist = self._depth_normalizer(self._depths) # [-1, 1]
        # dilated mask is used for sampling rays.
        self._masks_dilated   = self._dilate_mask_func(self._masks, iter_num=4)

        self._inputs = torch.cat([self._rgbs_normalized, self._depths_normalized], dim=1) # [B*N, 4(RGBD),H,W]
    
    def parse_KRTs(self, calibs):
        self._Ks  = calibs[0].view(-1, 3, 3) # [B*N, 3, 3]
        self._RTs = calibs[1].view(-1, 3, 4) # [B*N, 3, 4]
        self._Rs  = self._RTs[..., :3].contiguous() # [B*N, 3, 3]
        self._Ts  = self._RTs[..., -1:].contiguous() # [B*N, 3, 1]
        self._CAM = - torch.inverse(self._Rs) @ self._Ts # [B*N, 3, 1]
        self._CAM_ = self._CAM.view(-1, self.num_views, 3) # [B, N, 3]

    def parse_rays(self, batch_rays, rs):
        # the ray ori & dirs are located in the world coordinates.
        self._ray_ori = batch_rays[..., :3].contiguous()
        self._ray_dir = batch_rays[..., 3:].contiguous() # [B, N_rays, 3]
        self._ray_dir_world = self._ray_dir[:,:,None].repeat(1,1,self._num_samplings,1) # [B, N_rays, N_sampled, 3]
        # transform the rays dir to camera's space. dir_local = R @ dir_global. [B*N_v, 3, 3] @ [B*N_v, 3, N]
        ray_dir_expand = self._ray_dir.permute(0, 2, 1)[:,None].repeat(1,self.num_views,1,1)\
                        .view(self.batch_size*self.num_views, 3, -1)
        self._ray_dir_cam_ = (rs @ ray_dir_expand).permute(0,2,1).contiguous() # [B*N_v, N_rays, 3]
        self._ray_dir_cam  = self._ray_dir_cam_.view(self.batch_size, self.num_views, 
                                                     self.num_sampled_rays, 1, 3)\
                                                    .repeat(1,1,1,self._num_samplings,1) # [B, N_v, N_rays, N_sampled, 3]
        
    def get_filter(self):
        # obtain the 2D filter, 2D CNN for images.
        # RGB's UNet's representation.
        unet = HRNetUNet(self.opts, self.device) # using hrnet net as backbone.
        unet.init_weights()
        return unet

    def filter_features_by_images(self, rgbs, depths):
        return self.filter_2d( rgbs, depths )
    
    @torch.no_grad()
    def query_func(self, points):
        '''
            - points: size of (bz, N, 3)
            - proj_matrix: size of (bz, 4, 4)
            return: size of (bz, 1, N)
        '''
        new_points = points.permute(0,2,1)[:, [2,1,0],:]
        
        preds = self.forward_query_occ( new_points )
        return preds

    @torch.no_grad()
    def fusion_occ_volume(self, v0, v1):
        # new volume; 
        fiter_idx = v0 < 0.001;
        min_occ_th = 0.3 if self.opts.rend_full_body else 0.45
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

    def forward_build(self, input_images, calibs):
        # input_images: [B * N_views, 5, H, W](RGB, depths, masks.)
        # the batchified_rays are sampled from the target images' erorded masks: [B, N_rays, 6]
        # preprocessing & get inputs.
        self.parse_input(input_images) # get: RGB, DEPTH, K & RT matrix

        self._geo_features, self._rgb_features = self.filter_features_by_images( self._rgbs_normalized, self._depths_normalized )
        self._rgb_high_res_features = self.filter_2d_high_res( self._rgbs_high_res_n )
        
        # 2. obtain K:[B*N, 3, 3], RTs:[B*N,3,4];
        self.parse_KRTs(calibs)
        self._inputs_data = [self._rgbs, self._depths, self._masks, self._Ks, self._RTs]
        self._input_volumes = [self._occupied_volumes, self._num_occ_voxels]

        # 3. build the tsdf & init the octree.
        if self.opts.test_rays_ui:
            self._occupied_volumes, self._num_occ_voxels, self._tsdf_vol, self._color_vol, self._vol_origin, self._vol_bbox, self._voxel_size = \
                self._integrate_depths(self._inputs_data, self._input_volumes, self.batch_size, 
                                       _level_volumes=self.opts.octree_level, _volume_res_rate=self.opts.octree_rate,
                                       _tsdf_th_low=self.opts.tsdf_th_low, _tsdf_th_high=self.opts.tsdf_th_high, build_octree=False)
        else:
            self._occupied_volumes, self._num_occ_voxels, _, _, self._vol_origin, self._vol_bbox, self._voxel_size = \
                self._integrate_depths(self._inputs_data, self._input_volumes, self.batch_size, 
                                       _level_volumes=self.opts.octree_level, _volume_res_rate=self.opts.octree_rate,
                                       _tsdf_th_low=self.opts.tsdf_th_low, _tsdf_th_high=self.opts.tsdf_th_high, build_octree=False)
        # adopting real-time pifu during inference.
        if (not self.is_train) and self.opts.support_post_fusion:
            new_b_min = self._vol_bbox[:,[2,1,0],0].unsqueeze(1) # [BZ, 1, 3]
            new_b_max = self._vol_bbox[:,[2,1,0],1].unsqueeze(1) # [BZ, 1, 3]
            self._reconEngine.b_min = new_b_min
            self._reconEngine.b_max = new_b_max
            output_volume = self._reconEngine()
            # forward the volume, [B,N,N,N];
            output_volume = F.interpolate( output_volume, (self.opts.volume_dim, self.opts.volume_dim, self.opts.volume_dim) )[0]
            
            # for geometrics.
            # import mcubes
            # verts, faces = mcubes.marching_cubes(output_volume[0].cpu().detach().numpy(), 0.5)
            # mcubes.export_obj(verts, faces, './realtime_pifu.obj')
            
            # post volume-fusion.
            self.fusion_occ_volume(output_volume, self._occupied_volumes[0])
        
        VoxelEncoding.build_multi_res_volumes(self._occupied_volumes, self._num_occ_voxels, 
                                              self.opts.octree_level, self.opts.octree_rate, self._device_Id)
        self._total_num_occupied_voxels, self._octree_nodes_index = \
            self.octree_optioner.init_octree(self._occupied_volumes, self._num_occ_voxels, self._voxel_size, self._vol_bbox)
        torch.cuda.synchronize()

    @torch.no_grad()
    def forward_ray_intersect(self, ray_oris, ray_dirs, num_points, octree_opt):
        B, num_rays = ray_oris.shape[:2]
        # step 1. ray-voxel intersections.
        idx_c, min_depths_c, max_depths_c, ray_total_valid_len  = \
            octree_opt.ray_intersection(ray_oris, ray_dirs, instersect_start_level=self.opts.instersect_start_level,
                                        th_early_stop=self.opts.early_stop_th, only_voxel_traversal=True, is_training=False, 
                                        opt_num_intersected_voxels=-1 if self.is_train and self.opts.phase == 'training' else self.opts.max_intersected_voxels)

        self._idx = idx_c # randomly sampling coarse points.
        if self.is_train and self.opts.phase == 'training': # when training, randomly sample points on rays.
            rays_noise_c = torch.rand([B, num_rays, num_points], dtype=torch.float32, device=ray_oris.device)
        else:
            rays_noise_c = self.rays_noise_c.clone()
        
        # step 2. uniformly sampling points in the octree two layers' voxels.
        sampled_idx_c, sampled_depths_c = \
            octree_opt.ray_voxels_points_sampling_coarse(idx_c, min_depths_c, max_depths_c, rays_noise_c, ray_total_valid_len, 
                                                         self.opts.instersect_start_level, num_points)
        sorted_depths, sampled_sort_idxs = octree_opt.sort_sampling(sampled_depths_c) # [B, N_rays, N_samples]
        
        if self.opts.test_rays_ui and str(self.device) == 'cuda:0': # when gpu_id==0, apply the gui.
            self.forward_test_rays_ui(sampled_depths_c, sampled_idx_c) # testing rays and voxels.
        
        return sampled_idx_c, sampled_depths_c, sorted_depths, sampled_sort_idxs


    def forward_sample_feature(self, feats_geo, depths, feats_rgb, rgbs_n, xy, z):
        sampled_geo_feats = F.grid_sample(feats_geo, xy, align_corners=True, mode='bilinear')[..., 0] # [B*N, C, N_points.]
        sampled_geo_feats = sampled_geo_feats.permute(0,2,1).contiguous() # geo_feats;

        sampled_depths    = F.grid_sample(depths, xy, align_corners=True, mode='bilinear')[..., 0] # [B*N, 1, N_points.]
        t_psdf = torch.clamp( z - sampled_depths, -0.01, 0.01 ) * (1.0 / 0.01);
        t_psdf = t_psdf.permute(0,2,1).contiguous() # t_psdf;

        sampled_rgb_feats = F.grid_sample( feats_rgb, xy, align_corners=True, mode='bilinear' )[..., 0] # [B*N, C, N_points.]
        sampled_rgb_feats = sampled_rgb_feats.permute(0,2,1).contiguous() # geo_feats;

        sampled_rgbs    = F.grid_sample( rgbs_n, xy, align_corners=True, mode='bilinear' )[..., 0]
        sampled_rgbs    = sampled_rgbs.permute(0,2,1).contiguous()

        return sampled_geo_feats, sampled_rgb_feats, t_psdf, sampled_rgbs
    
    def transform_dist_to_depths(self, ray_dirs, num_rays, dists, target_calibs):
        rs = target_calibs[1][..., :3].contiguous().view(-1, 3, 3) # [B*N_t, 3, 3];
        dirs = ray_dirs.view(self.batch_size, -1, num_rays, 3).view(-1, num_rays, 3).permute(0,2,1) # [B*Nt, 3, N_rays]
        dirs_cam = (rs @ dirs).permute(0,2,1).reshape(self.batch_size, -1, num_rays, 3).reshape(self.batch_size, -1, 3)
        scale_z = dirs_cam[..., -1] # [B, N_rays]
        # cos_theta = z;, [x,y,z] \cdot [0,0,1] = z
        self._scale_z = scale_z.clone()
        return dists * scale_z[..., None] # [B, N_rays, N_p]
    
    def forward_ray_nerf(self, ray_oris, ray_dirs, num_points, target_calibs):
        # Given the ray ori and dirs, return the output depth & features
        # 1. ray intersect with voxels, rays: [B, N_rays, 3] (ori and dirs.) -> [B, N_rays, N_points]
        ray_oris = ray_oris.view(self.batch_size, -1, 3);
        ray_dirs = ray_dirs.view(self.batch_size, -1, 3);

        # noted that this depth is the z axis's length, transform dists to depths.
        _, _, sorted_dists, _ = self.forward_ray_intersect(ray_oris, ray_dirs, num_points, self.octree_optioner)
        sorted_depths = self.transform_dist_to_depths( self._ray_dir, self.num_sampled_rays_pv, sorted_dists, target_calibs )
        
        if not (self.is_train and self.opts.phase == 'training'):
            o_c_feats = torch.full( [self.batch_size, self.num_sampled_rays, 30], 0, device=sorted_depths.device, dtype=torch.float32 )
            o_c = torch.full( [self.batch_size, self.num_sampled_rays, 3], 0, device=sorted_depths.device, dtype=torch.float32 )
            o_d = torch.full( [self.batch_size, self.num_sampled_rays, 1], 0, device=sorted_depths.device, dtype=torch.float32 )

            valid_rays = torch.prod(sorted_dists != -1, dim=-1) # [B, N_rays], all rays are valid for training.
            valid_rays_idx = torch.where( valid_rays[0] == True )[0]
            if valid_rays_idx.shape[0] == 0: # no valid rays;
                return o_c_feats, o_c, o_d

            sorted_depths = sorted_depths[:, valid_rays_idx]; sorted_dists  = sorted_dists[:, valid_rays_idx]
            rays_points  = ray_oris[:,valid_rays_idx][:,:,None] + ray_dirs[:,valid_rays_idx][:,:,None] * sorted_dists[...,None] # [1, N_rays', N_p, 3]
            ray_dir_cam  = self._ray_dir_cam[:, :, valid_rays_idx]
        else:
            valid_rays = None
            rays_points = ray_oris[:,:,None] + ray_dirs[:,:,None] * sorted_dists[...,None] # [B, N_rays, N_p, 3]
            ray_dir_cam = self._ray_dir_cam.clone()

        _, proj_xy, proj_z = proj_persp(rays_points.view(self.batch_size, -1, 3).permute(0,2,1), 
                                              self._Ks, self._Rs, self._Ts, 
                                              sorted_depths.view(self.batch_size,-1,1).permute(0,2,1), self.num_views)

        proj_xy = proj_xy.permute(0,2,1).view( proj_z.shape[0], -1, 1, 2 );  # [B*Nv, N_p, 2 and 1] 
        proj_z = proj_z.permute(0,2,1)
        proj_z_normalized = z_normalizer( proj_z, self._depth_mid, self._depth_dist ) # [B*Nv, Np, 1]
        proj_z_normalized = proj_z_normalized.view(self.batch_size, self.num_views, -1, self._num_samplings, 1)
        proj_z = proj_z.view( proj_z.shape[0], 1, -1 )

        # 2. sampling features and pass MLPs.
        sampled_geo_feats, sampled_rgb_feats, t_psdf, sampled_rgbs  \
            = self.forward_sample_feature( self._geo_features, self._depths, self._rgb_features, self._rgbs_normalized, proj_xy, proj_z )
    
        occ, geo_feat   = self._nerf_func.forward_density( proj_z_normalized, t_psdf, sampled_geo_feats )
        rgbs, rgbs_feat = self._nerf_func.forward_color( sampled_rgb_feats, sampled_rgbs, geo_feat, ray_dir_cam)
        
        # 3. rendering.
        output_rgbs_feats, output_d, _, _ \
            = self._volume_render_torch.forward_occ(sorted_depths, occ, torch.cat([rgbs, rgbs_feat], dim=-1), 
                                                    self.opts.valid_depth_max)

        output_rgbs = output_rgbs_feats[...,:3]; output_rgbs_feat = output_rgbs_feats[...,3:]
        
        if not (self.is_train and self.opts.phase == 'training'):
            o_c_feats[:, valid_rays_idx] = output_rgbs_feat.clone()
            o_c[:, valid_rays_idx]       = output_rgbs.clone()
            o_d[:, valid_rays_idx]       = output_d.clone()
            
            return o_c_feats, o_c, o_d
        
        return output_rgbs_feat, output_rgbs, output_d
   
    def forward_query(self, batch_rays, target_calibs):
        # 1. get the num sampled rays.
        if not self.is_train or (self.is_train and self.opts.phase == 'validating'):
            self.num_sampled_rays = batch_rays.shape[1] # [B, N_rays, 6]
            self.num_sampled_rays_pv = self.num_sampled_rays // 1; # only one target view.
            
        self.parse_rays(batch_rays, self._Rs) # get rays' ori, dirs.
        
        # 2. pass the ray-nerf.
        return self.forward_ray_nerf(self._ray_ori, self._ray_dir, self._num_samplings, target_calibs)

    def rays_interpolate(self, ray_dirs, num_rays, t_ks, t_rs):
        ks = t_ks.view(-1,3,3); rs = t_rs.view(-1,3,3);
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
        dirs_ep_n = F.normalize( dirs_ep, dim=-1 ).view(self.batch_size, -1, *dirs_ep.shape[-2:]).view(self.batch_size,-1,3) # [B, Nv*Nr*4, 3]
        
        return dirs_ep_n

    def forward_upsampling_query(self, results_nerf, neighbor_views, target_calibs):
        o_c_feats, o_c, o_d = results_nerf

        if not (self.is_train and self.opts.phase == 'training'):
            o_c_detail = torch.full( [self.batch_size, self.num_sampled_rays, 4, 3], 0, device=o_c.device, dtype=torch.float32 )
            valid_rays = o_d != -1
            valid_rays_idx = torch.where( valid_rays[0,:,0] == True )[0]
            if valid_rays_idx.shape[0] == 0: # no valid rays;
                return o_c_detail.view(self.batch_size,-1,3), o_d, None
            
            num_sampled_rays = valid_rays_idx.numel()
            ray_ori  = self._ray_ori[:, valid_rays_idx]; ray_dir = self._ray_dir[:, valid_rays_idx]
            ray_rgbs = o_c[:, valid_rays_idx]; output_d = o_d[:, valid_rays_idx];
            scale_z  = self._scale_z[:, valid_rays_idx]; o_c_feats_ = o_c_feats[:, valid_rays_idx]
        else:
            ray_dir = self._ray_dir; num_sampled_rays = self.num_sampled_rays_pv
            ray_ori = self._ray_ori; output_d = o_d; scale_z = self._scale_z
            o_c_feats_ = o_c_feats; ray_rgbs = o_c;

        rays_dirs_ep = self.rays_interpolate( ray_dir, num_sampled_rays, target_calibs[0], target_calibs[1][..., :3].contiguous() )
        ray_ori  = ray_ori[:,:,None].repeat(1,1,4,1).view(self.batch_size,-1,3)
        output_d = output_d[:,:,None].repeat(1,1,4,1).view(self.batch_size,-1,1) # [B, 4*N_rays, 1]
        scale_z  = scale_z[:,:,None].repeat(1,1,4).view(self.batch_size,-1,1)
        ray_rgbs = ray_rgbs[:,:,None].repeat(1,1,4,1).view(self.batch_size, -1,3) # coarse rgb values;
        o_c_feats_ = o_c_feats_[:,:,None].repeat(1,1,4,1).view(self.batch_size, -1, o_c_feats_.shape[-1]) # shared four features;

        # get surface points.
        ray_dists  = output_d / scale_z
        surface_pos = ray_ori  + rays_dirs_ep * ray_dists
        _, proj_xy, proj_z = proj_persp(surface_pos.permute(0,2,1), self._Ks, self._Rs, self._Ts, output_d.permute(0,2,1), self.num_views)
        proj_xy = proj_xy.permute(0,2,1).view( self.batch_size, self.num_views, -1, 4, 2 );
        proj_z  = proj_z.permute(0,2,1).view( self.batch_size, self.num_views, -1, 4, 1 ); # [B, N_rays, 2 , 1]

        rgbs_details, weights = self._sampling_multi_view_feats.sample_neighbor( 
            self._rgb_high_res_features, self._rgbs_high_res, ray_rgbs, o_c_feats_, self._depths, neighbor_views,
            rays_dirs_ep, self._Rs, proj_xy, proj_z, self._nerf_func
        )

        rgbs_details = rgbs_details.view( self.batch_size, -1, 3 )
        weight_c     = weights.view( self.batch_size, -1, 1 )

        if not (self.is_train and self.opts.phase == 'training'):
            o_c_detail[:, valid_rays_idx] = rgbs_details.view(self.batch_size, -1, 4, 3).clone()
            return o_c_detail.view( self.batch_size, -1, 3 ), o_d, weight_c

        return rgbs_details, o_d, weight_c
    
    def forward_query_occ(self, sampled_points):
        sampled_points.requires_grad_(True)
        # self._proj_xy_sp, self._proj_z_sp = self.project3d.project(sampled_points, [self._Ks, self._RTs])
        _, proj_xy, proj_z = proj_persp( sampled_points, self._Ks, self._Rs, self._Ts, None, self.num_views )
        proj_xy = proj_xy.permute(0,2,1).view( proj_z.shape[0], -1, 1, 2 );  # [B*Nv, N_p, 2 and 1]
        # sample geo-features, and diff-z.
        if self.opts.lam_reg == 0 or (not self.is_train):
            sampled_geo_feats = F.grid_sample(self._geo_features, proj_xy, align_corners=True, mode='bilinear')[..., 0] # [B*N, C, N_points.]
            sampled_depths    = F.grid_sample(self._depths, proj_xy, align_corners=True, mode='bilinear')[..., 0] # [B*N, 1, N_points.]
        else: # using gradient.
            sampled_geo_feats = grid_sample(self._geo_features, proj_xy)[..., 0]
            sampled_depths    = grid_sample(self._depths, proj_xy)[..., 0]

        sampled_geo_feats = sampled_geo_feats.permute(0,2,1).contiguous()
        t_psdf = torch.clamp( proj_z - sampled_depths, -0.01, 0.01 ) * (1.0 / 0.01);
        t_psdf = t_psdf.permute(0,2,1).contiguous() # t_psdf;
                
        ####### get the mask labels of occ points ######
        in_img_flag = None
        if not self.is_train: 
            # xy: [B, N_v, N_rays, N_sampled, 2]
            in_img_flag  = (proj_xy[..., 0] >= -1.0) & (proj_xy[..., 0] <= 1.0) & (proj_xy[..., 1] >= -1.0) & (proj_xy[..., 1] <= 1.0)
            # the outside points are excluded, we need to smooth the boundaies's points.
            mask_value   = self._sampling_multi_view_feats.get_mask_labels(self._masks_dilated, proj_xy, self.num_views) # [B * N_v, 1, N_p].
            in_img_flag = in_img_flag.view(-1, 1) & (mask_value.view(-1, 1) > 0) # [B * N_v * N_p, 1]
            
            if self.num_views > 1:
                in_img_fusion = in_img_flag.view(self.batch_size, self.num_views, -1, 1) # [B, N_v, N_points]
                in_img_flag   = torch.prod(in_img_fusion, dim=1) # [B, N_points, 1]
                
            in_img_flag = in_img_flag.view(-1, 1) # [B * N_p, 1]
        #################################################
        
        proj_z_normalized = z_normalizer( proj_z.permute(0,2,1), self._depth_mid, self._depth_dist ).view(-1,1) # [B*Nv, Np, 1]
        occ = self._nerf_func.forward_density_( proj_z_normalized, t_psdf, sampled_geo_feats, in_img_flag )
        if self.is_train and self.opts.phase == 'training' and self.opts.lam_reg != 0:
            occ_grads = self.gradient( sampled_points, occ ).permute(0,2,1) # [B, 3, n] and [B, 1, n] -> [B, n, 3]
            return occ, occ_grads

        return occ

    def forward_test_rays_ui(self, sampled_depths, sampled_idx):
        # transform to the global point clouds.
        to_pcd_mesh(self._tsdf_vol, self._color_vol, self._vol_bbox, self._voxel_size)
        # draw all the points & voxels of the tree.
        draw_octree_nodes(self.octree_optioner.get_all_nodes(), self._octree_nodes_index, 
                          self._ray_ori, self._ray_dir, sampled_depths,
                          self._idx, sampled_idx)
        return
    
    def gradient(self, p, occ, no_grad=False):
        # p: positions, occ: predicted occupancy.
        # y = self.infer_occ(p)[...,:1], no_grad: if True, no create graph.
        d_output = torch.ones_like(occ, requires_grad=False, device=occ.device)
        gradients = torch.autograd.grad(
            outputs=occ,
            inputs=p,
            grad_outputs=d_output,
            create_graph=True if not no_grad else False, # if nograd, don't create graph here.
            # retain_graph=False, # the retain_graph is set same to the create_graph.
            only_inputs=True, 
            allow_unused=True)[0]

        return gradients

    def forward(self, input_images, calibs, neighbor_views, batch_rays=None, target_calibs=None, target_masks=None):
        batch_rays_idx = None
        # self.batch_rays = batch_rays
        # 1. build the octree options & imgs_features.
        self.forward_build(input_images, calibs)

        if batch_rays is None: # no given rays here.
            # 2. sample rays (from target masks, when training.) : [B, N_rays, 6(ori,dir)], [B, N_rays, 2(x,y)]
            batch_rays, batch_rays_idx = self.gen_rays(target_calibs, target_masks=target_masks)
            
        # 3. ray voxel intersection & pass nerf func & volume rendering.
        results_nerf = self.forward_query(batch_rays, target_calibs)
        results_nerf = self.forward_upsampling_query(results_nerf, neighbor_views, target_calibs)
    
        return results_nerf, batch_rays_idx

    @torch.no_grad()
    def gen_rays(self, target_calibs, target_masks=None):
        dilated_masks = None
        if target_masks is not None:
            target_masks = reshape_multiview_tensor(target_masks)[0] # [B*N_v, 1,h,w]
            # dialte the masks for some extent (when training)
            dilated_masks = self._dilate_mask_func(target_masks) # [B*N_v, 1, H, W]
        
        # [B, N_v, 3,3] and [B, N_v, 3,4]; The num of rays is not sure.
        ks, rts = target_calibs[0].view(self.batch_size, -1, 3, 3), target_calibs[1].view(self.batch_size, -1, 3, 4);
        rs = rts[..., :3].contiguous()   # [B, N_v, 3, 3]
        ts = rts[..., -1:].contiguous()  # [B, N_v, 3, 1]

        # generate rays in target view, when validating or rendering, target view : [1, H*W, 6];
        batch_rays, batch_rays_idx = generate_rays(ks, rs, ts, self.opts.load_size, masks=dilated_masks, body_bbox=self._vol_origin,
                                                   num_sampled_rays=self.opts.num_sampled_rays if self.is_train and self.opts.phase == 'training' else None,
                                                   in_patch=self.opts.ray_patch_sampling, in_bbox=self.opts.ray_bbox_sampling, 
                                                   device=self.device)

        return batch_rays, batch_rays_idx


if __name__ == '__main__':
    from options.RenTrainOptions import RenTrainOptions
    
    opts = RenTrainOptions().parse()
    device = 'cuda:0'
    
    render_net = BasicRenNet(opts, device)
    print(render_net)