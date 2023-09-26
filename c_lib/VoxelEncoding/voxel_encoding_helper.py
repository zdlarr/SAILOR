
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from c_lib.VoxelEncoding.dist import VoxelEncoding # Voxel encoding library.
from models.utils import grid_sample
import torch.nn.functional as F

MAX_VOL_DIST = 999999

class _voxel_tri_interpolation(Function):
    
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, octree, ray_ori, ray_dir, sampled_depths, sampled_idx, sampled_feats):
        # sampled_feats: [B, C, N_ray, N_sampled, 8]
        # batch_size, dim_feats, num_rays, num_sampled, _ = output_feats.shape
        assert (not ray_ori.requires_grad) and (not ray_dir.requires_grad) \
               and (not sampled_depths.requires_grad) and (not sampled_idx.requires_grad), "Not supported"
        ctx.octree = octree
        ctx.ray_ori = ray_ori
        ctx.ray_dir = ray_dir
        ctx.sampled_depths = sampled_depths
        ctx.sampled_idx = sampled_idx
        ctx.save_for_backward(sampled_feats)

        output_feats = octree.trilinear_aggregate_features(
            ray_ori, ray_dir,
            sampled_depths, sampled_idx,
            sampled_feats
        )
        # we only output the sampled features.
        return output_feats

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output_feats):
        # return the gradients of inputs of the forward.
        octree = ctx.octree
        ray_ori = ctx.ray_ori
        ray_dir = ctx.ray_dir
        sampled_depths = ctx.sampled_depths
        sampled_idx = ctx.sampled_idx
        sampled_feats, = ctx.saved_tensors
        
        grad_data = octree.trilinear_aggregate_features_backward(
            ray_ori, ray_dir,
            sampled_depths, sampled_idx,
            sampled_feats, grad_output_feats
        )
        # grad_data = torch.zeros_like(sampled_feats)
        if not ctx.needs_input_grad[5]:
            grad_data = None

        return None, None, None, None, None, grad_data
    

voxel_trilinear_interpolation = _voxel_tri_interpolation.apply


class _volume_rendering_inference(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, depths, sigmas, feats, t_thres=1e-4):
        device = int(str(depths.device).split(':')[-1])

        output_feats, output_depths, ws, output_alphas = VoxelEncoding.volume_rendering_occ_forward(
            depths, sigmas, feats, device, t_thres
        );

        return output_feats, output_depths, ws, output_alphas

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output_feats, grad_output_depths, grad_ws, grad_alphas):
        return None, None, None, None

volume_render_infer = _volume_rendering_inference.apply


class _volume_render_training(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, depths, depth_sort_idxs, sigmas, feats, t_thres=1e-4, using_occ=True):
        device = int(str(depths.device).split(':')[-1])
        
        ctx.depths = depths
        ctx.depth_sort_idxs = depth_sort_idxs
        ctx.t_thres = t_thres
        ctx.using_occ = using_occ
        
        output_feats, output_depths, ws, output_alphas = VoxelEncoding.volume_rendering_training_forward(
            depths, depth_sort_idxs,
            sigmas, feats, device, t_thres, using_occ
        );
        ctx.save_for_backward(sigmas, feats, output_feats, output_depths, ws, output_alphas)

        return output_feats, output_depths, ws, output_alphas

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output_feats, grad_output_depths, grad_ws, grad_alphas):
        depths = ctx.depths
        depth_sort_idxs = ctx.depth_sort_idxs
        t_thres = ctx.t_thres
        using_occ = ctx.using_occ
        sigmas, feats, output_feats, output_depths, ws, output_alphas = ctx.saved_tensors

        device = int(str(depths.device).split(':')[-1])
        
        grad_feats, grad_sigmas = VoxelEncoding.volume_rendering_training_backward(
            grad_output_feats, grad_output_depths, grad_ws, grad_alphas,
            output_feats, output_depths, ws, output_alphas,
            depths, depth_sort_idxs, sigmas, feats, device, t_thres, using_occ
        );

        if not ctx.needs_input_grad[2]:
            grad_sigmas = None
        if not ctx.needs_input_grad[3]:
            grad_feats = None

        return None, None, grad_sigmas, grad_feats, None, None
    
volume_render = _volume_render_training.apply


class _volume_render_torch(nn.Module):
    
    def forward(self, sampled_depths, sigma, feats, max_depths, valid_rays=None):
        # sampled_depths : [B, N_rays, N_p], sigma : [B, N_rays, N_p, 1], geo_feats: [B, N_rays, N_p, 64], valid_rays: [B, N_rays]
        # print(sampled_depths.shape, sigma.shape, feats.shape)
        # print(sampled_depths[0,100].cpu())
        
        batch_size, num_rays, num_sampled  = sampled_depths.shape
        
        _sampled_depths = sampled_depths.view(-1, num_sampled) # [B_size * Num_rays, N_sampled]
        density = sigma.view(-1, num_sampled)  # [batch_size * Num_rays,  N_samples]

        dists = _sampled_depths[:, 1:] - _sampled_depths[:, :-1] # [batch_size * Num_rays,  N_samples-1]
        # dists = torch.cat( [dists, torch.full([dists.shape[0], 1], max_depths, device=dists.device) - _sampled_depths[:, -1:]], -1)
        dists = torch.cat( [dists, torch.full([dists.shape[0], 1], 0.0, device=dists.device)], -1) # the last distance is 0;
    
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.full([dists.shape[0], 1], 0.0, device=dists.device), free_energy], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance[:, :-1] # probability of the ray hits something here.
        weights = weights.view(batch_size, num_rays, num_sampled, 1)
        
        depths = torch.sum(weights * sampled_depths[..., None], -2)
        alphas = torch.sum(weights * alpha.view(batch_size, num_rays, num_sampled, 1), -2)
        feats  = torch.sum(weights * feats, -2)
        ws     = torch.sum(weights, -2)

        if valid_rays is not None: # the invalid rays should be supervised ? 
            invalid_rays = (1 - valid_rays[..., None].int()).bool()
            depths[invalid_rays] = 0; feats[invalid_rays.repeat(1,1,feats.shape[-1])] = 0; 
            alphas[invalid_rays] = 0; ws[invalid_rays] = 0;
        
        # print(feats.shape, depths.shape, alphas.shape, ws.shape)
        return feats, depths, ws, alphas

    def forward_occ(self, sampled_depths, sigma, feats, max_depths, valid_rays=None):
        batch_size, num_rays, num_sampled  = sampled_depths.shape
        
        # _sampled_depths = sampled_depths.view(-1, num_sampled) # [B_size * Num_rays, N_sampled]
        alpha = sigma.view(-1, num_sampled) # [batch_size * Num_rays,  N_samples]

        epsilon = 1e-6
        weights = alpha * torch.cumprod(torch.cat( [ torch.ones((alpha.shape[0], 1), device=alpha.device), 1.-alpha + epsilon ], -1 ), -1)[:, :-1]
        weights = weights.view(batch_size, num_rays, num_sampled, 1)
        
        depths = torch.sum(weights * sampled_depths[..., None], -2)
        alphas = torch.sum(weights * alpha.view(batch_size, num_rays, num_sampled, 1), -2)
        feats  = torch.sum(weights * feats, -2)
        ws     = torch.sum(weights, -2)

        if valid_rays is not None:
            invalid_rays = (1 - valid_rays[..., None].int()).bool()
            depths[invalid_rays] = 0; feats[invalid_rays.repeat(1,1,feats.shape[-1])] = 0; 
            alphas[invalid_rays] = 0; ws[invalid_rays] = 0;
        
        return feats, depths, ws, alphas
        

volume_render_torch = _volume_render_torch()


class _sampling_rays(Function):
    
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx,  ):
        pass

    @staticmethod
    @custom_bwd
    def backward(ctx, ):
        pass



class _integrate_depths(nn.Module):
    """ 
    Given the octree instance, initialize the octree nodes and build the multi-level searching table.
    """

    def preprocess_data(self, batch_size, inputs_data):
        colors, depths, masks, Ks, RTs = inputs_data
        # reshape the tensor.
        h, w = depths.shape[-2:]
        num_views = depths.shape[0] // batch_size
        colors = torch.floor(colors[:, 2]*256.0*256.0 + colors[:, 1]*256.0 + colors[:, 0])

        if len(depths.shape) == 4:
            depths = depths[:,0] # [B*N, H, W]
        if len(masks.shape) == 4:
            masks = masks[:,0] # [B*N, H, W]

        depths = depths.view(-1, num_views, h, w).contiguous() # [B, N, H, W];
        colors = colors.view(-1, num_views, h, w).contiguous() # [B, N, H, W];
        masks  = masks.view(-1, num_views, h, w).contiguous().int() # [B, N, H, W];
        Ks     = Ks.view(-1, num_views, 3, 3).contiguous() # [B, N, 3, 3];
        RTs    = RTs.view(-1, num_views, 3, 4).contiguous() # [B, N, 3, 4];
        Rs     = RTs[..., :3, :3].contiguous() # [B, N, 3, 3];
        Ts     = RTs[..., :3, -1].contiguous() # [B, N, 3];

        return colors, depths, masks, Ks, RTs, Rs, Ts
    
    def clear_vol_buf(self, volumes, _level_volumes):
        _occupied_volumes, _num_occ_voxels = volumes
        # clear the buffer. assign -1 and 0;
        for i in range(_level_volumes):
            _occupied_volumes[i].fill_(-1);
            _num_occ_voxels[i].fill_(0);
        
        # _num_corner_points.fill_(0); # the corner points : [B, 1];

        return _occupied_volumes, _num_occ_voxels

    @torch.no_grad()
    def get_origin_bbox(self, inputs_data, batch_size=1, _th_bbox=0.025):
        _, depths, _, Ks, _, Rs, Ts = self.preprocess_data(batch_size, inputs_data)
        gpu_id = int(str(depths.device).split(':')[-1])

        _vol_origin = VoxelEncoding.get_origin_bbox(
            depths, Ks, Rs, Ts, _th_bbox, gpu_id
        )

        return _vol_origin
    
    @torch.no_grad()
    def get_target_center(self, inputs_data, batch_size=1, _th_boox=0.025):
        _, depths, _, Ks, _, Rs, Ts = self.preprocess_data(batch_size, inputs_data)
        gpu_id = int(str(depths.device).split(':')[-1])
        # the center position of the points. 
        _vol_center, _vol_num_pts = VoxelEncoding.get_center_xyz(
            depths, Ks, Rs, Ts, _th_boox, gpu_id
        )
        
        return _vol_center / _vol_num_pts
    
    @torch.no_grad()
    def forward(self, inputs_data, volumes,
                batch_size=1, _obs_weight=1.0, _th_bbox=0.025, _tsdf_th_low=8.0, _tsdf_th_high=8.0,
                _level_volumes=6, _volume_res_rate=2, build_octree=True):
        
        colors, depths, masks, Ks, RTs, Rs, Ts = self.preprocess_data(batch_size, inputs_data)
        # clear the volume buffer.
        _occupied_volumes, _num_occ_voxels = self.clear_vol_buf(volumes, _level_volumes)
        gpu_id = int(str(colors.device).split(':')[-1])
        
        # step 1. get the tsdf-volumes (or basic volume).
        tsdf_vol, _, color_vol, _vol_origin, vol_bbox, voxel_size = \
            VoxelEncoding.integrate(_occupied_volumes[0], _num_occ_voxels[0], 
                                    Ks, RTs, Rs, Ts,
                                    colors, depths, masks,
                                    _obs_weight, _th_bbox, 3.0, # 5 / 4
                                    _tsdf_th_low, _tsdf_th_high,
                                    gpu_id)
        
        # step 2. build the multi-scale volumes
        if build_octree:
            VoxelEncoding.build_multi_res_volumes(_occupied_volumes, _num_occ_voxels, 
                                                _level_volumes, 
                                                _volume_res_rate, gpu_id)
        # print(_num_occ_voxels, len(_occupied_volumes), _occupied_volumes[0].shape, _occupied_volumes[1].shape, _volume_res_rate, _level_volumes)
        
        return _occupied_volumes, _num_occ_voxels, tsdf_vol, color_vol, _vol_origin, vol_bbox, voxel_size

integrate_depths = _integrate_depths()


class _sampling_multi_view_feats(nn.Module):
    """
        Fuse multi-view's sampled features.
    """

    def calculate_relative_depths(self, sampled_depths, z, sigmoid_coef=50.0):
        # clamp in [-1,1]
        # diff_z = z - sampled_depths
        # return 2.0 / (1.0 + torch.exp(-1.0 * sigmoid_coef * diff_z)) - 1.0
        diff_z = (z - sampled_depths) ** 2
        return torch.exp(-1.0 * sigmoid_coef * diff_z)

    def calculate_depth_offset(self, sampled_depths, z, sigmoid_coef=50.0):
        # clamp in [-1, 1];
        diff_z = z - sampled_depths
        return 2.0 / (1.0 + torch.exp(-1.0 * sigmoid_coef * diff_z)) - 1.0

    def calculate_t_psdf(self, sampled_depths, z, truncated_val=0.01):
        # clamp to [-1, 1], first clip to [-val, val], then divide by val, clip to [-1,1], the -1 and 1 provided invalid signals.
        return torch.clamp(z - sampled_depths, -truncated_val, truncated_val) * (1.0 / truncated_val)
    
    # def sample_rgbs(self, rgbs, depths, proj_xy, proj_z, num_views):
    #     # rgbs: [B*N_views, C, H, W]; depths: [B*N_views, 1, H, W];
    #     # xy: [B, N_view, N_ray, N_sampled, 2]; z : [B, N_view, N_ray, N_sampled, 1]
    
    def get_mask_labels(self, dilated_masks, proj_xy, num_views):
        _batch_size = dilated_masks.shape[0] // num_views
        _proj_xy = proj_xy.view(num_views * _batch_size, -1, 1, 2) # [B, N_points, 1, 2]
        mask_value  = F.grid_sample(dilated_masks, _proj_xy, align_corners=False, mode='nearest')[..., 0]
        return mask_value
    
    def _sampling_multi_scale_features(self, fts, xy, user_grid_sample):
        # TODO: sampling multi-scale features.
        sampled_fts = []
        for ft2d in fts:
            if user_grid_sample:
                sampled_feats_rgbds = grid_sample(ft2d, xy)[..., 0]
            else:
                sampled_feats_rgbds = F.grid_sample(ft2d, xy, align_corners=True, mode='bilinear')[..., 0] # [B*N, C, N_points.]
                
            sampled_fts.append(sampled_feats_rgbds)
        
        return torch.cat(sampled_fts, dim=1) # [B *N, sum_{C}, N_P]

    def forward(self, rgbd_feats, rgbs, depths, proj_xy, proj_z, num_views, support_tpsdf=True, t_psdf=False, using_nvsnerf=False, using_transnerf=False, user_grid_sample=False):
        '''
        rgb_feats: [B * N_views, C, H, W], d_feats: [B * N_views, C', H', W'];
        depths: [B * N_views, 1, H, W]; batch_size = shape[0];
        xy: [B, N_view, N_ray, N_sampled, 9, 2]
        z : [B, N_view, N_ray, N_sampled, 9, 1]
        t_psdf: whether using t_psdf method or relative depths;
        return: sampled rgbd features;
        '''
        _batch_size = rgbd_feats.shape[0] // num_views
        _proj_xy = proj_xy.view(num_views * _batch_size, -1, 1, 2) # [B, N_points, 1, 2]
        # feature sampling : [B*N_view, C, N_points];
        sampled_feats_rgbds = self._sampling_multi_scale_features([rgbd_feats], _proj_xy, user_grid_sample)
        
        sampled_rgbs  = None
        if using_nvsnerf or using_transnerf: # [B*N_view, 3, N_points]
            if user_grid_sample:
                sampled_rgbs  = grid_sample(rgbs, _proj_xy)[..., 0]
            else:
                sampled_rgbs  = F.grid_sample(rgbs, _proj_xy, align_corners=True, mode='bilinear')[..., 0]
        
        diffz = None
        if support_tpsdf:
            # depths sampling.
            if len(depths.shape) == 3:
                depths = depths.view(-1, 1, depths.shape[-2], depths.shape[-1]).contiguous()

            _proj_z = proj_z.view(num_views * _batch_size, 1, -1) # [B*N_views, 1, N_points.]
            
            if user_grid_sample:
                sampled_depths = grid_sample(depths, _proj_xy)[..., 0] # [B*N, 1, N_points.]
            else:
                sampled_depths = F.grid_sample(depths, _proj_xy, align_corners=True, mode='bilinear')[..., 0] # [B*N, 1, N_points.]

            # calculate the truncated psdf value : clip(z-d, -0.01, 0.01);
            if t_psdf: # truncated value : 1cm; [-1,1]
                diffz = self.calculate_t_psdf(sampled_depths, _proj_z, truncated_val=0.01)
            else: # a continious representation. incase the nan results.
                # diffz = self.calculate_depth_offset(sampled_depths, _proj_z)
                diffz = self.calculate_relative_depths(sampled_depths, _proj_z, sigmoid_coef=200.0)
            
            # considering the color degrading, weighted by alpha.
            if using_nvsnerf or using_transnerf: # weighted the rgb values, the color will decrease when distance is large.
                # weight = 1 - torch.abs(diffz)
                # weight[weight==0] = 0.01
                # sampled_rgbs *= weight # [B*N_view, 3, N_points] * [B*N_v, 1, N_points], weighted by distance.
                sampled_rgbs = sampled_rgbs.permute(0,2,1).contiguous() # to [B*N_views, N_points, 3];

            # the z diff values : z and d.
            diffz = diffz.permute(0,2,1).contiguous() # [B*N_v, N_sampled*N_rays, 1]
        
        # [B*N_v, N_sampled*N_rays, 3] [B*N_v, N_sampled*N_rays, N_feats0, N_feats1]; [B*N_v, N_sampled*N_rays, 1]
        torch.cuda.synchronize()
        
        return sampled_rgbs, \
               sampled_feats_rgbds.permute(0,2,1).contiguous(), \
               diffz

    def sample_neighbor(self, rgbs_feats, rgbs, ray_c_rgbs, ray_rgb_feats, depths, neighbor_views, dirs, Rs, proj_xy, proj_z, nerf_func):
        # rays: [B, N_rays*4, 1]
        # rgbds_feats: [B*N_v, C, h*w]
        # depths: [B*N_v, 1, h, w]
        # xy, z : batch_size, num_views, num_rays, num_sampled_z, 2 or 1
        # neighbor_views : [B, N_t, 2]
        # K & RTs: [B*N_v, 3, 3]; [B*N_v, 3, 4]
        
        B, N_t, N_nei = neighbor_views.shape # [B, N_t, 2(N_neighbor_views)]
        n_views, N_rays = proj_xy.shape[1:3] # [B, N_views, N_rays, 4, 2 or 1]
        N_rays_pv = N_rays // N_t

        feats = []; weights_c = [];
        
        rgbs_feats = rgbs_feats.view(-1, n_views, *rgbs_feats.shape[-3:])
        depths_     = depths.view(-1, n_views, *depths.shape[-3:])
        rgbs_       = rgbs.view(-1, n_views, *rgbs.shape[-3:])
        dirs_       = dirs.view(B, N_t, -1, 3)
        Rs_         = Rs.view(-1, n_views, 3, 3)
        ray_c_rgbs = ray_c_rgbs.view(B, N_rays, 4, 3)
        ray_rgb_feats = ray_rgb_feats.view( B, N_rays, 4, ray_rgb_feats.shape[-1] )
        
        for b_id in range(B):
            for n_id in range(N_t):
                nei_ids           = neighbor_views[b_id, n_id] # [2];
                # select feats & pos
                select_rgbs_feat  = rgbs_feats[b_id, nei_ids] # [2, C, H, W]
                select_rgbs       = rgbs_[b_id, nei_ids] # [2, 3, H, W]
                select_depths     = depths_[b_id, nei_ids] # [2, 1, H, W] 
                select_dirs       = dirs_[b_id, n_id][None].repeat(N_nei,1,1) # [N_nei, N_rays_pv*4, 3]
                select_rs         = Rs_[b_id, nei_ids] # [N_nei, 3, 3]
                select_coarse_c   = ray_c_rgbs[b_id, n_id*N_rays_pv:(n_id+1)*N_rays_pv] # [N_rays_pv, 4, 3]
                select_ray_feats  = ray_rgb_feats[b_id, n_id*N_rays_pv:(n_id+1)*N_rays_pv] # [N_rays_pv, 4, 30]
                
                select_coarse_c   = select_coarse_c.view(-1, 3) # [N_rays_pv*4, 3]
                select_ray_feats  = select_ray_feats.view(-1, select_ray_feats.shape[-1]) # [N_rays_pv*4, 30]
                
                xy                = proj_xy[b_id, nei_ids, n_id*N_rays_pv:(n_id+1)*N_rays_pv] # [N_nei, N_rays_pv, 4, 2]
                z                 = proj_z[b_id, nei_ids, n_id*N_rays_pv:(n_id+1)*N_rays_pv] # [N_nei, N_rays_pv, 4, 1]
                # sample features.
                xy_ = xy.view(N_nei, -1, 1, 2) # [N_nei, N_rays_pv*Np, 1, 2]
                z_  = z.view(N_nei, -1, 1) # [N_nei, N_rays_pv*Np, 1]
                sam_feats_rgbs  = F.grid_sample(select_rgbs_feat, xy_, align_corners=True, mode='bilinear')[..., 0] # [N_nei, 3, N_rays_pv*Np]
                sam_rgbs        = F.grid_sample(select_rgbs, xy_, align_corners=True, mode='bilinear')[..., 0] # [N_nei, 3, N_rays_pv*Np]
                sam_depths      = F.grid_sample(select_depths, xy_, align_corners=True, mode='bilinear')[..., 0] # [N_nei, 1, N_rays_pv*Np]
                sam_dirs        = select_rs @ select_dirs.permute(0,2,1) # [N_nei, 3, N_rays]
                # sam_rays       = sam_rays.permute(0,2,1).view(N_nei, N_rays_pv, 4, 3)
                coff = self.calculate_relative_depths(sam_depths.permute(0,2,1), z_, sigmoid_coef=200.0) # [N_nei,1, N_rays_pv*Np]
                # coff = F.normalize(coff, p=1.0, dim=0) # [N_nei, N_rays_pv*N_p, 1]
                # fuse features, output 3 weights here;
                weights = nerf_func.predicted_fusion_weights( sam_feats_rgbs.permute(0,2,1), sam_dirs.permute(0,2,1), coff, select_ray_feats )
                # the final color are three colors' blending results, basic two blended colors & coarse color;
                final_color = weights[:,0:1]*sam_rgbs[0].permute(1,0) + weights[:,1:2]*sam_rgbs[1].permute(1,0) + weights[:,2:3]*select_coarse_c
                # suppress the coarse-weights; 
                weights_c.append( weights[:,-1:] )
                final_color = torch.clamp(final_color, 0, 1)
                feats.append( final_color )
        
        feats = torch.stack( feats, dim=0 ).view(B, N_t, N_rays_pv*4, -1).view(B, N_t, N_rays_pv, 4, -1).view(B, N_rays, 4, -1)
        weights_c = torch.stack( weights_c, dim=0 ).view(B, N_t, N_rays_pv*4, -1).view(B, N_t, N_rays_pv, 4, -1).view(B, N_rays, 4, -1)
        
        return feats, weights_c

    def sample_neighbor_rgb(self, rgb_feats, geo_feats_, neighbor_views, proj_xy, nerf_func):
        # Sample the neighbor views' rgb informations.
        # select the neighbor's geo features & rgb features.
        # geo_feats : [B, N_views, N_rays, N_points, C(C=16+2)]
        # rgb_feats, geo_feat, depths, NeiViews, ray_dir, Rs, proj_xy, proj_z, self._nerf_func
        # print(proj_xy.shape) # [B*Nv, N_rays*N_points, 2]

        B, N_t, N_nei = neighbor_views.shape # [B, N_t, 2(N_neighbor_views)]
        # n_views, N_rays = proj_xy.shape[1:3] # [B, N_views, N_rays, 4, 2 or 1];
        B, N_views, N_rays, N_points = geo_feats_.shape[:4]
        N_rays_pv = N_rays // N_t

        feats = [];
        # reshape tensors.
        rgb_feats_ = rgb_feats.view(-1, N_views, *rgb_feats.shape[-3:]) # [B, N_views, C, H, W]
        # depths_    = depths.view(-1, N_views, *depths.shape[-3:]) # [B,N_views, C, H, W]
        # dirs_      = dirs.view(B, N_t, -1, 3) # [B, N_t, N_rays, 3]
        # Rs_        = Rs.view(-1, N_views, 3, 3) # [B, N_v, 3, 3]

        proj_xy_   = proj_xy.view(B, N_views, N_rays, -1, 2);
        # proj_z_    = proj_z.view(B, N_views, N_rays, -1, 1);

        for b_id in range(B):
            for n_id in range(N_t):
                nei_ids = neighbor_views[b_id, n_id] # [2];
                # select geo & rgb features.
                select_rgb_feats = rgb_feats_[b_id, nei_ids] # [2, C, H, W];
                select_geo_feats = geo_feats_[b_id, nei_ids, n_id*N_rays_pv:(n_id+1)*N_rays_pv] # [2, N_rays_pv, N_points, C]

                # depths & dirs & Rs;
                # select_depths    = depths_[b_id, nei_ids] # [2, 1, H, W];
                # select_dirs      = dirs_[b_id, n_id][None].repeat(N_nei,1,1) # [2, N_points, 3]
                # select_rs        = Rs_[b_id, nei_ids] # [2, 3, 3]
                # sampled xy, z position;
                xy               = proj_xy_[b_id, nei_ids, n_id*N_rays_pv:(n_id+1)*N_rays_pv] # [2, N_rays_pv, N_p, 2]
                # z                = proj_z_[b_id, nei_ids, n_id*N_rays_pv:(n_id+1)*N_rays_pv]   # [2, N_rays_pv, N_p, 1]
                
                xy_ = xy.view(N_nei, -1, 1, 2) # [2, N_rays_pv*Np, 1, 2]
                # z_  = z.view(N_nei, -1, 1)     # [N_nei, N_rays_pv*Np, 1]
                # get the features of the rgb & depths features & sampled dirs & sampled coff.
                sam_feats_rgb = F.grid_sample(select_rgb_feats, xy_, align_corners=True, mode='bilinear')[..., 0]
                sam_feats_geo = select_geo_feats.view( N_nei, -1, select_geo_feats.shape[-1] ).permute(0,2,1)
                # sam_depths    = F.grid_sample(select_depths, xy_, align_corners=True, mode='nearest')[..., 0]
                # sam_dirs      = select_rs @ select_dirs.permute(0,2,1) # [N_nei, 3, N_points.]
                # coff = self.calculate_relative_depths(sam_depths.permute(0,2,1), z_, sigmoid_coef=200.0) # [N_nei,1, N_rays_pv*Np]
                # print(sam_feats_geo.shape, sam_feats_rgb.shape, sam_dirs.shape, coff.shape)
                fused_feats = nerf_func.agg_features(sam_feats_rgb.permute(0,2,1), sam_feats_geo.permute(0,2,1))
                # [N_rays_pv*N_points, 32]
                feats.append(fused_feats)
        
        # [B, N_rays_pv*Nt, N_points, Cdim.]
        feats = torch.stack( feats, dim=0 ).view(B, N_t, N_rays_pv, N_points, -1).view(B, N_rays, N_points, -1)
        
        return feats
        
    def average_aggregate(self, feats, batch_size, num_views, num_rays, num_sampled_points, _record_corners=False):
        '''
        feats: [B*N_view, C, N_points.]
        '''
        n_points = feats.shape[-1]
        feats_view = feats.view(batch_size, num_views, -1, n_points)
        feats_aggregate = torch.mean(feats_view, dim=1, keepdim=False) # [B, C, N_points]
        if _record_corners:
            feats_aggregate = feats_aggregate.view(batch_size, -1, num_rays, num_sampled_points, 9 if _record_corners else _)
        else:
            feats_aggregate = feats_aggregate.view(batch_size, -1, num_rays, num_sampled_points)

        return feats_aggregate


sampling_multi_view_feats = _sampling_multi_view_feats()


class OcTreeOptions(nn.Module):

    def __init__(self, levels=6, batch_size=1, rate=2, vol_dim=256, n_max_hits=512, device='cuda:0'):
        super(OcTreeOptions, self).__init__()
        self._levels = levels; self._batch_size = batch_size; self._rate = rate;
        self._gpu_id = int(str(device).split(':')[-1])

        self._octree = VoxelEncoding.MultiLayerOctree(levels, batch_size, rate, self._gpu_id)
        # init num of the max hited voxels.
        self._vol_dim = (vol_dim, vol_dim, vol_dim)
        self._n_max_hits = n_max_hits
        self._n_max_hits_coarse = None
    
    @torch.no_grad()
    def _init_octree(self, _occupied_volumes, _num_occ_voxels, _voxel_size, _vol_bbox):
        # total_num_occupied_voxels : to record the number of all occupied valid nodes.
        # octree_nodes_index : to record the total index of the valid nodes.
        # num_corners: record the num of the corner points [B, 1]
        self._vol_dim = _occupied_volumes[0].shape[-3:] # get the default volume's shape.

        assert (len(_occupied_volumes) == self._levels and len(_num_occ_voxels) == self._levels)
        # initialize the properites of the octrees.
        total_num_occupied_voxels = []
        octree_nodes_index = []
        for i,num_voxels in enumerate(_num_occ_voxels):
            total_num_occupied_voxels.append(torch.sum(num_voxels))
            octree_nodes_index.append(sum(total_num_occupied_voxels[:i]))

        self._octree.init_octree(_occupied_volumes, total_num_occupied_voxels, _num_occ_voxels, octree_nodes_index,
                                 _voxel_size, _vol_bbox) # about 1ms.
        self._octree.build_octree()
        
        return total_num_occupied_voxels, octree_nodes_index
    
    @torch.no_grad()
    def _ray_intersection(self, _ray_ori, _ray_dir, _instersect_start_level=1, th_early_stop=-1, 
                          only_voxel_traversal=True, is_training=False, opt_num_intersected_voxels=-1):
        # e.g., 256 // 2**(4 - 0 -1) = 16, 16 * 3 = 48 (num of total voxels), suppose totally travesal 3*voxels_len.
        self._n_max_hits_coarse = self._vol_dim[0]*3 // self._rate**(self._levels - _instersect_start_level - 1)
        
        if only_voxel_traversal:
            self._n_max_hits_coarse = 1024
            # self._n_max_hits_coarse = 128 + 128
            # When training, only uniformly sampling, else adapative sampling.
            # early stop and opt_num_intersected_voxels are used to control the voxels' travel
            _idx_c, _min_depths_c, _max_depths_c, ray_total_valid_len \
                = self._octree.voxel_traversal(_ray_ori, _ray_dir, self._n_max_hits_coarse, _instersect_start_level, 
                                               th_early_stop, is_training, opt_num_intersected_voxels)
                                               
            return _idx_c, _min_depths_c, _max_depths_c, ray_total_valid_len
        
        # octree traversal ( recursion type here. )
        _idx, _min_depths, _max_depths, _ray_total_valid_len, _idx_c, _min_depths_c, _max_depths_c \
            = self._octree.octree_traversal(_ray_ori, _ray_dir, self._n_max_hits_coarse, self._n_max_hits,
                                            _instersect_start_level, th_early_stop, opt_num_intersected_voxels)
        return _idx, _min_depths, _max_depths, _ray_total_valid_len, \
               _idx_c, _min_depths_c, _max_depths_c
               
    @torch.no_grad()
    def _sort_sampling(self, sampled_depths):
        return self._octree.sort_samplings(sampled_depths)

    @torch.no_grad()
    def _get_octree_corner_features(self, img_feats):
        # step 1. get the corners' position xyz,
        all_corners = self._octree.generate_corner_points() # [N_voxels, 8, 3];
        # step 2. proj to 2D xy, z;
        # step 3. sampling the images' features, 
        # grid_sample(img_feats, all_corners_xy)[..., 0]        

    @torch.no_grad()
    def _ray_voxels_sampling_points(self, _idx, _min_depths, _max_depths, _ray_total_valid_len, _uniform_noise, _instersect_start_level, _n_sampled_points):
        # Points sampling; since the points are near the object surface, we only sample nearing points.
        _sampled_idx, _sampled_depths = self._octree.ray_voxels_points_sampling(_idx, _min_depths, _max_depths, _ray_total_valid_len,
                                                                                _uniform_noise, _instersect_start_level, _n_sampled_points)
        return _sampled_idx, _sampled_depths
    
    @torch.no_grad()
    def _ray_voxels_points_sampling_coarse(self, _idx, _min_depths, _max_depths, _uniform_noise, _ray_total_valid_len, _instersect_start_level, _n_sampled_points):
        # the uniform points sampling, to sampling points in the coarse voxels. 
        _sampled_idx, _sampled_depths = self._octree.ray_voxels_points_sampling_coarse(_idx, _min_depths, _max_depths, _uniform_noise, _ray_total_valid_len,
                                                                                       _instersect_start_level, _n_sampled_points)
        return _sampled_idx, _sampled_depths

    @torch.no_grad()
    def _project_sampled_xyz(self, _ray_ori, _ray_dir, _sampled_depths, _sampled_idx, Ks, RTs, _ori_w, _ori_h, _num_views, _record_corners=False):
        # Project the sampled points to x,y coordinates in screen space, and z in camera space.
        if len(Ks.shape) == 3:
            Ks = Ks.view(self._batch_size, -1, 3, 3).contiguous() # [B, N, 3, 3];
        if len(RTs.shape) == 3:
            RTs = RTs.view(self._batch_size, -1, 3, 4).contiguous() # [B, N, 3, 4];

        _proj_xy, _proj_z = self._octree.project_sampled_xyz(_ray_ori, _ray_dir,
                            _sampled_depths, _sampled_idx,
                            Ks, RTs, 
                            _ori_w, _ori_h, _num_views, _record_corners)
        
        return _proj_xy, _proj_z

    def init_octree(self, occupied_volumes, num_occ_voxels, voxel_size, vol_bbox):
        # call the inner function.
        return self._init_octree(occupied_volumes, num_occ_voxels, voxel_size, vol_bbox)

    def ray_intersection(self, ray_ori, ray_dir, instersect_start_level=1, th_early_stop=-1, 
                         only_voxel_traversal=True, is_training=False, opt_num_intersected_voxels=-1):
        return self._ray_intersection(ray_ori, ray_dir, instersect_start_level, 
                                      th_early_stop, only_voxel_traversal, is_training, opt_num_intersected_voxels)
    
    def ray_voxels_sampling_points(self, idx, min_depths, max_depths, ray_total_valid_len, uniform_noise, instersect_start_level=1, n_sampled_points=128):
        return self._ray_voxels_sampling_points(idx, min_depths, max_depths, ray_total_valid_len, uniform_noise, instersect_start_level, n_sampled_points)

    def ray_voxels_points_sampling_coarse(self, idx, min_depths, max_depths, uniform_noise, ray_total_valid_len, instersect_start_level=1, n_sampled_points=128):
        return self._ray_voxels_points_sampling_coarse(idx, min_depths, max_depths, uniform_noise, ray_total_valid_len, instersect_start_level, n_sampled_points)
    
    def project_sampled_xyz(self, _ray_ori, _ray_dir, _sampled_depths, _sampled_idx, Ks, RTs, _ori_w, _ori_h, _num_views, _record_corners=False):
        return self._project_sampled_xyz(_ray_ori, _ray_dir, _sampled_depths, _sampled_idx, Ks, RTs, _ori_w, _ori_h, _num_views, _record_corners)
    
    def sort_sampling(self, _sampled_depths):
        return self._sort_sampling(_sampled_depths)
    
    @torch.no_grad()
    def get_all_nodes(self):
        # all nodes tensors : [x,y,z,size,valid,batch_idx] * N
        return self._octree.get_nodes_tensor()

    @torch.no_grad()
    def get_octree(self):
        return self._octree
