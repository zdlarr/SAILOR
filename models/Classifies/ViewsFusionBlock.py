""" 
    Views fusion block
    1. the view to view multi-heads transformer fusion block, using 8 heads and 2 layers without position encoding.
    2. the transformer layer multi-head attention module is proposed to be used in head features; since the head only contain RGB features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks import get_norm_layer
from models.Classifies.SurfaceClassifierPF import SurfaceClassifier
import matplotlib.pyplot as plt
import os, sys
        
"""
    View feature fusion based on error between queried z & original depth.
"""

class MultiViewWeightedFusion(nn.Module):

    def __init__(self, opts, device):
        super(MultiViewWeightedFusion, self).__init__()
        self.opts = opts
        self.device = device
        
        # the fusion_weights: [B, N_v, N_p].
        self.fusion_weights = None
        self.num_views = opts.num_views

        # feature fusion scale;
        self.weight_scale = 100.0
    
    def calculate_fusion_func(self, xy, z, depths, depths_edge_mask, sample_func):
        if self.opts.merge_method == 'distance':
            self.pre_calculate_fusion_weight0(xy, z, depths, depths_edge_mask, sample_func)
        elif self.opts.merge_method == 'visible':
            self.pre_calculate_fusion_weight1(xy, z, depths, depths_edge_mask, sample_func)
        elif self.opts.merge_method == 'mean':
            self.pre_calculate_fusion_weight2(xy, z, depths, depths_edge_mask, sample_func)
        else:
            raise Exception('Non valid feature fusion methods.')

    def calculate_psdf(self, xy, z, depths, depths_edge_mask, sample_func):
        assert self.opts.type_input == 'rgbd' or self.opts.type_input == 'depth_only', 'do not support t_psdf.'
        edge_mask_c1f = sample_func(depths_edge_mask, xy) # default, bilinear;
        edge_mask_c1b = edge_mask_c1f > 0.01
        depth_queried = sample_func(depths, xy, edge_mask=edge_mask_c1b) # [B * N_v, 1, N_points.]
        return z - depth_queried

    def pre_calculate_fusion_weight0(self, xy, z, depths, depths_edge_mask, sample_func):
        """
            Fusion by ratio.
            1. Directly projected each points to 2D image, and get the pixel-aligned depth, 
            2. calculate the depth error between the aligned-pixel depth & original z. (get abs results)
            3. according to the min_distance to fuse each view's feature.
            Fusion by ratio: [0, 1];
        """
        batch_num = xy.shape[0] // self.num_views
        
        # calculate the truncated_psdf values.
        psdf = self.calculate_psdf(xy, z, depths, depths_edge_mask, sample_func)
        psdf = torch.abs(psdf) # [B * N_v, 1, N_points];
        psdf = psdf.view(batch_num, self.num_views, -1) # [B, N_v, N_points]
        
        # method  1. only select one view;
        # self.fusion_weights = torch.zeros_like(psdf)
        # self.fusion_weights[torch.where(psdf == torch.min(psdf, dim=1, keepdim=True)[0])] = 1

        # method  2. the features are fused by the ratio of the distance
        # weights = 1.0 / (psdf + 1e-7) # [B, N_v, N_points]
        
        # method  3. based on softmax function.
        weights = torch.exp(- self.weight_scale * psdf) + 1e-7 # according to the feature weights.
        
        # method  4. based on the softplus function.
        # weights = torch.log( 1 + torch.exp(- self.weight_scale * psdf) )
        
        self.fusion_weights = weights / torch.sum(weights, dim=1, keepdim=True)
    
    def pre_calculate_fusion_weight1(self, xy, z, depths, depths_edge_mask, sample_func):
        """
            Fusion by the visible rate.
            visible: psdf < 0. else psdf > 0
        """
        batch_num = xy.shape[0] // self.num_views
        psdf = self.calculate_psdf(xy, z, depths, depths_edge_mask, sample_func) # [B * N_v, 1, N_points];
        psdf = psdf.view(batch_num, self.num_views, -1) # [B, N_v, N_points]
        self.fusion_weights = torch.zeros_like(psdf)
        self.fusion_weights[psdf < 5] = 1 # the visible threshold is set to 5.
    
    def pre_calculate_fusion_weight2(self, xy, z, depths, depths_edge_mask, sample_func):
        # average fusion.
        batch_num = xy.shape[0] // self.num_views
        num_points = xy.shape[-1]
        self.fusion_weights = torch.ones(batch_num, self.num_views, num_points, device=self.device) / self.num_views # [B, N_v, N_p]

    def fusion_features(self, features):
        # sum ( fusion_weights * features(B, N_v, C, N_p) , dim=1)
        assert self.fusion_weights is not None, 'No fusion weights existed.'
        return torch.sum(self.fusion_weights[:, :, None, :] * features, dim=1, keepdim=False)

        
class FaceWeightedFusion(MultiViewWeightedFusion):

    def __init__(self, opts, device):
        MultiViewWeightedFusion.__init__(self, opts, device)
        self.scale_psdf = 1e3
        self.scale_gaussian = 0.98
        self.scale_xy = 10 # for controling the weight for xy circle weights, smaller and the face weights more;
        self.max_pool = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.iter = 15 # to control the fusion decrease border line;
        self.mask_erode = None
        self.xyzrgb = [] # for recording rgb data.

    def obtain_face_mask_weight(self, mask):
        # calculate the facial weights, (border.)
        mask_erode = mask.clone()
        mask_final = torch.zeros_like(mask)
        init_rate = 0.1
        for i in range(self.iter):
            mask_erode_nxt = -self.max_pool(-mask_erode)
            mask_erode_border = mask_erode - mask_erode_nxt
            rate = init_rate + i * ((1 - init_rate) / self.iter) # 0.2, 0.28, 0.36, .... 1.0;
            if rate > 1.0:
                rate = 1.0

            mask_erode_border *= rate
            mask_final += mask_erode_border
            mask_erode = mask_erode_nxt

        mask_final += mask_erode

        self.mask_erode = mask_final

        # for facial eroded mask.
        # plt.imshow(self.mask_erode[0,0].cpu().numpy())
        # plt.axis('off')
        # plt.colorbar()
        # plt.savefig('./masked_weights.jpg', transparent=True, bbox_inches='tight', pad_inches=0.0)
        # plt.show()

    def calculate_fusion_weights(self, bid, xy, z, face_depth, face_depth_edge_mask, face_mask, sample_func, x_weighted=True):
        # calculate the fusion weights of the projected points on faces.
        # the points near to the face own the larger fusion weights. 
        # xy: [B, 2, N_points]. usually the B is 1 here.
        # step 1. get the psdf of the query points.
        psdf = self.calculate_psdf(xy, z, face_depth, face_depth_edge_mask, sample_func) # [B, 1, N_points];
        # step 2. get the sampled label for face region (val = 1) when the face region is located in;
        edge_mask_c1f = sample_func(face_depth_edge_mask, xy)
        edge_mask_c1b = edge_mask_c1f > 0.01
        labels_in_region = sample_func(self.mask_erode[bid:bid+1], xy, edge_mask_c1b) # [B, 1, N_points] where B = 1 here.
        # step 3. get the max xy. distance from center.
        # max_xy_distance, _  = torch.max(torch.sqrt(torch.sum(xy ** 2, dim=1)), dim=-1) # [B]
    
        # method 1. exp
        # psdf = torch.abs(psdf) # the distance
        # psdf_weight = 1.0 / (psdf + 1e-7)
        # normalization to [0,1]; transform the weight;
        # psdf_weight = 2.0 / (1.0 + torch.exp(-self.scale_psdf * psdf_weight)) - 1.0 # [B, 1, N_points]
        
        # method 2. gaussian, when depth = 0, psdf is large, and the psdf_weight.
        psdf_weight = torch.exp(- psdf ** 2 * self.scale_psdf) # [B, 1, N_points]

        # get the gaussian weights for border weighted in (xy) axis;
        # scale_gaussian = - torch.log(torch.tensor(0.05)) / (max_xy_distance ** 2) # solve a from the function : exp(- a x^2) = 0.4
        # gaussian_weight = torch.exp(-(xy[:, 0:1]**2 + xy[:, 1:]**2) * scale_gaussian[:, None, None]) # [B, 1, N_points]
        # gaussian_weight = torch.exp(-(xy[:, 0:1]**2 + xy[:, 1:]**2) * self.scale_gaussian) # [B, 1, N_points]

        # multiply the fusion weights;
        if x_weighted:
            self.fusion_weights = psdf_weight * labels_in_region
        else:
            self.fusion_weights = psdf_weight
        
    def save_points_weight(self, xy, z):
        import numpy as np
        # xy: [B, 2, N_p]; z: [B, 1, N_p]; weights: [B, 1, N_p]. Only save the first one.
        r = (self.fusion_weights[0].permute(1,0) * 200 + 30).cpu().int().numpy() # [N,1]
        g = (self.fusion_weights[0].permute(1,0) * 200 + 30).cpu().int().numpy()
        b = (self.fusion_weights[0].permute(1,0) * 255 + 85).cpu().int().numpy()
        xy_ = xy[0].permute(1, 0).cpu().numpy() # [N, 2]
        z_  = z[0].permute(1, 0).cpu().numpy() # [N, 1]

        to_save = np.concatenate([xy_, z_, r, g, b], axis=-1) # [N, 6]
        idx_regions = np.where((z_ < 0.3) & (z_ > -1.0))[0] # threshold.
        if idx_regions.shape[0] != 0:
            to_save = to_save[idx_regions]
            self.xyzrgb.append(to_save)

    def save_points_weight_(self):
        import numpy as np
        ply_dir = os.path.join(self.opts.model_dir, 'val_results')
        save_path = os.path.join(ply_dir, 'points_fusion_weights.ply')

        to_save = np.concatenate(self.xyzrgb, axis=0)
        return np.savetxt(save_path,
                        to_save,
                        fmt='%.6f %.6f %.6f %d %d %d',
                        comments='',
                        header=(
                            'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                            to_save.shape[0])
                        )

    def calculate_fusion_weights_v2(self, xy):
        self.fusion_weights = torch.exp(-((xy[:, 0:1] * self.scale_xy)**2 + (xy[:, 1:] * self.scale_xy)**2) * self.scale_gaussian) # [B, 1, N_points]
    
    def fusion(self, occ_body, occ_face):
        # according to the fusion weights, fuse the occ.
        # occ_body: [B, 1, N_points]; occ_face: [B, 1, N_points], weights: [B, 1, N_points;]
        return self.fusion_weights * occ_face + (1 - self.fusion_weights) * occ_body


        

        
        