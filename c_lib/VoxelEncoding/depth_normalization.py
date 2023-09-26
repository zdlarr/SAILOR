
import sys, os
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

from c_lib.VoxelEncoding.dist import VoxelEncoding # Voxel encoding library.
from torch.autograd import Function, gradcheck, Variable, grad


class _depth_normalizer(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, depths, valid_min_depth, valid_max_depth, divided2=True, user_dist=-1):
        # inputs shape: [B, N_views, H, W]
        device = int(str(depths.device).split(':')[-1])
        if divided2:
            normalized_depth, mask_depths, mid, dist = VoxelEncoding.depth_normalize(depths, 
                                                                                    valid_min_depth, valid_max_depth, user_dist, False, True,
                                                                                    device)
        else:
            normalized_depth, mask_depths, mid, dist = VoxelEncoding.depth_normalize(depths, 
                                                                                    valid_min_depth, valid_max_depth, user_dist, False, False,
                                                                                    device)

        return normalized_depth, mask_depths, mid, dist

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_a, grad_b, grad_c, grad_d, grad_e):
        return None, None, None, None, None


depth_normalizer = _depth_normalizer.apply


# The depth normalization class, approxmately 1ms for 3 views' data
class DepthNormalizer(nn.Module):

    def __init__(self, opts, divided2=True):
        super(DepthNormalizer, self).__init__()
        # obtain the min & max valid depth, default as [0.2, 5.0]
        self._min_valid_depth = opts.valid_depth_min
        self._max_valid_depth = opts.valid_depth_max
        self._num_views       = opts.num_views
        self._batch_size      = opts.batch_size
        self._divided2        = divided2

    @torch.no_grad()
    def forward(self, depths, user_dists=-1):
        # depths shape [B * N_v, 1, h,w]
        h, w = depths.shape[-2:]
        depths = depths.view(self._batch_size, -1, h, w) # [B, N, H, W]

        normalized_depth, mask_depths, mid, dist \
            = depth_normalizer(depths, self._min_valid_depth, self._max_valid_depth, self._divided2, user_dists)

        # [B, N, H, W] -> [B*N, 1, H, W]
        normalized_depth = normalized_depth.view(-1, 1, h, w)
        mask_depths      = mask_depths.view(-1, 1, h, w)

        return normalized_depth, mask_depths, mid, dist



