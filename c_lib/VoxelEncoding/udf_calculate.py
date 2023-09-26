
import sys, os
import time
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

from c_lib.VoxelEncoding.dist import VoxelEncoding # Voxel encoding library.
from torch.autograd import Function, gradcheck, Variable, grad

# calculate the udf values between points and pcds

class _udf_calculating(Function):
    
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, ray_ori, ray_dir, sampled_depths, pcds):
        # get gpu id first.
        gpu_id = int(str(ray_ori.device).split(':')[-1]) # 0 or 1.
        # udfs: [b,n_rays,n_sampled,1], dirs: [b,n_rays,n_sampled,3], valid: [b,n_rays,n_sampled,1]
        # t0 = time.time()
        udfs, udf_dirs, valid = VoxelEncoding.udf_calculating(ray_ori, ray_dir, sampled_depths, pcds, gpu_id)
        # torch.cuda.synchronize()
        # t1 = time.time()
        
        # print(t1 - t0, 'second');
        
        return udfs, udf_dirs, valid
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad0, grad1, grad2):
        return None, None, None, None
    
class _udf_calculating_v2(Function):
    
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, pcds0, pcds1):
        # get gpu id first.
        gpu_id = int(str(pcds0.device).split(':')[-1]) # 0 or 1.
        # udfs: [b,n_p0,1], dirs: [b,n_p0,3], valid: [b,n_p0,1]
        # t0 = time.time()
        udfs, udf_dirs, valid = VoxelEncoding.udf_calculating_v2(pcds0, pcds1, gpu_id)
        # torch.cuda.synchronize()
        # t1 = time.time()
        
        # print(t1 - t0, 'second');
        
        return udfs, udf_dirs, valid
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad0, grad1, grad2):
        return None, None


udf_calculating    = _udf_calculating.apply
udf_calculating_v2 = _udf_calculating_v2.apply

if __name__ == '__main__':
    test_pcds = torch.rand([1, 550000, 3]).to('cuda:0')
    # test_ori = torch.zeros([2, 2048, 3]).to('cuda:0')
    # test_dir = torch.ones([2, 2048, 3]).to('cuda:0')
    # test_sampled_depth = torch.zeros([2, 2048, 128, 1]).to('cuda:0')
    test_pcds0 = torch.zeros([1, 15000, 3]).to('cuda:0')
    udfs, dirs, valids = udf_calculating_v2(test_pcds0, test_pcds)
    d = torch.min( torch.sqrt( torch.sum(test_pcds**2, dim=-1) ))
    print( udfs[0,0], d, udfs.shape)

    # t0 = torch.rand([400000, 1]).to('cuda:0')
    # t1 = torch.rand([600000, 1]).to('cuda:0')
    
    # o = torch.min(torch.abs(t0.repeat(1, t1.shape[0]) - t1.repeat(t0.shape[0], 1)), dim=0)
    # print(o.shape)