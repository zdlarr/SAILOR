
import sys, os
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

from c_lib.VoxelEncoding.dist import VoxelEncoding # Voxel encoding library.
from torch.autograd import Function, gradcheck, Variable, grad

class _freq_encoder(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, inputs, deg_dim):
        # inputs : [n_points, input_dim]
        device = int(str(inputs.device).split(':')[-1])

        outputs = VoxelEncoding.freq_encode_forward(inputs, deg_dim, device)

        ctx.save_for_backward(inputs, outputs)
        ctx.infos = [deg_dim, device]
        return outputs
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        inputs, outputs = ctx.saved_tensors
        deg_dim, device = ctx.infos

        grad_inputs = VoxelEncoding.freq_encode_backward(grad_outputs, inputs, outputs, deg_dim, device)
        
        return grad_inputs, None

freq_encoder = _freq_encoder.apply



class FreqEncoder(nn.Module):

    def __init__(self, input_dim=3, degree=4):
        super().__init__()

        self._input_dim  = input_dim
        self._degree     = degree
        self._output_dim = input_dim + input_dim*degree*2

    def forward(self, inputs, *args):
        prefix_shape = list(inputs.shape[:-1])
        if args: # if the args existed, which save the num_rays.
            batch_size, num_views, num_rays, num_sampled = args
            inputs = inputs.view(batch_size*num_views*num_rays*num_sampled, self._input_dim)
        else:
            inputs = inputs.view(-1, self._input_dim)

        outputs = freq_encoder(inputs, self._degree)

        if args:
            outputs = outputs.view([batch_size, num_views, num_rays, num_sampled, self._output_dim])
        else:
            outputs = outputs.view(prefix_shape + [self._output_dim])
        
        return outputs


if __name__ == '__main__':    
    p = torch.randn([5, 3, 1024, 128, 3], requires_grad=True).cuda()
    freq_encoder0 = FreqEncoder()
    output = freq_encoder0(p)
    L = torch.mean(output)
    print(grad(outputs=L, inputs=p)[0].cpu().shape) 