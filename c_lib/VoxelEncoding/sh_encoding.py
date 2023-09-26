
import sys, os
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

from c_lib.VoxelEncoding.dist import VoxelEncoding # Voxel encoding library.
from torch.autograd import Function, gradcheck, Variable, grad

# SH encoding.
class _sh_encoder(Function):
    
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, inputs, deg_dim, calc_grad_inputs=False):
        # inputs: [N_points, input_dim]
        # return: [N_points, output_dim];
        inputs = inputs.contiguous()
        device = int(str(inputs.device).split(':')[-1])
        output_dim = deg_dim ** 2 # define the output dims.
        
        if calc_grad_inputs: # the inputs dim: inputs.shape[-1];
            dy_dx = torch.empty([inputs.shape[0], inputs.shape[1] * output_dim], dtype=inputs.dtype, device=inputs.device)
        else:
            dy_dx = None
        
        outputs = VoxelEncoding.sh_encode_forward(inputs, deg_dim, dy_dx, device);
        
        ctx.save_for_backward(inputs, dy_dx) # input data & dy/dx.
        ctx.infos = [deg_dim, device]
        
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad0):
        # grad: [B, C**2]
        inputs, dy_dx = ctx.saved_tensors
        
        if dy_dx is not None:
            grad0 = grad0.contiguous()
            deg_dim, device = ctx.infos
            grad_inputs = VoxelEncoding.sh_encode_backward(grad0, inputs, deg_dim, dy_dx, device)
            return grad_inputs, None, None
        
        return None, None, None
    

sh_encoder = _sh_encoder.apply


class SHEncoder(nn.Module):
    
    def __init__(self, input_dim=3, degree=4):
        super().__init__()
        
        self._input_dim  = input_dim
        self._degree     = degree # [1,8]
        self._output_dim = degree ** 2
        self._normalize  = True
        
        assert self._input_dim == 3, "SH encoder only supports input dim == 3"
        assert self._degree > 0 and self._degree <= 8, "SH encoder only supports degree in [1,8]"
    
    def __repr__(self):
        return f"SHEncoder: input_dim={self._input_dim} degree={self._degree}"
    
    def normalize(self, x):
        return (x - 0.5) * 2.0 # in [-1, 1]
    
    def forward(self, inputs):
        # inputs : [N, 3];
        if self._normalize:
            inputs = self.normalize(inputs)
        
        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.reshape(-1, self._input_dim)
        
        outputs = sh_encoder(inputs, self._degree, inputs.requires_grad)
        outputs = outputs.reshape(prefix_shape + [self._output_dim])
        
        return outputs


class SHEncoder_torch(nn.Module):
    def __init__(self, input_dim=3, degree=4):
    
        super().__init__()

        self.input_dim = input_dim
        self.degree = degree

        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5

        self.output_dim = degree ** 2

        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ]

    def forward(self, input, **kwargs):

        result = torch.empty((*input.shape[:-1], self.output_dim), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                #result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                result[..., 6] = self.C2[2] * (3.0 * zz - 1) # xx + yy + zz == 1, but this will lead to different backward gradients, interesting...
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return result


if __name__ == '__main__':
    B = 25600
    C = 3
    degree = 4
    import tinycudann as tcnn

    enc1 = SHEncoder(degree=degree).cuda()
    enc2 = SHEncoder_torch(degree=degree).cuda()
    
    enc3 = tcnn.Encoding(
        n_input_dims=3,
        encoding_config={
            "otype": "SphericalHarmonics",
            "degree": 4,
        },
    )
    
    x1 = torch.rand(B, 3).cuda() # in [0, 1]
    x1 = x1 / (torch.norm(x1, dim=-1, keepdim=True) + 1e-8)
    x1.requires_grad_(True)
    
    # x2 = x1.detach().clone() * 2 - 1
    x2 = x1.clone() * 2 - 1
    x3 = x1.clone()
    x3.requires_grad_(True)

    y1 = enc1(x1)
    y2 = enc2(x2)
    
    y1.sum().backward()
    y2.sum().backward()
    
    print(x1.grad)
    print(x2.grad)
    
    y3 = enc3(x3)
    y3.sum().backward()
    print(x3.grad)