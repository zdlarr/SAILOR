"""
    Models' utils for our GeoNet & RenderNet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val


def index2D(feat, uv, mode='bilinear'):
    # feat: [B, C, H, W]
    # grid sampling 2D features.
    uv = uv.transpose(1, 2) # [B, N, 2]
    uv = uv.unsqueeze(2) # [B,N,1,2]
    samples = F.grid_sample(feat, uv, align_corners=True, mode=mode) # [B, C, N ,1];
    # samples = grid_sample(feat, uv) # since the original grid_sample is not differential, reimplemented grid-sample function.
    return samples[..., 0] # [B, C, N]

def index3D(feat, uv3d, mode='bilinear'):
    # feats: [B, C, D, H, W];
    # grid sampling 3D features.
    grid3d = uv3d.permute(0, 2, 1).unsqueeze(1).unsqueeze(1) # [B, 1, 1, N, 3]
    samples = F.grid_sample(feat, grid3d, mode=mode) # [B, C, 1, 1, N]
    return samples[:, :, 0, 0, :] # [B, C, N]


def reshape_multiview_tensor(tensors):
    # the multi-view tensors: [B, N_views, ...] * N_tensors (list)
    if not isinstance(tensors, list):
        tensors = [tensors]

    for i in range(len(tensors)):
        # the shape of the tensor: [B, N_views, ...].
        assert isinstance(tensors[i], torch.Tensor), 'error type of tensor'
        shape = tensors[i].shape
        tensors[i] = tensors[i].view(shape[0] * shape[1], *shape[2:])
    
    # the returned tensors are in list.
    return tensors


def reshape_sample_tensor(sample_tensor, n_views):
    # sample_tensor : [B, *shape]
    if n_views == 1:
        return sample_tensor
        
    # reshape to [B * N_views, 3, N_points].
    len_shape = len(sample_tensor.shape) - 1
    sample_tensor = sample_tensor.unsqueeze(dim=1)
    sample_tensor = sample_tensor.repeat(1, n_views, *([1] * len_shape)) # [B, N_views, 3, N_points]
    sample_tensor = sample_tensor.view(
        sample_tensor.shape[0] * sample_tensor.shape[1],
        *sample_tensor.shape[2:]
    )
    return sample_tensor # [B * N_views, 3, N_p].


def build_mesh_grid(size, device):
    grid_x, grid_y = torch.meshgrid(torch.arange(0, size, 1), torch.arange(0, size, 1))
    grids_volume   = torch.stack([grid_x, grid_y, torch.ones(size, size)], dim=0).to(device) # [3, H, W]
    grids_volume   = grids_volume[[1, 0, 2]]
    return grids_volume


class Depth2Normal(nn.Module):
    
    def __init__(self, opts, device):
        super(Depth2Normal, self).__init__()
        self.opts = opts
        self.device = device
    
    def get_points_coordinate(self, depth, int_inv):
        b, c, h, w = depth.shape
        device = depth.device
        y, x = torch.meshgrid([torch.arange(0, h, 1, dtype=torch.float32, device=device), torch.arange(0, w, 1, dtype=torch.float32, device=device)])
        # y, x = y.contiguous(), x.contiguous()
        y, x = y.unsqueeze(0).repeat(b, 1, 1).view(b, -1), x.unsqueeze(0).repeat(b, 1, 1).view(b, -1)
        
        # y, x = torch.cat([y.view(1, h * w)] * b, dim=0), torch.cat([x.view(1, h * w)] * b, dim=0)
        # print(y.shape)
        if self.opts.project_mode == 'ortho':
            xyz = torch.stack((x, y, depth.view(b, -1)), dim=1) # [b, 3, H * W]; [x, y, depth]
            xyz = int_inv @ xyz # [b, 3, H * W]， pt_screen = K @ pts -> pts = K^{-1} @ pt_screen
        elif self.opts.project_mode == 'perspective':
            xyz = torch.stack((x, y, torch.ones_like(x)), dim=1) # [b, 3, H * W] [X,Y,1]
            xyz = (int_inv @ xyz) * depth.view(b, 1, -1)

        del x, y
        return xyz.view(b, 3, h, w)

    def points2normal(self, p):
        p_ctr = p[:, :, 1:-1, 1:-1]
        vw = p_ctr - p[:, :, 1:-1, 2:]
        vs = p[:, :, 2:, 1:-1] - p_ctr
        ve = p_ctr - p[:, :, 1:-1, :-2]
        vn = p[:, :, :-2, 1:-1] - p_ctr
        n1 = torch.cross(vs, vw) 
        n2 = torch.cross(vn, ve)
        n1 = F.normalize(n1, p=2, dim=1, eps=1e-7)
        n2 = F.normalize(n2, p=2, dim=1, eps=1e-7)
        n = n1 + n2
        n = F.normalize(n, p=2, dim=1, eps=1e-7)
        
        del p_ctr, vw, vs, ve, vn, n1, n2
        return F.pad(n, [1]*4, mode='reflect') # [B,3,H,W]

    def forward(self, depths, K, scale=1.0):
        # input depths: [B,C,H,W];
        depths_scaled = depths * scale # since the depth is of m degree, transform to cm. e.g., scale sometime is needed
        
        int_inv = torch.inverse(K) # [B,3,3]

        p = self.get_points_coordinate(depths_scaled, int_inv)
        n = self.points2normal(p) * 0.5 + 0.5 # [-1, 1] to [0, 1]
        del depths_scaled, p

        return n
    

class project3D(nn.Module):
    """
        The projected 3d points: [B, 3, N];
        cam_calibs: [K, RT];
        proj_type: ['perspective', 'ortho']
    """

    def __init__(self, opts, proj_type='perspective'):
        super(project3D, self).__init__()
        self.opts = opts
        if proj_type == 'perspective':
            self.project_func = self.proj_persp
        elif proj_type == 'ortho':
            self.project_func = self.proj_ortho
        else:
            raise Exception('Error projection type.')
    
    def project(self, points, cam_calibs):
        K = cam_calibs[0]
        R, T = cam_calibs[1][:, :3, :3], cam_calibs[1][:, :3, -1:]
        return self.project_func(points, K, R, T)
    
    def proj_ortho(self, points, K, R, T):
        # R: [B, 3, 3]; T: [B, 3, 1]
        # To cameras' coordinate.
        pts = R @ points + T
        # To screen coordinate
        depth = pts[:, -1:, :].clone()  # [B,1, N]
        pts_screen = K @ pts # [f1 * x_c + f_x, f2 * y_c + f_y]
        # pts_screen: [B, 2, N];  depth: [B, 1, N]
        return (pts_screen[:, :2, :] - self.opts.load_size // 2) / (self.opts.load_size // 2), depth
    
    def proj_persp(self, points, K, R, T, depth_thres=0.01): # divide by / depth.
        pts = R @ points + T # [B, 3, N]
        # To screen coordinate
        depth = pts[:, -1:, :].clone() # [B, 1, N]
        invalid = depth < depth_thres # invalid depth values.
        depth_ = depth.clone() # get depth values.
        depth_[invalid] = 1 # assign with 1, don't divide by this depth value.
        
        pts /= depth_ # [x / d, y / d, 1.];
        pts_screen = K @ pts # [f1 * x / d + f_x, f2 * y / d + f_y]
        # in dataset, the intrinsic matrix has been normalized to [f1 / fx, f2 / fy, 1], -> [-1,1]
        grids = (pts_screen[:, :2, :] - self.opts.load_size // 2) / (self.opts.load_size // 2) # to [-1,1] [B, 2, N]
        grids[invalid.repeat(1,2,1)] = -2 # invalid with -2.
        
        return grids, depth

    
    # plt.axis('equal')
    # plt.scatter(pts_screen.cpu().numpy()[2, 0, :], pts_screen.cpu().numpy()[2, 1, :])
    # plt.show()
    
    # The projected points, i.e., pts_screen are projected to screen space, and no normalization on depth.
    # return pts_screen, depth # [B * N_v, 1, N_points]
    
    
class ExtractDepthEdgeMask(nn.Module):
    
    def __init__(self, thres_):
        super().__init__()
        self.thres = thres_
    
    def forward(self, batch_depth):
        B, _, H, W = batch_depth.size()
        patch = F.unfold(batch_depth, kernel_size=3, padding=1, stride=1)
        min_v, _ = patch.min(dim=1, keepdim=True)
        max_v, _ = patch.max(dim=1, keepdim=True)

        mask = (max_v - min_v) > self.thres # extract the depths' dis-continous region.
        mask = mask.view(B, 1, H, W).float()
        
        return mask
    

class DilateMask(nn.Module):
    # expand the mask's boundaries.
    def __init__(self):
        super().__init__()
        kernel = [[0.00, -0.25, 0.00],
                  [-0.25, 1.00, -0.25],
                  [0.00, -0.25, 0.00]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.padding_func = nn.ReplicationPad2d(1) 
        
    def forward(self, batch_mask, iter_num):
        # batch_masks : [B, 1, H, W]
        mask = batch_mask.clone()
        # iteratively dilate the binary mask.
        for _ in range(iter_num):
            padding_mask = self.padding_func(mask) # the padding mask tensor
            res = F.conv2d(padding_mask, self.weight, bias=None, stride = 1, padding=0)
            mask[res.abs() > 0.0001]  = 1.0 # the residual maskes.
            mask[res.abs() <= 0.0001] = 0.0
            
        return mask

# sometimes, the eroded masks don't have enough region, the sampled rays may not cover the all the voxels.
# this may lead to not coverage.
class DilateMask2(nn.Module):
    def __init__(self, ksize=7):
        super(DilateMask2, self).__init__()
        self._iter_num = 8
        self.max_pool = nn.MaxPool2d(kernel_size=ksize, stride=1, padding=ksize // 2)
    
    def forward(self, batch_mask, iter_num=None):
        mask = batch_mask.clone()
        for _ in range(iter_num if iter_num is not None else self._iter_num):
            mask = self.max_pool(mask)
        
        mask[mask.abs() > 0.0001] = 1.0
        mask[mask.abs() <= 0.0001] = 0.0
        return mask
    
class Sine(nn.Module):
    
    def __init__(self, w0=30):
        super().__init__()
        self.w0 = w0
        
    def forward(self, x):
        return torch.sin(self.w0 * x)
    

class SirenLayer(nn.Conv1d):
    
    def __init__(self, input_dim, out_dim, kernel_size, *args, is_first=False, activation=True, **kwargs):
        self.is_first = is_first
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.w0 = 30
        self.c = 6
        super().__init__(input_dim, out_dim, kernel_size, *args, **kwargs)
        self.activation = Sine(self.w0) if activation else None
        
    # override.
    def reset_parameters(self) -> None:
        # NOTE: in offical SIREN, first run linear's original initialization, then run custom SIREN init.
        #       hence the bias is initalized in super()'s reset_parameters()
        super().reset_parameters()
        with torch.no_grad():
            dim = self.input_dim
            w_std = (1 / dim) if self.is_first else (math.sqrt(self.c / dim) / self.w0)
            self.weight.uniform_(-w_std, w_std)
    
    def forward(self, x):
        out = super().forward(x) # first layer is the original dense layer.
        if self.activation is None:
            return out
        return self.activation(out)


class DenseLayer(nn.Conv1d):
    def __init__(self, input_dim: int, out_dim: int, kernel_size: int, *args, activation=None, norm=None, **kwargs):
        super().__init__(input_dim, out_dim, kernel_size, *args, **kwargs)
        self.activation = activation
        self.norm = norm

    def forward(self, x):
        out = super().forward(x)
        if (self.activation is None) and (self.norm is None):
            return out
        
        if self.norm is None and (self.activation is not None):
            return self.activation(out)
        else:
            assert (self.norm is not None) and (self.activation is not None), 'err inputs'
            return self.activation(self.norm(out))
    

class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super(Embedder, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input: torch.Tensor):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out
    

def get_embedder(multires, input_dim=3):
    if multires < 0:
        return nn.Identity(), input_dim

    embed_kwargs = {
        "include_input": True,
        "input_dim": input_dim,
        "max_freq_log2": multires - 1,
        "N_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim

if __name__ == '__main__':
    emb_fn, _ = get_embedder(6, 2)
    x = torch.zeros([6, 1000, 2]).cuda()
    x = emb_fn(x)
    y = torch.zeros([6, 128, 200]).cuda()
    siren_layer = DenseLayer(128, 64, 1).cuda()
    y = siren_layer(y)
    print(x.shape, y.shape)