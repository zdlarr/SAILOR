"""
    Utils for data augmentation, data preprocessing, 
    e.g, 1. the augmentation added on depth information;
         2. the normalization of the depth.
         
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import sys, os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy import ndimage

def normalize_depth(opts, depth, if_clip=True):
    
    if opts.depth_normalize == 'linear':
        z = (depth - opts.z_size) / opts.z_bbox_len
    
        # z = ((depth - opts.z_size) / (opts.z_size // 2)) * (opts.load_size // 2) / opts.z_size
    elif opts.depth_normalize == 'exp':
        t = 1 / (opts.z_size // 4)
        z = 2.0 / (1 + np.exp(- t * depth)) - 1.0 # normalized in [-1, 1]
    elif opts.depth_normalize == 'adaptive':
        z, mid_depths, dis_depths = normalize_face_depth(opts, depth, return_prop=True, threshold=2*opts.z_bbox_len)
        if if_clip:
            if isinstance(depth, np.ndarray):
                z = np.clip(z, -1, 1)
            else:
                z = torch.clamp(z, -1, 1)
                
        return z, mid_depths, dis_depths
    else:
        raise Exception('Non support normalizaion methods.')
    
    return z


def unnormalize_depth(opts, depth, mid_depths=None, dis_depths=None):
    
    if opts.depth_normalize == 'linear':
        z = depth * opts.z_bbox_len + opts.z_size
        # z = (depth * opts.z_size / (opts.load_size // 2)) * (opts.z_size // 2) + opts.z_size
    elif opts.depth_normalize == 'exp':
        t = 1 / (opts.z_size // 4)
        z = - np.log((2.0 / depth + 1.0) - 1) / t
    elif opts.depth_normalize == 'adaptive':
        z = unnormalize_face_depth(opts, depth, mid_depths, dis_depths)
    else:
        raise Exception('Non support normalizaion methods.')

    return z


def normalize_face_depth(opts, depth, return_prop=False, if_clip=False, threshold=0.3, dis=None):
    # the distance for depth is assigned for face.
    # depth : [B, 1, H, W]
    resize = False
    is_np = False

    if isinstance(depth, np.ndarray):
        depth = torch.as_tensor(depth, dtype=torch.float)
        is_np = True

    if len(depth.shape) == 3:
        depth = depth[None, ...]
        resize = True
        
    b, c, h, w = depth.shape
    depth_view = depth.view(b, -1)
    max_depths, _ = torch.max(depth_view, dim=1) # []
    depth_sort, _ = torch.sort(depth_view, dim=1) # [b, N], unsort big -> small;
    depth_n = depth.clone()
    mid_depths = []
    dis_depths = []
    for i in range(b):
        # depthst = depth_sort[i][torch.where(depth_sort[i] > 0.4)[0]] # get the sorted depth list.
        max_depth = max_depths[i] # select the max_depth;
        # mid_depth = depthst[depthst.shape[0] // 2] # get the mid depth value.
        # max_depth - threshold.
        depth_larger_than_thre = torch.where(depth_sort[i] > 0.2)[0];
        if not depth_larger_than_thre.numel(): # don't contain any depth's value.
            min_depth = 0.2
        else:
            min_depth = depth_sort[i][depth_larger_than_thre][0] # at least larger than 30 cm, for kinect camera we used.
        
        # min_depth = depth_sort[i][depth_larger_than_thre][0]
        
        mid_depth = (max_depth + min_depth) / 2.0 # the depth normalization method.
        if dis is None: # the distance between max & min depth is the normalization distance.
            dis_depth = (max_depth - mid_depth) / 1.0 # to normalize the depth in the range [-1, 1]. e.g., [3, 5]. mid=4.0, dis=2.0
        else:
            dis_depth = dis
        # min_depth = max_depth - 2 * dis_depth
        if dis_depth <= 0:
            dis_depth += 1e-7;

        depth_n[i] = (depth[i] - mid_depth) / dis_depth # in [-x, 1]
        
        # dis_depth = (max_depth - min_depth) / 2
        # depth_n[i] = (depth[i] - mid_depth) / dis_depth
        # the clipp operation: robust to the distance value;
        if if_clip: # [-1, 1]
            depth_n[i] = torch.clamp(depth_n[i], -torch.max(depth_n[i]).item(), torch.max(depth_n[i]).item()) # clip to the fixed size

        mid_depths.append(mid_depth)
        dis_depths.append(dis_depth)
    
    if is_np:
        depth_n = depth_n.cpu().numpy()

    if resize:
        depth_n = depth_n[0]

    if return_prop:
        return depth_n, mid_depths, dis_depths

    return depth_n

def unnormalize_face_depth(opts, depth, mid_depths, dis_depths):
    b, c, h, w = depth.shape
    depth_un = depth.clone() # no replace's problem;
    # this unnormalize has a problem that -> the outlier value can not be reprojected to 0;
    for i in range(b):
        depth_un[i] = depth[i] * dis_depths[i] + mid_depths[i]

    return depth_un

def normalize_face_depth_(opts, depth, mid_depths, dis_depths=None):
    #depth [B,1,N_f]
    b, _, n_p = depth.shape
    depth_n = depth.clone() # no replace's problem;
    for i in range(b):
        if dis_depths is None: # when dis is None. default as 0.1 (width as 0.2);
            depth_n[i] = (depth[i] - mid_depths[i]) / opts.z_bbox_len
        else:
            depth_n[i] = (depth[i] - mid_depths[i]) / (dis_depths[i] + 1e-8)

    return depth_n


def get_gaussian_filter(kernel_size=3):
    if kernel_size == 3:
        gaussian_filter = np.array([[1, 2, 1],[2, 4, 2],[1, 2, 1]]) / 16.
    elif kernel_size == 5:
        gaussian_filter = np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4],
                                    [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], 
                                    [1, 4, 7, 4, 1]]) / 273.
    else:
        raise Exception('Not support gaussian kernel.')

    return gaussian_filter


def get_mean_filter(kernel_size=3):
    if kernel_size == 3:
        mean_filter = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]]) / 9.
    elif kernel_size == 5:
        mean_filter = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 
                                [1, 1, 1, 1, 1]]) / 25.
    else:
        raise Exception('Not support mean kernel.')

    return mean_filter


def filter(image, filter, kernel_size=5):
    assert isinstance(image, np.ndarray)
    im_dim = image.shape
                            
    if len(im_dim) == 3:
        assert im_dim[-1] == 3, 'image shape must be, i.e, [H, W, 3];'
        group_num = 3
        image_torch = torch.from_numpy(image).float().permute(2, 0, 1)[None,...] # [1, 3, H, W]
        filter = np.repeat(filter[None, None, :, :], group_num, 0) # generate gaussian filter.

    elif len(im_dim) == 2:
        group_num = 1
        image_torch = torch.from_numpy(image).float()[None, None, ...] # [1,1,H, W]
        filter = np.repeat(filter[None, None, :, :], group_num, 0)

    else:
        raise Exception('Not support shape of the image.')

    filter = Variable(torch.FloatTensor(filter))
    image_torch = F.conv2d(image_torch, filter, groups=group_num, padding=kernel_size // 2, stride=1, dilation=1)
    
    if len(im_dim) == 3:
        image_smooth = image_torch[0].permute(1, 2, 0).numpy()
    elif len(im_dim) == 2:
        image_smooth = image_torch[0,0].numpy()

    return image_smooth


def erode_border(image, depth, rect_width=2):
    rand = random.random()
    if rand > 0.75:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,8))
    elif rand <= 0.75 and rand > 0.5:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,7))
    elif rand <= 0.5 and rand > 0.25:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))

    # step1. dilate and erode operations.
    dilate = cv2.dilate(image, kernel)
    erode  = cv2.erode(image, kernel)
    # step2. calculate the difference.
    # Do substract of (erode, dilate) to get the border.
    border = cv2.absdiff(dilate, erode) * np.random.rand(*erode.shape) # random add noise at the erode border.
    th = np.random.randint(2.5, 8) #
    depth[border > th] = 0

    return depth


def add_holes(depth, mask, rate=0.1):
    # add holes in the depth regions;
    # depth: [H, W]; mask: [1, H, W];
    h, w = mask.shape[1:]
    mask_rep = np.copy(mask[0])
    area_idx = np.where(mask_rep == 1) # get where the area of the people. [idx0, idx1];
    num_idxs = area_idx[0].size
    num_noise_pixel = int(num_idxs * rate)
    selected_idx = np.random.choice(range(0, num_idxs), num_noise_pixel,  replace=False)
    num_noise_pixel_ = int(num_noise_pixel // 4) # approximate 1 / 4 's data here 

    # randomly add larger holes.
    ks_v1 = random.randint(2,5)
    ks_v2 = random.randint(2,5)
    for k in range(ks_v1):
        for l in range(ks_v2):
            i = np.clip(area_idx[0][selected_idx[:num_noise_pixel_]] + k, 0, h - 1)
            j = np.clip(area_idx[1][selected_idx[:num_noise_pixel_]] + l, 0, w - 1)
            depth[i, j] = 0

    depth[area_idx[0][selected_idx[num_noise_pixel_:]], area_idx[1][selected_idx[num_noise_pixel_:]]] = 0
    return depth


def add_gaussian_noise_inregion(depth, noise_sigma = [0.6, 0.6, 0.6]):
    # first define a 1 / 4 map size， then the gaussian noise is added, and the map is upsampled to original size.
    dshape = depth.shape
    # three patent noises 1. resolution 1 / 2;  2. resolution 1 / 4; 3. original resolution.
    noise_r1 = np.random.randn(dshape[0], dshape[1]) * noise_sigma[0]
    noise_r2 = np.random.randn(dshape[0] // 2, dshape[1] // 2) * noise_sigma[1]
    noise_r4 = np.random.randn(dshape[0] // 4, dshape[1] // 4) * noise_sigma[2]
    
    noise_r2 = cv2.resize(noise_r2, dshape, interpolation=cv2.INTER_LANCZOS4) # upsampling to a smoother gaussian noise.
    noise_r4 = cv2.resize(noise_r4, dshape, interpolation=cv2.INTER_LANCZOS4) # upsampling to a smoother gaussian noise.
    return noise_r1 + noise_r2 + noise_r4 + depth


def add_kinect_noise(depth, normal_map):
    # input depth: cm.
    # the indoor scene TOF noise simulation, (depth: [512, 512]; normal_map: [3, 512, 512]);
    # z_sigma(z, \theta) = 1.5 - 0.5 * z + 0.3 z^2 + 0.1 z^{3/2} \frac{\theta^2}{(\pi / 2 - \theta)^2}
    depth_m = depth / 100
    c,h,w = normal_map.shape
    # in camera coordinate, the basic axis is the direction.
    basic_axis = np.array([[0,0,1]]) # 
    normal_map = (np.copy(normal_map) - 0.5) * 2. # [-1,1]
    cos = (basic_axis @ normal_map.reshape(c,-1)).reshape(h, w)
    angle = np.arccos(cos) # [H, W];
    sigma_z_theta_mm = 1.5 - 0.5 * depth_m + 0.3 * depth_m**2 + 0.1 * depth_m**(3/2) * (angle**2 / (np.pi / 2 - angle + 1e-7)**2)
    sigma_z_theta_cm = sigma_z_theta_mm / 10

    return depth + np.random.randn(h, w) * sigma_z_theta_cm


def add_gaussian_shifts(depth, std=1/2.0):

    rows, cols = depth.shape 
    gaussian_shifts = np.random.normal(0, std, size=(rows, cols, 2))
    gaussian_shifts = gaussian_shifts.astype(np.float32)

    # creating evenly spaced coordinates  
    xx = np.linspace(0, cols-1, cols)
    yy = np.linspace(0, rows-1, rows)

    # get xpixels and ypixels 
    xp, yp = np.meshgrid(xx, yy)

    xp = xp.astype(np.float32)
    yp = yp.astype(np.float32)

    xp_interp = np.minimum(np.maximum(xp + gaussian_shifts[:, :, 0], 0.0), cols)
    yp_interp = np.minimum(np.maximum(yp + gaussian_shifts[:, :, 1], 0.0), rows)

    depth_interp = cv2.remap(depth, xp_interp, yp_interp, cv2.INTER_LINEAR)

    return depth_interp


def filterDisp(disp, dot_pattern_, invalid_disp_):

    size_filt_ = 9

    xx = np.linspace(0, size_filt_-1, size_filt_)
    yy = np.linspace(0, size_filt_-1, size_filt_)

    xf, yf = np.meshgrid(xx, yy)

    xf = xf - int(size_filt_ / 2.0)
    yf = yf - int(size_filt_ / 2.0)

    sqr_radius = (xf**2 + yf**2)
    vals = sqr_radius * 1.2**2

    vals[vals==0] = 1
    weights_ = 1 /vals

    fill_weights = 1 / ( 1 + sqr_radius)
    fill_weights[sqr_radius > 9] = -1.0 

    disp_rows, disp_cols = disp.shape
    dot_pattern_rows, dot_pattern_cols = dot_pattern_.shape

    lim_rows = np.minimum(disp_rows - size_filt_, dot_pattern_rows - size_filt_)
    lim_cols = np.minimum(disp_cols - size_filt_, dot_pattern_cols - size_filt_)

    center = int(size_filt_ / 2.0)

    window_inlier_distance_ = 0.1

    out_disp = np.ones_like(disp) * invalid_disp_

    interpolation_map = np.zeros_like(disp)

    for r in range(0, lim_rows):

        for c in range(0, lim_cols):

            if dot_pattern_[r+center, c+center] > 0:
                                
                # c and r are the top left corner 
                window  = disp[r:r+size_filt_, c:c+size_filt_] 
                dot_win = dot_pattern_[r:r+size_filt_, c:c+size_filt_]
  
                valid_dots = dot_win[window < invalid_disp_]

                n_valids = np.sum(valid_dots) / 255.0 
                n_thresh = np.sum(dot_win) / 255.0 

                if n_valids > n_thresh / 1.2: 

                    mean = np.mean(window[window < invalid_disp_])

                    diffs = np.abs(window - mean)
                    diffs = np.multiply(diffs, weights_)

                    cur_valid_dots = np.multiply(np.where(window<invalid_disp_, dot_win, 0), 
                                                 np.where(diffs < window_inlier_distance_, 1, 0))

                    n_valids = np.sum(cur_valid_dots) / 255.0

                    if n_valids > n_thresh / 1.2: 
                    
                        accu = window[center, center] 

                        assert(accu < invalid_disp_)

                        out_disp[r+center, c + center] = round((accu)*8.0) / 8.0

                        interpolation_window = interpolation_map[r:r+size_filt_, c:c+size_filt_]
                        disp_data_window     = out_disp[r:r+size_filt_, c:c+size_filt_]

                        substitutes = np.where(interpolation_window < fill_weights, 1, 0)
                        interpolation_window[substitutes==1] = fill_weights[substitutes ==1 ]

                        disp_data_window[substitutes==1] = out_disp[r+center, c+center]

    return out_disp


def aug_depth(depth, mask, kinect_pattern, K, holes_rate=0.006):
    h, w = depth.shape
    scale_factor = 100 # transform to m.
    assert K[0,0] == K[1,1], 'error focal length.'
    focal_length = K[0,0].item()
    baseline_m = 1
    invalid_disp_ = 99999999.9

    depth_m = np.copy(depth) / scale_factor
    depth_interp = add_gaussian_shifts(depth_m)

    disp_= focal_length * baseline_m / (depth_interp + 1e-10)
    depth_f = np.round(disp_ * 8.0) / 8.0

    out_disp = filterDisp(depth_f, kinect_pattern, invalid_disp_)

    depth_aug = focal_length * baseline_m / out_disp
    depth_aug[out_disp == invalid_disp_] = 0 
    depth_aug *= scale_factor
    depth_aug += np.random.normal(size=(h,w)) * (1/6.0) + 0.5 # add gaussian noise.

    # add holes.
    if random.random() > 0.4:
        depth_aug = add_holes(depth_aug, mask, rate=holes_rate)
    
    depth_padding = np.pad(depth_aug, (1,1), mode='edge') # padding the depth map to : [3, H + 2, W + 2];

    return depth_aug


def aug_depth_v2(depth, normal_map, mask, noise_sigma=[0.6, 0.6, 0.6], border_erode_width=2, holes_rate=0.006):
    # the input depth is added noise without normalization.
    # normal_map: [H, W];
    depth_aug = np.copy(depth) # the augmented depth images;
    
    # step1. smooth using Gaussian filter
    gfilter3, gfilter5 = get_gaussian_filter(kernel_size=3), get_gaussian_filter(kernel_size=5)
    # smooth filter.
    depth_aug = filter(depth_aug, gfilter5, kernel_size=5)
    depth_aug = filter(depth_aug, gfilter5, kernel_size=5)

    # step2.add kinect noise.
    # depth_aug = add_kinect_noise(depth_aug, normal_map)
    # depth_aug = np.clip(depth_aug, 0, np.max(depth) + 100)

    # step3. erode the border of the depth image with random morph rects.
    depth_aug = erode_border(depth, depth_aug, rect_width=2)
    # depth_aug *= mask.numpy()[0]

    # step3. adding gaussian noise in regions.
    depth_aug = add_gaussian_noise_inregion(depth_aug, noise_sigma)

    # step4. randomly add some holes.
    if random.random() > 0.4:
        depth_aug = add_holes(depth_aug, mask, rate=holes_rate)

    return depth_aug

def get_grid(shape, int_K):
    B, height, width, C = shape
    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32),
                           torch.arange(0, width, dtype=torch.float32)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
    xyz = int_K @ xyz # [B, 3, H*W]

    return xyz.view(B, 3, height, width)


# def get_points_coordinate(depth, int_K):
#     B, height, width, C = depth.shape
#     y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32),
#                            torch.arange(0, width, dtype=torch.float32)])
#     y, x = y.contiguous(), x.contiguous()
#     y, x = y.view(height * width), x.view(height * width)
#     xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
#     xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
#     xyz = int_K @ xyz # [B, 3, H*W]
#     depth_xyz = xyz * depth.view(1, 1, -1)  # [B, 3, Ndepth, H*W]
    
#     return depth_xyz.view(B, 3, height, width)


def denoise_depth(opts, depths, normal_est: torch.Tensor, K: torch.Tensor):
    # step 1. init depth;
    # depths[depths <= opts.z_size - opts.bbox_max[0]] = opts.z_size
    # depths[depths < 0] = 0
    depths = ndimage.median_filter(depths, 5) # to filling the black holes.

    depths_normalized = normalize_depth(opts, depths)
    depths_normalized = cv2.bilateralFilter(depths_normalized.astype(np.float32), 5, 0.2, 0.2)
    depths = unnormalize_depth(opts, depths_normalized)
    
    init_depth = np.copy(depths)
    init_depth = torch.from_numpy(init_depth)[None, ..., None]
    valid_depth = init_depth > 0 # normal map's valid values.
    
    # step 2. get normal map (estimate by network).
    normal_torch = normal_est.permute(1,2,0)[None, ...] # [3, h, w] -> [B, h,w,3], ranged in [-1,1]
    b, h, w, c = normal_torch.shape
    
    # step 3. get grids. and depth data.
    K_inv = torch.inverse(K)[None, ...] # [B, 3, 3]
    grid = get_grid(normal_torch.shape, K_inv)
    points = (grid.view(b, 3, -1) * init_depth.view(1, 1, -1)).view(b, 3, h, w)
    grid_patch = F.unfold(grid, kernel_size=3, stride=1, padding=2, dilation=2)
    grid_patch = grid_patch.view(1, 3, 9, h, w)
    points_matrix = F.unfold(points, kernel_size=3, stride=1, padding=2, dilation=2)
    matrix_a = points_matrix.view(1, 3, 9, h, w) # (B, 3, 25, H, W)
    _, _, depth_data = torch.chunk(matrix_a.permute(0, 3, 4, 2, 1), chunks=3, dim=4) # (B, H, W, 25, 1)

    # step 4. get normal neighbourhood matrix
    norm_matrix = F.unfold(normal_torch.permute(0, 3, 1, 2), kernel_size=3, stride=1, padding=2, dilation=2)
    matrix_c = norm_matrix.view(1, 3, 9, h, w)
    matrix_c = matrix_c.permute(0, 3, 4, 2, 1)  # (B, H, W, 25, 3)
    normal_torch_expand = normal_torch.unsqueeze(-1)

    angle = torch.matmul(matrix_c, normal_torch_expand)
    valid_condition = torch.gt(angle, 1e-5)
    valid_condition_all = valid_condition.repeat(1, 1, 1, 1, 3)
    tmp_matrix_zero = torch.zeros_like(angle)
    valid_angle = torch.where(valid_condition, angle, tmp_matrix_zero)

    lower_matrix = torch.matmul(matrix_c, grid.permute(0, 2, 3, 1).unsqueeze(-1))
    condition = torch.gt(lower_matrix, 1e-5)
    tmp_matrix = torch.ones_like(lower_matrix)
    lower_matrix = torch.where(condition, lower_matrix, tmp_matrix)
    lower = torch.reciprocal(lower_matrix)

    valid_angle = torch.where(condition, valid_angle, tmp_matrix_zero)
    upper = torch.sum(torch.mul(matrix_c, grid_patch.permute(0, 3, 4, 2, 1)), dim=4)
    ratio = torch.mul(lower, upper.unsqueeze(-1))
    estimate_depth = torch.mul(ratio, depth_data)

    valid_angle = torch.mul(valid_angle, torch.reciprocal((valid_angle.sum(dim=(3, 4), keepdim=True)+1e-5).repeat(1, 1, 1, 9, 1)))
    depth_stage1 = torch.mul(estimate_depth, valid_angle).sum(dim=(3, 4))
    depth_stage1 = depth_stage1.squeeze().unsqueeze(2).numpy()[..., 0]

    # print(depth_stage1.shape, np.max(depth_stage1), np.min(depth_stage1))
    # plt.imshow(depth_stage1)
    # plt.show()
    # exit()
    return depth_stage1


# denoise the depth.
def denoise_depth_v1(opts, depths):
    # step 1. median filters to fill the black holes in the depths map;
    depths[depths < 0] = 0
    depths = ndimage.median_filter(depths, 5) # to filling the black holes.
    # depths = ndimage.median_filter(depths, 3) # to filling the black holes.

    # step 2. smooth the depth map.
    depths_normalized = normalize_depth(opts, depths)
    # print(np.max(depths_normalized), np.min(depths_normalized))
    depths_normalized = cv2.bilateralFilter(depths_normalized.astype(np.float32), 7, 0.5, 0.5)
    # print(np.max(depths_normalized), np.min(depths_normalized))
    depths = unnormalize_depth(opts, depths_normalized)

    # mfilter3 = get_gaussian_filter(kernel_size=5)
    # depths = filter(depths, mfilter3, kernel_size=5)
    
    return depths


def depth2normal(depth): # transform the depths to normal maps.
    # step 1, padding the depth map.
    h, w = depth.shape
    depth_padding = np.pad(depth, (1,1), mode='edge') # padding the depth map to : [3, H + 2, W + 2];

    # step 2, calculate the middle difference.
    dzdx = - (depth_padding[2:, 1:-1] - depth_padding[:-2, 1:-1]) * 0.5 # [h, w]'s map.
    dzdy = - (depth_padding[1:-1, 2:] - depth_padding[1:-1, :-2]) * 0.5
    dz = np.ones([h, w]) # assign one for the z axis 's value.
    n = np.sqrt(dzdx ** 2 + dzdy ** 2 + dz ** 2) + 1e-7
    
    # to [0, 1] map.
    dx = (dzdx / n) * 0.5 + 0.5  
    dy = (dzdy / n) * 0.5 + 0.5
    dz = (dz   / n) * 0.5 + 0.5
    
    normal = np.stack([dx, dy, dz], axis=0) # [3, h, w]
            
    # plt.imshow(np.transpose(normal, (1, 2, 0)))
    # plt.show()
    # exit()

    return normal

def pseudo(depth):
    # depth: [B,1,H,W]
    H, W = depth.shape
    output = np.zeros([H, W, 3], dtype=np.uint8)
    
    gray = (depth * 255).astype(np.uint8)
    output = cv2.applyColorMap(gray, cv2.COLORMAP_RAINBOW)

    return output


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
    
    def proj_persp(self, points, K, R, T): # divide by / depth.
        pts = R @ points + T # [B, 3, N]
        # To screen coordinate
        depth = pts[:, -1:, :].clone() # [B, 1, N]
        pts /= depth # [x / d, y / d, 1.];
        pts_screen = K @ pts # [f1 * x / d + f_x, f2 * y / d + f_y]
        # in dataset, the intrinsic matrix has been normalized to [f1 / fx, f2 / fy, 1], -> [-1,1]
        return (pts_screen[:, :2, :] - self.opts.load_size // 2) / (self.opts.load_size // 2), depth


def projection3D(opts, points: torch.Tensor, K: torch.Tensor, RT: torch.Tensor):
    # project 3D points to 2D xy, all data are torch tensor;
    R = RT[:3, :3]; T = RT[:3, -1:]
    
    def proj_ortho(points, K, R, T):
        pts = R @ points + T # [3, N_p]
        depth = pts[-1:, :].clone() # [1, N]
        pts_screen = K @ pts  # [3, N_p];
        return pts_screen[:2, :], depth

    def proj_persp(points,  K, R, T, thres=0.01):
        pts = R @ points + T # [3, N_p]
        depth = pts[-1:, :].clone() # [1, N]
        invalid = depth <= thres # [1,N]
        depth_ = depth.clone()
        depth_[invalid] = 1.0
        
        pts /= depth_
        pts_screen = K @ pts  # [3, N_p];
        grids = pts_screen[:2, :] # [0 ~ h, 0 ~ w]
        grids[invalid.repeat(2,1)] = -1
         
        return grids, depth
    
    if opts.project_mode == 'perspective':
        return proj_persp(points, K, R, T)
    elif opts.project_mode == 'ortho':
        return proj_ortho(points, K, R, T)


def save_samples_truncted_prob(output_dir, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    '''
    save_path = os.path.join(output_dir, 'points.ply')
    r = (prob > 0.5).reshape([-1, 1]) * 255
    b = (prob > 0.5).reshape([-1, 1]) * 255
    g = (prob > 0.5).reshape([-1, 1]) * 255

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(save_path,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )


if __name__ == '__main__':
    depth_path = '/home/dz/my_Rendering/pifu_dataset/render_dataset2/DEPTH/Male_08/30_0.npy'
    mask_path = '/home/dz/my_Rendering/pifu_dataset/render_dataset2/MASK/Male_08/30_0.png'
    
    depth = np.load(depth_path, allow_pickle=True)
    mask  = transforms.ToTensor()(Image.open(mask_path).convert('L')).float()

    depth_aug = aug_depth(depth, mask)
    # the depth map's average metrics is opts.z_size. (200 mean)
    max_depth, min_depth = np.max(depth_aug), np.min(depth_aug)
    depth_aug = (depth_aug - min_depth) / (max_depth - min_depth)
    
    plt.imshow(depth_aug)
    plt.show()