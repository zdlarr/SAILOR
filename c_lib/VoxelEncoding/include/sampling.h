#pragma once

#include "cuda_helper.h"

torch::Tensor rays_calculating(
    torch::Tensor cam_intr, // [B, N_v, 3, 3]
    torch::Tensor cam_R, // [B, N_v, 3, 3]
    torch::Tensor cam_T, // [B, N_v, 3]
    const int load_size, // 512
    const int device
);

torch::Tensor rays_calculating_parallel(
    torch::Tensor cam_intr, // [B, N_v, 3, 3]
    torch::Tensor cam_R, // [B, N_v, 3, 3]
    torch::Tensor cam_T, // [B, N_v, 3]
    const int load_size, // 512
    const int device,
    const int num_gpus
);

torch::Tensor rays_selecting( // sample N_rays per view from the ray_ori_dir.
    torch::Tensor ray_ori_dir,  // [B, N_v, H, W, 6]
    torch::Tensor sampled_xy, // [B, N_v, N_rays, 2]
    const int device
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rgbdv_selecting( // sample RGBDs per view from the target rgbds, when depth is 0 (invalid.)
    torch::Tensor target_rgbs,    // [B, N_v, H, W, 3]
    torch::Tensor target_normals,    // [B, N_v, H, W, 3]
    torch::Tensor target_depths,  // [B, N_v, H, W, 1]
    torch::Tensor sampled_xy,     // [B, N_v, N_rays, 2]
    const int device
);

// return UDF, 
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> udf_calculating( // calculate the UDF values between queried points and pcds.
    torch::Tensor ray_ori,    // [B, N_rays, 3]
    torch::Tensor ray_dir,  // [B, N_rays, 3]
    torch::Tensor sampled_depths, // [B, N_rays, N_sampled, 1]
    torch::Tensor pcds, // [B, N_points, 3];
    const int device // the device for cuda.
);

// return UDF, 
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> udf_calculating_v2( // calculate the UDF values between queried points and pcds.
    torch::Tensor pcds0, // [B, N_p0, 3];
    torch::Tensor pcds1, // [B, N_p1, 3];
    const int device
);

