#pragma once

#include "cuda_helper.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
fusion_cuda_integrate(
    // torch::Tensor tsdf_vol_torch, // [B, H, W, K]
    // torch::Tensor weight_vol_torch, // [B, H, W, K]
    // torch::Tensor color_vol_torch, // [B, H, W, K]
    torch::Tensor occupied_vol_torch, // [B, H, W, K]
    torch::Tensor num_occ_voxel_torch, // [B]
    // torch::Tensor vol_origin_torch, // [B, 3], (h,w,k) three min vol_dim;
    // torch::Tensor vol_bbox_torch, // [B, 3, 2], (h,w,k)
    // torch::Tensor voxel_size_torch, // [B, 3, 2]
    torch::Tensor cam_intr_torch, // [B, N, 3, 3]
    // torch::Tensor cam_intr_inv_torch, // [B, N, 3, 3]
    torch::Tensor cam_pose_torch, // [B, N, 3, 4]
    torch::Tensor cam_r_torch, // [B, N, 3, 3]
    torch::Tensor cam_t_torch, // [B, N, 3]
    torch::Tensor color_ims_torch, torch::Tensor depth_ims_torch, torch::Tensor mask_ims_torch, // [B, N, H, W]
    const float obs_weight, const float th_bbox, const float trunc_weight, // default as 5.0
    const float tsdf_th_low, const float tsdf_th_high, const int device
);


torch::Tensor get_origin_bbox(
    torch::Tensor depth_ims_torch,
    torch::Tensor cam_intr_torch, // [B, N, 3, 3]
    // torch::Tensor cam_intr_inv_torch, // [B, N, 3, 3]
    // torch::Tensor cam_pose_torch, // [B, N, 3, 4]
    torch::Tensor cam_R_torch, // [B, N, 3, 3]
    torch::Tensor cam_T_torch, // [B, N, 3]
    const float th_bbox, //default as 0.01m
    const int device
);

std::vector<torch::Tensor> get_center_xyz(
    torch::Tensor depth_ims_torch,
    torch::Tensor cam_intr_torch, // [B, N, 3, 3]
    // torch::Tensor cam_intr_inv_torch, // [B, N, 3, 3]
    // torch::Tensor cam_pose_torch, // [B, N, 3, 4]
    torch::Tensor cam_R_torch, // [B, N, 3, 3]
    torch::Tensor cam_T_torch, // [B, N, 3]
    const float th_bbox, //default as 0.01m
    const int device
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
fusion_cuda_integrate_refined(
    torch::Tensor occupied_vol_torch, // [B, H,W,K]
    torch::Tensor num_occ_voxel_torch, // [B]
    // torch::Tensor vol_origin_torch, // [B, 3, 2]
    // torch::Tensor vol_bbox_torch, // [B, 3, 2]
    // torch::Tensor voxel_size_torch, // [B, 1]
    torch::Tensor cam_intr_torch, // [B, N, 3, 3]
    // torch::Tensor cam_intr_inv_torch, // [B, N, 3, 3]
    torch::Tensor cam_pose_torch, // [B, N, 3, 4]
    torch::Tensor cam_R_torch, // [B, N, 3, 3]
    torch::Tensor cam_T_torch, // [B, N, 3]
    // torch::Tensor color_ims_torch, // [B, N, H, W]
    torch::Tensor depth_ims_torch,  // [B, N, H, W]
    // torch::Tensor mask_ims_torch, // [B, N, H, W]
    const float obs_weight, 
    const float th_bbox, //default as 0.01m
    const float tsdf_th_low, const float tsdf_th_high, // default as 6;
    const int device
);