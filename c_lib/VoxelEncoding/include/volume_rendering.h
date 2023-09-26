#pragma once

#include "cuda_helper.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> volume_rendering_training_forward(
    torch::Tensor sampled_depth,
    torch::Tensor sampled_sort_idx,
    torch::Tensor sigmas,
    torch::Tensor rgbs,
    const int device,
    const float t_thresh,
    const bool support_occ
);

std::tuple<torch::Tensor, torch::Tensor> volume_rendering_training_backward(
    torch::Tensor grad_output_rgbs, torch::Tensor grad_output_depths, torch::Tensor grad_weight_sums, torch::Tensor grad_output_alphas,
    // torch::Tensor grad_rgbs, torch::Tensor grad_sigmas,
    torch::Tensor output_rgbs, torch::Tensor output_depths, torch::Tensor output_ws, torch::Tensor output_alphas, 
    torch::Tensor sampled_depth, // [B, N_ray, N_sampled], the sampled depths are sorted values.
    torch::Tensor sampled_sort_idx, // [B, N_ray, N_sampled], the sorted idx of the sampled z;
    torch::Tensor sigmas, // [B, N_ray, N_sampled, 1]
    torch::Tensor rgbs, // [B, N_ray, N_sampled, 3]
    const int device, const float t_thresh, const bool support_occ
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> volume_rendering_occ_forward(
    torch::Tensor sampled_depth, // [B, N_ray, N_sampled], the sampled depths are sorted values.
    torch::Tensor occs, // [B, N_ray, N_sampled, 1]
    torch::Tensor feats, // [B, N_ray, N_sampled, feat_dim], can be rgb data or feats.
    const int device, const float t_thresh
);