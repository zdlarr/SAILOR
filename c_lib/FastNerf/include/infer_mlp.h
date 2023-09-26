#pragma once

#include "cuda_helper.h"
    
// the fast infer rendering，
// 1. forward_density.
// 2. forward_color.
// 3. predict the upsampling weights;

// output the points' density & rgbs, rgb_features;
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> infer_fast_rendering(
    // inputs, now are all [N_points, C]
    torch::Tensor sorted_dists, // [B, N_rays, N_points]
    //
    torch::Tensor points_z, // [B, N_view, N_rays, N_points, 1]
    torch::Tensor points_tpsdf, // [B, N_view, N_rays, N_points, 1]
    torch::Tensor sampled_feats_geo, // [B, N_view, N_rays, N_points, 16]
    torch::Tensor sampled_rgbs, // [B, N_view', N_rays, N_points, 3]
    torch::Tensor sampled_feats_rgb, // [B, N_view', N_rays, N_points, 16]
    torch::Tensor rays_dir, // [B, N_view', N_rays, N_points, 3]
    // the parameters of the MLP;
    torch::Tensor net_params_density, // [N_inputs*W0+B0 + W0*W1+B1 + ....] one dim tensor;
    torch::Tensor net_params_color, // [N_inputs*W0+B0 + W0*W1+B1 + ....] one dim tensor;
    // properties;
    const int batch_size, const int n_rays, const int n_sampled, // e.g., 1, 1024, 64
    const int geo_num_views, const int tex_num_views, // e.g., 8, 4
    const int device
);

torch::Tensor infer_density_mlp(
    // now all data are [N_points, C]
    torch::Tensor points_z, // [B, N_view, N_rays, N_points, 1]
    torch::Tensor points_tpsdf, // [B, N_view, N_rays, N_points, 1]
    torch::Tensor sampled_feats_geo, // [B, N_view, N_rays, N_points, 16]
    // mlp
    torch::Tensor net_params_density,
    const int batch_size, const int num_points, const int num_views,
    const int device
);

// predicts upsampling features' fusion weights; [N_rays, 3]
torch::Tensor infer_upsampling(
    torch::Tensor feats_neighbors, // [16 + 3 + 1] * 2; 40 dim
    torch::Tensor ray_rgb_feats, // [N_rays, 30]
    torch::Tensor net_params, // [N_inputs*W0+B0 + W0*W1+B1 + ....] one dim tensor;
    const int device
);