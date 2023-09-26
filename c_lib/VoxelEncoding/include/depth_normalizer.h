#pragma once

#include "cuda_helper.h"

// return the normalized depth maps, depth mid, depth distance;
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> depth_normalize(
    torch::Tensor depths,
    const float valid_min_depth,
    const float valid_max_depth,
    const float user_dist,
    const bool multi_mask, // whether multiply by the masks.
    const bool divided2, // whether divided by 2.
    const int   device
);