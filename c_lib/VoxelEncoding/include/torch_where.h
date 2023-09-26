#pragma once

#include "cuda_helper.h"

std::vector<torch::Tensor> tensor_selection(
    torch::Tensor input_tensor, // [B, C, H, W]
    torch::Tensor pha, // [B, 1, H, W]
    const float t_thresh, 
    const int device
);

std::vector<torch::Tensor> bbox_mask(
    torch::Tensor masks, // [B,1,H,W]
    const int device
);