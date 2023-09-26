#pragma once

#include "cuda_helper.h"

torch::Tensor depth2color(
    torch::Tensor depths, torch::Tensor masks, // [2,1,h,w]
    torch::Tensor kc, torch::Tensor kd, torch::Tensor rt_d2c, // [2,3,3(4)]
    const int tar_h, const int tar_w, const int dsize, const float d_thresh,
    const int device
);