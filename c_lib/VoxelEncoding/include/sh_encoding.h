#pragma once

#include "cuda_helper.h"

torch::Tensor sh_encode_forward(
    torch::Tensor inputs, 
    const uint32_t degree, 
    torch::optional<torch::Tensor> dy_dx,
    const int device
);

torch::Tensor sh_encode_backward(
    torch::Tensor grad, 
    torch::Tensor inputs, 
    const uint32_t degree, 
    torch::Tensor dy_dx,
    const int device
);