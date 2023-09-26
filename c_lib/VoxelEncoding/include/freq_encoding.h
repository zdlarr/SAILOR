#pragma once

#include "cuda_helper.h"

torch::Tensor freq_encode_forward(
    torch::Tensor input_tensor,
    const int deg, 
    const int device
);

torch::Tensor freq_encode_backward(
    torch::Tensor grad_output,
    torch::Tensor inputs,
    torch::Tensor outputs,
    const int deg,
    const int device
);