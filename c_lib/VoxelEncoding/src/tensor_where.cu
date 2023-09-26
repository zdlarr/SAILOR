
#include <math.h>
#include <vector>
#include <stdint.h>
#include <assert.h>

#include "../include/torch_where.h"
#include <string>
#include <stdio.h>
#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>

__global__ void tensor_selection_kernel(
    const float * pha, // [B, 1, H, W]
    float * output_tensor, // [B, C, H, W]
    float * output_mask, // [B, 1, H, W]
    const int batch_size, 
    const int channel_dim,
    const int height,
    const int width,
    const float thresh
) {
    #pragma unroll
    CUDA_KERNEL_LOOP(k, batch_size*channel_dim*height*width) {
        uint32_t batch_idx   = (uint32_t) (k / (channel_dim*height*width));
        uint32_t channel_idx = (uint32_t) (k - batch_idx*channel_dim*height*width) / (height*width);
        uint32_t h_idx       = (uint32_t) (k - batch_idx*channel_dim*height*width - channel_idx*height*width) / width;
        uint32_t w_idx       = (uint32_t) k - batch_idx*channel_dim*height*width - channel_idx*height*width - h_idx*width;

        // get the mask value;
        uint32_t pha_idx = batch_idx*1*height*width + h_idx*width + w_idx;
        float mask_value = pha[pha_idx];
        // if invalid value, set zeros (mask & inputs.)
        if (mask_value < thresh) { output_tensor[k] = 0.0; output_mask[pha_idx] = 0.0; }
        else {output_mask[pha_idx] = 1.0;}
    }
}

// approx. < 1ms
// using mask to filter tensors.
std::vector<torch::Tensor> tensor_selection(
    torch::Tensor input_tensor, // [B, C, H, W]
    torch::Tensor pha, // [B, 1, H, W]
    const float t_thresh, 
    const int device
) {
    cudaSetDevice(device); // on GPU device.
    CUDA_CHECK_ERRORS();

    cudaStream_t curr_stream = at::cuda::getCurrentCUDAStream();

    const uint32_t batch_size  = pha.size(0);
    const uint32_t channel_dim = input_tensor.size(1);
    const uint32_t height = pha.size(2);
    const uint32_t width  = pha.size(3);

    torch::Tensor output_tensor = input_tensor.clone();
    torch::Tensor mask          = pha.clone();
    
    int block_size, grid_size, min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, tensor_selection_kernel, 0, 0);
    grid_size = (batch_size * channel_dim * height * width + block_size - 1) / block_size;
    
    tensor_selection_kernel <<< grid_size, block_size, 0, curr_stream >>> (
        pha.contiguous().data_ptr<float>(),
        output_tensor.contiguous().data_ptr<float>(),
        mask.contiguous().data_ptr<float>(),
        batch_size, channel_dim, height, width, t_thresh
    );
    CUDA_CHECK_ERRORS();

    return {output_tensor, mask};
}

__global__ void bbox_mask_kernel(
    const float * masks, // [b,1,h,w]
    int * bbox_min, // [b,2]
    int * bbox_max, // [b,2]
    const int batch_size,
    const int height,
    const int width
) {
    #pragma unroll
    CUDA_KERNEL_LOOP(k, batch_size*height*width) {
        int b_idx = (int) k / (height*width);
        int h_idx = (int) (k - b_idx*height*width) / width;
        int w_idx = (int) k - b_idx*height*width - h_idx*width;

        if (masks[k] > 0) { // in mask's region.
            atomicMin(bbox_min + b_idx*2 + 0, h_idx); // h_idx
            atomicMin(bbox_min + b_idx*2 + 1, w_idx); // w_idx
            atomicMax(bbox_max + b_idx*2 + 0, h_idx); // h_idx
            atomicMax(bbox_max + b_idx*2 + 1, w_idx); // w_idx
        }
    }
}

std::vector<torch::Tensor> bbox_mask(
    torch::Tensor masks, // [B,1,H,W]
    const int device
) {
    cudaSetDevice(device);
    CUDA_CHECK_ERRORS();

    cudaStream_t curr_stream = at::cuda::getCurrentCUDAStream();

    const uint32_t batch_size = masks.size(0);
    const uint32_t height = masks.size(2);
    const uint32_t width  = masks.size(3);

    torch::Tensor bbox_min = torch::full({batch_size, 2},  99999, masks.options()).to(torch::kInt);
    torch::Tensor bbox_max = torch::full({batch_size, 2}, -99999, masks.options()).to(torch::kInt);

    int block_size, grid_size, min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, bbox_mask_kernel, 0, 0);
    grid_size = (batch_size * height * width + block_size - 1) / block_size;

    bbox_mask_kernel <<< grid_size, block_size, 0, curr_stream >>> (
        masks.contiguous().data_ptr<float>(),
        bbox_min.contiguous().data_ptr<int>(),
        bbox_max.contiguous().data_ptr<int>(),
        batch_size, height, width
    );

    return {bbox_min, bbox_max};
}