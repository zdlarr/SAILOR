#include <math.h>
#include <vector>
#include <stdint.h>
#include <assert.h>

// #include <omp.h>
#include "../include/cuda_helper.h"
#include <ATen/cuda/CUDAContext.h>
#include "../include/depth_normalizer.h"
// #include "./intersection.cpp"
#include <string>
#include <stdio.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>


template <typename scalar_t>
__device__ static scalar_t atomicMin(scalar_t* address, scalar_t val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return (scalar_t) __int_as_float(old);
}

template <typename scalar_t>
__device__ static scalar_t atomicMax(scalar_t* address, scalar_t val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return (scalar_t) __int_as_float(old);
}

template <typename scalar_t>
__global__ void depth_normalization_kernel (
    const scalar_t * __restrict__ depths, // [B, N_view, H, W]
    scalar_t * min_depths, // [B, N_view, 1]
    scalar_t * max_depths, // [B, N_view, 1]
    const scalar_t valid_min_depth,
    const scalar_t valid_max_depth,
    const uint32_t batch_size, const uint32_t num_views, 
    const uint32_t im_h, const uint32_t im_w
) {
    CUDA_KERNEL_LOOP(k, batch_size*num_views*im_h*im_w) {
        // parallel for each pixel.
        uint32_t batch_idx = (uint32_t) k / (num_views*im_h*im_w);
        uint32_t view_idx  = (uint32_t) (k - batch_idx*num_views*im_h*im_w) / (im_h*im_w);
        uint32_t h_idx     = (uint32_t) (k - batch_idx*num_views*im_h*im_w - view_idx*im_h*im_w) / im_w;
        uint32_t w_idx     = (uint32_t) k - batch_idx*num_views*im_h*im_w - view_idx*im_h*im_w - h_idx*im_w;
        
        // get the pixel depth value. (uint in m);
        scalar_t depth_value = depths[k];
        // when d < min_d or d > max_d, not valid depth.
        if (depth_value < valid_min_depth || depth_value > valid_max_depth) continue;

        // atomic compare the depth values, get the max & min depth for each view.
        atomicMin<scalar_t>(min_depths + batch_idx*num_views*1+view_idx*1, depth_value);
        atomicMax<scalar_t>(max_depths + batch_idx*num_views*1+view_idx*1, depth_value);
    }
}


template <typename scalar_t>
__global__ void get_mask_depths_kernel (
    const scalar_t * __restrict__ depths, // [B, N_view, H, W]
    scalar_t * mask_depths, // [B, N_view, 1]
    const scalar_t valid_min_depth,
    const scalar_t valid_max_depth,
    const uint32_t batch_size, const uint32_t num_views, 
    const uint32_t im_h, const uint32_t im_w
) {
    CUDA_KERNEL_LOOP(k, batch_size*num_views*im_h*im_w) {
        // parallel for each pixel.
        // uint32_t batch_idx = (uint32_t) k / (num_views*im_h*im_w);
        // uint32_t view_idx  = (uint32_t) (k - batch_idx*num_views*im_h*im_w) / (im_h*im_w);
        // uint32_t h_idx     = (uint32_t) (k - batch_idx*num_views*im_h*im_w - view_idx*im_h*im_w) / im_w;
        // uint32_t w_idx     = (uint32_t) k - batch_idx*num_views*im_h*im_w - view_idx*im_h*im_w - h_idx*im_w;
        
        // get the pixel depth value. (uint in m);
        scalar_t depth_value = depths[k];
        // when d < min_d or d > max_d, not valid depth.
        if (depth_value < valid_min_depth || depth_value > valid_max_depth) mask_depths[k] = (scalar_t) 0.0f;
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> depth_normalize(
    torch::Tensor depths, // [B, N_v, h, w]
    const float valid_min_depth,
    const float valid_max_depth,
    const float user_dist, // default as -1;
    const bool multi_mask, // whether multiply by the masks.
    const bool divided2, // whether divided by 2.
    const int device
) {
    cudaSetDevice(device); // on GPU device.
    CUDA_CHECK_ERRORS();

    const uint32_t batch_size = depths.size(0);
    const uint32_t num_views  = depths.size(1);
    const uint32_t im_h       = depths.size(2);
    const uint32_t im_w       = depths.size(3);

    cudaStream_t curr_stream = at::cuda::getCurrentCUDAStream();

    // init tensors;
    torch::Tensor min_depths  = torch::full({batch_size, num_views, 1, 1}, 99999.0f,  depths.options());
    torch::Tensor max_depths  = torch::full({batch_size, num_views, 1, 1}, -99999.0f, depths.options());

    static constexpr uint32_t block_size = 128;
    uint32_t grid_size = (batch_size*num_views*im_h*im_w + block_size - 1) / block_size;
    
    torch::Tensor mask_depths = torch::full({batch_size, num_views, im_h, im_w}, 1.0f, depths.options());
    if (multi_mask) {
        // get the depths' mask maps.
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            depths.scalar_type(), "get_mask_depths", ([&] {
                get_mask_depths_kernel<scalar_t> <<< grid_size, block_size, 0, curr_stream >>> (
                    depths.contiguous().data_ptr<scalar_t>(),
                    mask_depths.contiguous().data_ptr<scalar_t>(),
                    (scalar_t) valid_min_depth, (scalar_t) valid_max_depth,
                    batch_size, num_views, im_h, im_w
                );
            })
        );
    }

    // depth normalization.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        depths.scalar_type(), "depth_normalize", ([&] {
            depth_normalization_kernel<scalar_t> <<< grid_size, block_size, 0, curr_stream >>> (
                depths.contiguous().data_ptr<scalar_t>(),
                min_depths.contiguous().data_ptr<scalar_t>(),
                max_depths.contiguous().data_ptr<scalar_t>(),
                (scalar_t) valid_min_depth, (scalar_t) valid_max_depth,
                batch_size, num_views, im_h, im_w
            );
        })
    );

    // get the output tensors, incase the exceeding.
    torch::Tensor normalized_depth;
    torch::Tensor distance = max_depths - min_depths + 1e-10; // the distance of the tensors: [B, N, 1, 1]
    torch::Tensor mid      = (max_depths + min_depths) / 2; // the center, [B, N, 1, 1];

    // the normalized results out of bound is not received;
    if (user_dist < 0.0f) { // using the distance calculate by depth maps to normalize.
        if (multi_mask) {
            normalized_depth = (depths - mid) * mask_depths / distance;
            if (divided2) {
                normalized_depth *= 2.0;
            }
            // torch::clamp((depths - mid) / (distance / 2), -1.0, 1.0); // to [-1, 1];
            torch::Tensor max_value = torch::max(normalized_depth);
            // normalized_depth = torch::clamp(normalized_depth, -max_value.item(), max_value.item()); // to [-1, 1] or [-0.5, 0.5];
            normalized_depth = torch::clamp(normalized_depth, -1, 1); // to [-1, 1] or [-0.5, 0.5];
        } else {
            normalized_depth = (depths - mid) / distance;
            if (divided2) {
                normalized_depth *= 2.0;
            }
            torch::Tensor max_value = torch::max(normalized_depth);
            normalized_depth = torch::clamp(normalized_depth, -1, 1); // to [-1, 1] or [-0.5, 0.5];
        }
    } else { // using given user's distance to normalize.
        if (multi_mask) {
            normalized_depth = (depths - mid) * mask_depths / user_dist; // no clip here, since the backgrounds are marked as 0.
        } else {
            normalized_depth = (depths - mid) / user_dist;
            torch::Tensor max_value = torch::max(normalized_depth);
            // normalized_depth = torch::clamp(normalized_depth, -max_value.item(), max_value.item());
            normalized_depth = torch::clamp(normalized_depth, -1, 1); // to [-1, 1] or [-0.5, 0.5];

        }
    }

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>{normalized_depth, mask_depths, mid, distance};
}