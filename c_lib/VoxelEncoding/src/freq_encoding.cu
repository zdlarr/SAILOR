#include <math.h>
#include <vector>
#include <stdint.h>
#include <assert.h>

#include "../include/freq_encoding.h"
#include <string>
#include <stdio.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>

inline constexpr __device__ float PI() {return 3.141592653589793f;}

template <typename scalar_t>
__global__ void freq_encoding_kernel(
    const scalar_t * __restrict__ inputs, // [B, N_view, N_rays, N_sampled, 3] -> [N, 3]
    scalar_t * outputs, // [B, N_rays, N_sampled, C]; C=D+D*deg*2 -> [N, C]
    // const uint32_t batch_size, const uint32_t num_rays, const uint32_t num_sampled,
    // const uint32_t num_views, // default as 3;
    const uint32_t N_points, // B*N_v*N_rays*N_sampled
    const uint32_t inputs_dim, // input dim default as 3;
    const uint32_t deg_dim, // the frequency of log dim, max_freq_log;
    const uint32_t outputs_dim // C = D + D * deg * 2;
    // const uint32_t corners_num // if record 
) {
    // parallel for per-elements.
    CUDA_KERNEL_LOOP(k, N_points*outputs_dim) {
        const uint32_t n_idx = (uint32_t) k / outputs_dim;
        const uint32_t o_idx = (uint32_t) k - n_idx*outputs_dim;
        
        // locate the index of the tensors.
        inputs  += n_idx * inputs_dim;
        outputs += k;

        if (o_idx < inputs_dim) {
            outputs[0] = inputs[o_idx];
        } else {
            const uint32_t curr_in_idx = o_idx % inputs_dim;
            const uint32_t col_idx     = o_idx / inputs_dim - 1;
            const uint32_t freq        = col_idx / 2;
            const scalar_t phase_shift = (scalar_t)(col_idx % 2) * ((scalar_t) PI() / 2);
            outputs[0] = __sinf( scalbnf( inputs[curr_in_idx], freq ) + phase_shift );
        }
    }
}

template <typename scalar_t>
__global__ void freq_encoding_backward_kernel(
    const scalar_t * __restrict__ grad_output,
    const scalar_t * __restrict__ outputs,
    // grad for outputs.
    const uint32_t N_points, // B*N_v*N_rays*N_sampled
    const uint32_t inputs_dim, // input dim default as 3;
    const uint32_t deg_dim, // the frequency of log dim, max_freq_log;
    const uint32_t outputs_dim, // C = D + D * deg * 2;
    scalar_t  * grad_inputs
) {
    // parallel for per-elements.
    CUDA_KERNEL_LOOP(k, N_points*inputs_dim) {
        const uint32_t n_idx = (uint32_t) k / inputs_dim;
        const uint32_t o_idx = (uint32_t) k - n_idx*inputs_dim;

        // to locate the tensors.
        grad_output += n_idx*outputs_dim;
        outputs     += n_idx*outputs_dim;
        grad_inputs += k;

        // to the target pos.
        scalar_t result = grad_output[o_idx]; // the gradients for item d.
        grad_output     += inputs_dim;
        outputs         += inputs_dim;
        
        // grad_inputs = 1*grad_output[oidx] + 2^deg * (cos(x*2^deg) * grad_out0 - sin(x*2^deg) * grad_output1);
        for (uint32_t f=0; f < deg_dim; f++) {
            result += scalbnf(1.0f, f) * (grad_output[o_idx] * outputs[o_idx + inputs_dim] 
                    - outputs[o_idx] * grad_output[inputs_dim + o_idx]);

            grad_output += 2 * inputs_dim;
            outputs     += 2 * inputs_dim;
        }

        grad_inputs[0] = result;
    }
}


torch::Tensor freq_encode_forward(
    torch::Tensor input_tensor,
    const int deg, 
    const int device
) {
    cudaSetDevice(device); // on GPU device.
    CUDA_CHECK_ERRORS();

    const uint32_t num_points = input_tensor.size(0);
    const uint32_t inputs_dim = input_tensor.size(1);
    // C=D+D*deg*2
    const uint32_t outputs_dim = inputs_dim + inputs_dim*deg*2;

    // tensor. 
    torch::Tensor outputs = torch::empty({num_points, outputs_dim}, input_tensor.options());
 
    static constexpr uint32_t block_size = 128;
    uint32_t grid_size = (num_points*outputs_dim + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input_tensor.scalar_type(), "freq_encode_forward", ([&] {
            freq_encoding_kernel<scalar_t> <<< grid_size, block_size >>> (
                input_tensor.contiguous().data_ptr<scalar_t>(),
                outputs.contiguous().data_ptr<scalar_t>(),
                num_points, inputs_dim, (uint32_t) deg, outputs_dim               
            );
        })
    );
    return outputs;   
}

torch::Tensor freq_encode_backward(
    torch::Tensor grad_output,
    torch::Tensor inputs,
    torch::Tensor outputs,
    const int deg,
    const int device
) {
    cudaSetDevice(device);
    CUDA_CHECK_ERRORS();

    const uint32_t num_points  = inputs.size(0);
    const uint32_t inputs_dim  = inputs.size(1);
    const uint32_t outputs_dim = inputs_dim + inputs_dim*deg*2; // C = D+D*deg*2;

    // tensor : [B, N, inputs_dim];
    torch::Tensor grad_inputs = torch::empty({num_points, inputs_dim}, grad_output.options());

    static constexpr uint32_t block_size = 128;
    uint32_t grid_size = (num_points*inputs_dim + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_output.scalar_type(), "freq_encode_backward", ([&] {
            freq_encoding_backward_kernel<scalar_t> <<< grid_size, block_size >>> (
                grad_output.contiguous().data_ptr<scalar_t>(),
                outputs.contiguous().data_ptr<scalar_t>(),
                num_points, inputs_dim, (uint32_t) deg, outputs_dim,
                grad_inputs.contiguous().data_ptr<scalar_t>()
            );
        })
    );
    
    return grad_inputs;
}