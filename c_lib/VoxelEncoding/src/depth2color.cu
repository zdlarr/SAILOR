
#include <math.h>
#include <vector>
#include <stdint.h>
#include <assert.h>

#include "../include/depth2color.h"
#include <string>
#include <stdio.h>
#include <iostream>
#include <ATen/cuda/CUDAContext.h>

__global__ void inverse_mat3x3(
    const float* Ms, // [N_v, 3, 3]
    const int num_views,
    float* inv_Ms // [N_v, 3, 3]
) {
    CUDA_KERNEL_LOOP(i, num_views) {
        size_t basic_id = i*9;
        float a1 = Ms[basic_id+0],
              b1 = Ms[basic_id+1],
              c1 = Ms[basic_id+2],
              a2 = Ms[basic_id+3],
              b2 = Ms[basic_id+4],
              c2 = Ms[basic_id+5],
              a3 = Ms[basic_id+6],
              b3 = Ms[basic_id+7],
              c3 = Ms[basic_id+8];
              
        float det = a1*(b2*c3-c2*b3)-a2*(b1*c3-c1*b3)+a3*(b1*c2-c1*b2);
        inv_Ms[basic_id+0] = (b2*c3-c2*b3) / det;
        inv_Ms[basic_id+1] = (b3*c1-c3*b1) / det;
        inv_Ms[basic_id+2] = (b1*c2-c1*b2) / det;
        inv_Ms[basic_id+3] = (a3*c2-c3*a2) / det;
        inv_Ms[basic_id+4] = (a1*c3-c1*a3) / det;
        inv_Ms[basic_id+5] = (a2*c1-c2*a1) / det;
        inv_Ms[basic_id+6] = (a2*b3-b2*a3) / det;
        inv_Ms[basic_id+7] = (b1*a3-a1*b3) / det;
        inv_Ms[basic_id+8] = (b2*a1-a2*b1) / det;
    }
}

__global__ void depth2color_fillin_kernel(
   const float *depth2color0, // [nv,1,h',w']
   const float *masks, // [nv,1,h',w']
   float *depth2color1, // [nv,1,h',w']
   const float d_thresh, 
   const int dsize,
   const int num_views,
   const int h,
   const int w
) {
    #pragma unroll
    CUDA_KERNEL_LOOP(k, num_views*h*w) {
        if (masks[k] == 0.0f) continue; // only process valid regions.
        float d_tmp = depth2color0[k];
        if (d_tmp >= 1e-5) continue;
        
        uint32_t v_idx = (uint32_t) k / (h*w);
		uint32_t h_idx = (uint32_t) (k - v_idx*h*w) / w;
		uint32_t w_idx = (uint32_t) k - v_idx*h*w - h_idx*w;

        float d_sum = 0.0f, d_num = 0.0f, d_final = 0.0f;
        float d_max = -9999.f, d_min = 9999.f;
        int dbidx   = v_idx*h*w;

        for (int i =-dsize; i <= dsize; i++) {
            for (int j =-dsize; j <= dsize; j++) {
                int h_idx_ = h_idx + i, w_idx_ = w_idx + j;
                if (h_idx_ >= 0 && h_idx_ < h && w_idx_ >= 0 && w_idx_ < w) {
                    // get new depths.
                    float d_p = depth2color0[dbidx + h_idx_*w + w_idx_];
                    if (d_p > 1e-5) {
                        d_sum += d_p;
                        d_num += 1.0f;
                        // get min-max d in the region.
                        if (d_p > d_max) d_max = d_p;
                        if (d_p < d_min) d_min = d_p;
                    }
                }
            }
        }
        if (d_num > 0 && (d_max - d_min) <= d_thresh) {
            d_final = (float) d_sum / d_num;
        }

        depth2color1[k] = d_final;
    }
}

__global__ void depth2color_projection_kernel(
    const float *depths, // [Nv, 1, h, w]
    const float *kc, // [Nv, 3, 3]
    const float *kd, // [Nv, 3, 3]
    const float *kd_inv, // [Nv, 3, 3]
    const float *rt_d2c, // [Nv, 3, 4]
    const float *masks, // [Nv, 1, h', w']
    float *depths2color, // [Nv, 1, h', w']
    const int num_views,
    const int o_h,
    const int o_w,
    const int tar_h,
    const int tar_w
) {
    #pragma unroll
    CUDA_KERNEL_LOOP(k, num_views*o_h*o_w) {
        uint32_t v_idx = (uint32_t) k / (o_h*o_w);
		uint32_t h_idx = (uint32_t) (k - v_idx*o_h*o_w) / o_w;
		uint32_t w_idx = (uint32_t) k - v_idx*o_h*o_w - h_idx*o_w;

        float d = depths[k], v_cam0[3] = {0.0f}, v_cam1[3] = {0.0f};
        float pc1[2] = {0.0f};

        int b_kidx = v_idx*9;
        int b_rtidx = v_idx*12;
        int b_d2c_idx = v_idx*tar_h*tar_w;
        if (d < 0.2f || d > 2.2f) continue;
        // get point's position in cam coordinate. (K_inv @ [w,h,1]) * d.
        v_cam0[0] = (kd_inv[b_kidx +0]*w_idx + kd_inv[b_kidx +1]*h_idx + kd_inv[b_kidx +2])*d;
        v_cam0[1] = (kd_inv[b_kidx +3]*w_idx + kd_inv[b_kidx +4]*h_idx + kd_inv[b_kidx +5])*d;
        v_cam0[2] = (kd_inv[b_kidx +6]*w_idx + kd_inv[b_kidx +7]*h_idx + kd_inv[b_kidx +8])*d;

        v_cam1[0] = rt_d2c[b_rtidx +0]*v_cam0[0] + rt_d2c[b_rtidx +1]*v_cam0[1] + rt_d2c[b_rtidx +2]*v_cam0[2] + rt_d2c[b_rtidx +3];
        v_cam1[1] = rt_d2c[b_rtidx +4]*v_cam0[0] + rt_d2c[b_rtidx +5]*v_cam0[1] + rt_d2c[b_rtidx +6]*v_cam0[2] + rt_d2c[b_rtidx +7];
        v_cam1[2] = rt_d2c[b_rtidx +8]*v_cam0[0] + rt_d2c[b_rtidx +9]*v_cam0[1] + rt_d2c[b_rtidx+10]*v_cam0[2] + rt_d2c[b_rtidx+11];
        
        v_cam1[0] /= v_cam1[2];
        v_cam1[1] /= v_cam1[2];

        pc1[0] = kc[b_kidx +0]*v_cam1[0] + kc[b_kidx +1]*v_cam1[1] + kc[b_kidx +2];
        pc1[1] = kc[b_kidx +3]*v_cam1[0] + kc[b_kidx +4]*v_cam1[1] + kc[b_kidx +5];

        int u = (int) round(pc1[0]), v = (int) round(pc1[1]);

        if (u >= 0 && u < tar_w && v >= 0 && v < tar_h) {
            int idx = b_d2c_idx+ v*tar_w + u;
            depths2color[idx] = v_cam1[2] * masks[idx]; // no depth in mask regions.
        }
    }
}

torch::Tensor depth2color(
    torch::Tensor depths, torch::Tensor masks, // [2,1,h,w]
    torch::Tensor kc, torch::Tensor kd, torch::Tensor rt_d2c, // [2,3,3(4)]
    const int tar_h, const int tar_w, const int dsize, const float d_thresh,
    const int device
) {
    cudaSetDevice(device); // on GPU device.
    CUDA_CHECK_ERRORS();

    cudaStream_t curr_stream = at::cuda::getCurrentCUDAStream();

    const uint32_t num_views = depths.size(0);
    const uint32_t o_h       = depths.size(2);   
    const uint32_t o_w       = depths.size(3);
    // warped depths. [2,1,h',w']

    torch::Tensor depths2color  = torch::full({num_views, 1, tar_h, tar_w}, 0.0, depths.options());
    torch::Tensor kd_inv = torch::full({num_views, 3, 3}, 0, kd.options());

    int block_size, grid_size, min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, inverse_mat3x3, 0, 0);
    grid_size = (num_views + block_size - 1) / block_size;
    // inverse 3x3 matrix.
    inverse_mat3x3 <<< grid_size, block_size, 0, curr_stream >>> (
        kd.contiguous().data_ptr<float>(), num_views, 
        kd_inv.contiguous().data_ptr<float>()
    );

    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, depth2color_projection_kernel, 0, 0);
    grid_size = (num_views*o_h*o_w + block_size - 1) / block_size;

    depth2color_projection_kernel <<< grid_size, block_size, 0, curr_stream >>> (
        depths.contiguous().data_ptr<float>(),
        kc.contiguous().data_ptr<float>(),
        // parameters.
        kd.contiguous().data_ptr<float>(),
        kd_inv.contiguous().data_ptr<float>(),
        rt_d2c.contiguous().data_ptr<float>(),
        // outputs.
        masks.contiguous().data_ptr<float>(),
        depths2color.contiguous().data_ptr<float>(),
        num_views, o_h, o_w, tar_h, tar_w
    );
    
    torch::Tensor depths2colorc = depths2color.clone();
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, depth2color_fillin_kernel, 0, 0);
    grid_size = (num_views*tar_h*tar_w + block_size - 1) / block_size;

    depth2color_fillin_kernel <<< grid_size, block_size, 0, curr_stream >>> (
        depths2color.contiguous().data_ptr<float>(),
        masks.contiguous().data_ptr<float>(),
        depths2colorc.contiguous().data_ptr<float>(),
        d_thresh, dsize, num_views, tar_h, tar_w
    );

    CUDA_CHECK_ERRORS();
    return depths2colorc;
}