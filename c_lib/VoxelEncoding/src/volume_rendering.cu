#include <math.h>
#include <vector>
#include <stdint.h>
#include <assert.h>

#include "../include/volume_rendering.h"
#include <string>
#include <stdio.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#define FEATS_DIM 3

// the volume rendering function

template <typename scalar_t>
__global__ void volume_rendering_training_forward_kernel(
    // const scalar_t * ray_ori, // [B, N_ray, 3]
    // const scalar_t * ray_dir, // [B, N_ray, 3]
    // const scalar_t * max_depth, // [B, N_ray, N_max_hits.]
    const scalar_t * sampled_depth, // [B, N_ray, N_sampled], the sampled depths.
    const int * sampled_sort_idx, // [B, N_ray, N_sampled], the sorted idx of the sampled z;
    // output of the ffmlp.
    const scalar_t * __restrict__ sigmas, // [B, N_ray, N_sampled, 1]
    const scalar_t * __restrict__ feats, // [B, N_ray, N_sampled, feat_dim]
    // output the values.
    scalar_t * output_feats, // [B, N_ray, feat_dim]
    scalar_t * output_depths, // [B, N_ray, 1]
    scalar_t * weight_sum, // [B, N_ray, 1] final alpha of the ray.
    scalar_t * output_alphas, // [B, N_ray, 1] final alpha maps.
    const uint32_t batch_size, const uint32_t num_rays, const uint32_t num_sampled,
    const scalar_t t_thres
) {
    // parallel per ray.
    #pragma unroll
    CUDA_KERNEL_LOOP(k, batch_size * num_rays) {
        uint32_t batch_idx = (uint32_t) (k / num_rays);
        uint32_t ray_idx   = k - batch_idx * num_rays;
        
        uint32_t basic_ray_idx   = batch_idx*num_rays*FEATS_DIM + ray_idx*FEATS_DIM;
        uint32_t basic_depth_idx = batch_idx*num_rays*1 + ray_idx*1;
        uint32_t basic_sampled_idx = batch_idx*num_rays*num_sampled + ray_idx*num_sampled + 0;
        uint32_t basic_feats_idx   = batch_idx*num_rays*num_sampled*FEATS_DIM + ray_idx*num_sampled*FEATS_DIM;

        // invalid ray, just return 0 (rgb, depth, weights) for each ray.
        // printf("min,max: %f, %f \n", sampled_depth[basic_sampled_idx + sampled_sort_idx[basic_sampled_idx+0]], sampled_depth[basic_sampled_idx + sampled_sort_idx[basic_sampled_idx+127]]);
        if (sampled_depth[basic_sampled_idx + sampled_sort_idx[basic_sampled_idx+num_sampled-1]] < 0) 
        { // if the the largest sampled z < 0 , the ray is invalid;
            output_depths[basic_depth_idx] = (scalar_t) 0;
            weight_sum[basic_depth_idx]    = (scalar_t) 0;
            output_alphas[basic_depth_idx] = (scalar_t) 0;
            // #pragma unroll 64
            for (int i=0; i < FEATS_DIM; i++) {
                output_feats[basic_ray_idx+i] = (scalar_t) 0; // rgb is assigned as 0;
            }
            continue;
        }

        // scalar_t ray_ori_p[3], ray_dir_p[3], ray_p[3];
        
        // #pragma unroll 3
        // for (int i=0; i < 3; i++) {
        //     ray_ori_p[i] = ray_ori[basic_ray_idx+i];
        //     ray_dir_p[i] = ray_dir[basic_ray_idx+i];
        // }

        scalar_t T = (scalar_t) 1.0f;
        scalar_t ws = 0, alphas = 0, d = 0, feat[FEATS_DIM] = {0};
        
        // rendering function: 
        // alpha = 1 - exp(-sigma*delta)
        // T = prod(1 - alpha)
        // w = alpha * T
        // C = sum(w * c)
        // d = sum(w * t)
        
        for (int i=0; i < num_sampled; i++) {
            // get current depths on ray.
            uint32_t sam_idx   = sampled_sort_idx[basic_sampled_idx+i];
            scalar_t sam_depth = sampled_depth[basic_sampled_idx+sam_idx];
            if (sam_depth < 0) continue; // -1 is invalid point.
            scalar_t sigma = sigmas[basic_sampled_idx+sam_idx];

            // get the delta_t;
            scalar_t sam_depth_nxt, delta_i;
            if (i < num_sampled - 1) { // get the next sampled depth value.
                uint32_t sam_idx_nxt;
                for (int m=i+1; m < num_sampled; m++) { // incase meeting -1 depth values.
                    sam_idx_nxt   = sampled_sort_idx[basic_sampled_idx+m];
                    sam_depth_nxt = sampled_depth[basic_sampled_idx+sam_idx_nxt];
                    if (sam_depth_nxt > 0) break;
                }
                delta_i = sam_depth_nxt - sam_depth;
            } else {
                delta_i = (scalar_t) 1e10; // the last point, inf delta.
            }

            // alpha = 1 - e^(-sigma * delta)
            const scalar_t alpha  = (scalar_t) 1.0f - __expf(- sigma * delta_i);
            const scalar_t weight = alpha * T; // w = alpha * T
            // output rgbd.
            // #pragma unroll 3
            for (int j=0; j < FEATS_DIM; j++) {
                feat[j] += weight * feats[basic_feats_idx+sam_idx*FEATS_DIM+j];
            }
            d += weight * sam_depth;
            ws += weight;
            alphas += weight * alpha;

            T *= (scalar_t) 1.0f - alpha; // T = prod (1 - alpha)

            // minimal remained transmittence
            if (T < t_thres) break;
        }
        //
        output_depths[basic_depth_idx] = d;
        weight_sum[basic_depth_idx]    = ws;
        output_alphas[basic_depth_idx] = alphas;
        // #pragma unroll 3
        for (int i=0; i < FEATS_DIM; i++) {
            output_feats[basic_ray_idx+i] = feat[i]; // rgb is assigned as 0;
        }
        //
    }
}


template <typename scalar_t>
__global__ void volume_rendering_occ_training_forward_kernel(
    // const scalar_t * ray_ori, // [B, N_ray, 3]
    // const scalar_t * ray_dir, // [B, N_ray, 3]
    // const scalar_t * max_depth, // [B, N_ray, N_max_hits.]
    const scalar_t * sampled_depth, // [B, N_ray, N_sampled], the sampled depths.
    const int * sampled_sort_idx, // [B, N_ray, N_sampled], the sorted idx of the sampled z;
    // output of the ffmlp.
    const scalar_t * __restrict__ sigmas, // [B, N_ray, N_sampled, 1]
    const scalar_t * __restrict__ feats, // [B, N_ray, N_sampled, FEATS_DIM]
    // output the values.
    scalar_t * output_feats, // [B, N_ray, FEATS_DIM]
    scalar_t * output_depths, // [B, N_ray, 1]
    scalar_t * weight_sum, // [B, N_ray, 1] final alpha of the ray.
    scalar_t * output_alphas, // [B, N_ray, 1] the final alpha map value.
    const uint32_t batch_size, const uint32_t num_rays, const uint32_t num_sampled,
    const scalar_t t_thres
) {
    // parallel per ray.
    #pragma unroll
    CUDA_KERNEL_LOOP(k, batch_size * num_rays) {
        uint32_t batch_idx = (uint32_t) (k / num_rays);
        uint32_t ray_idx   = k - batch_idx * num_rays;
        
        uint32_t basic_ray_idx   = batch_idx*num_rays*FEATS_DIM + ray_idx*FEATS_DIM;
        uint32_t basic_depth_idx = batch_idx*num_rays*1 + ray_idx*1;
        uint32_t basic_sampled_idx = batch_idx*num_rays*num_sampled + ray_idx*num_sampled + 0;
        uint32_t basic_feats_idx    = batch_idx*num_rays*num_sampled*FEATS_DIM + ray_idx*num_sampled*FEATS_DIM;

        // invalid ray, just return 0 (rgb, depth, weights) for each ray.
        // printf("min,max: %f, %f \n", sampled_depth[basic_sampled_idx + sampled_sort_idx[basic_sampled_idx+0]], sampled_depth[basic_sampled_idx + sampled_sort_idx[basic_sampled_idx+127]]);
        if (sampled_depth[basic_sampled_idx + sampled_sort_idx[basic_sampled_idx+num_sampled-1]] < 0) 
        { // if the the largest sampled z < 0 , the ray is invalid;
            output_depths[basic_depth_idx] = (scalar_t) 0;
            weight_sum[basic_depth_idx]    = (scalar_t) 0;
            output_alphas[basic_depth_idx] = (scalar_t) 0;
            // #pragma unroll 3
            for (int i=0; i < FEATS_DIM; i++) {
                output_feats[basic_ray_idx+i] = (scalar_t) 0; // rgb is assigned as 0;
            }
            continue;
        }

        // scalar_t ray_ori_p[3], ray_dir_p[3], ray_p[3];
        
        // #pragma unroll 3
        // for (int i=0; i < 3; i++) {
        //     ray_ori_p[i] = ray_ori[basic_ray_idx+i];
        //     ray_dir_p[i] = ray_dir[basic_ray_idx+i];
        // }

        scalar_t T = (scalar_t) 1.0f;
        scalar_t ws = 0, d = 0, alphas=0, feat[FEATS_DIM] = {0};
        
        // rendering function: 
        // alpha = occupancy values.
        // T = prod(1 - alpha)
        // w = alpha * T
        // C = sum(w * c)
        // d = sum(w * t)
        
        for (int i=0; i < num_sampled; i++) {
            // get current depths on ray.
            uint32_t sam_idx   = sampled_sort_idx[basic_sampled_idx+i];
            scalar_t sam_depth = sampled_depth[basic_sampled_idx+sam_idx];
            if (sam_depth < 0) continue; // -1 is invalid point.

            // get the delta_t;
            // scalar_t sam_depth_nxt, delta_i;
            // if (i < num_sampled - 1) { //  get the next sampled depth value.
            //     uint32_t sam_idx_nxt;
            //     for (int m=i+1; m < num_sampled; m++) {
            //         sam_idx_nxt = sampled_sort_idx[basic_sampled_idx+m];
            //         sam_depth_nxt = sampled_depth[basic_sampled_idx+sam_idx_nxt];
            //         if (sam_depth_nxt > 0) break;
            //     }
            //     delta_i = sam_depth_nxt - sam_depth;
            // } else {
            //     delta_i = (scalar_t) 1e10; // the last point, inf delta.
            // }

            // alpha = 1 - e^(-sigma * delta)
            // const scalar_t alpha  = (scalar_t) 1.0f - __expf(- sigma * delta_i);

            // For Occupany model. 
            const scalar_t alpha  = sigmas[basic_sampled_idx+sam_idx];
            const scalar_t weight = alpha * T; // w = alpha * T
            // output rgbd.
            // #pragma unroll 3
            for (int j=0; j < FEATS_DIM; j++) {
                feat[j] += weight * feats[basic_feats_idx+sam_idx*FEATS_DIM+j];
            }
            d += weight * sam_depth;
            ws += weight;
            alphas += weight * alpha;

            T *= (scalar_t) 1.0f - alpha; // T = prod (1 - alpha)

            // minimal remained transmittence
            if (T < t_thres) break;
        }
        //
        output_depths[basic_depth_idx] = d;
        weight_sum[basic_depth_idx]    = ws;
        output_alphas[basic_depth_idx] = alphas;
        // #pragma unroll 3
        for (int i=0; i < FEATS_DIM; i++) {
            output_feats[basic_ray_idx+i] = feat[i]; // rgb is assigned as 0;
        }
        //
    }
}



template <typename scalar_t>
__global__ void volume_rendering_training_backward_kernel(
    // gradients outputs.
    const scalar_t * __restrict__ grad_output_feats, // [B, N_rays, 3]
    const scalar_t * __restrict__ grad_depths, // [B, N_rays, 1]
    const scalar_t * __restrict__ grad_weight_sums, // [B, N_rays, 1]
    const scalar_t * __restrict__ grad_output_alphas, // [B, N_rays, 1]
    // inputs
    const scalar_t * sampled_depth, // [B, N_ray, N_sampled], the sampled depths.
    const int * sampled_sort_idx, // [B, N_ray, N_sampled], the sorted idx of the sampled z;
    // output of the ffmlp.
    const scalar_t * __restrict__ sigmas, // [B, N_ray, N_sampled, 1]
    const scalar_t * __restrict__ feats, // [B, N_ray, N_sampled, 3]
    // output of the Volume Rendering.
    const scalar_t * output_feats, // [B, N_rays, 3]
    const scalar_t * output_depths, // [B, N_rays, 1]
    const scalar_t * output_ws, // [B, N_rays, 1]
    const scalar_t * output_alphas, // [B, N_rays, 1]
    // output the values.
    scalar_t * grad_feats, // [B, N_ray, N_sampled, 3]
    scalar_t * grad_sigmas, // [B, N_ray, N_sampled, 1]
    const uint32_t batch_size, const uint32_t num_rays, const uint32_t num_sampled,
    const scalar_t t_thres
) {
    CUDA_KERNEL_LOOP(k, batch_size * num_rays) {
        uint32_t batch_idx = (uint32_t) (k / num_rays);
        uint32_t ray_idx   = k - batch_idx * num_rays;
        
        uint32_t basic_ray_idx   = batch_idx*num_rays*FEATS_DIM + ray_idx*FEATS_DIM;
        uint32_t basic_depth_idx = batch_idx*num_rays*1 + ray_idx*1;
        uint32_t basic_sampled_idx = batch_idx*num_rays*num_sampled + ray_idx*num_sampled + 0;
        uint32_t basic_feats_idx    = batch_idx*num_rays*num_sampled*FEATS_DIM + ray_idx*num_sampled*FEATS_DIM;
        
        // invalid ray (the largest sampled z < 0); no gradients for sigmas and color.
        if (sampled_depth[basic_sampled_idx + sampled_sort_idx[basic_sampled_idx+num_sampled-1]] < 0)
            continue;

        scalar_t T = (scalar_t) 1.0f; // initial T is 1.0f;
        scalar_t ws = 0, alphas = 0, d = 0, feat[FEATS_DIM] = {0};
        
        // scalar_t rgb_final[3] = {output_rgbs[basic_ray_idx+0], output_rgbs[basic_ray_idx+1], output_rgbs[basic_ray_idx+2]};
        scalar_t feat_final[FEATS_DIM] = {0};
        // #pragma unroll 3
        for (int j=0; j < FEATS_DIM; j++) {
            feat_final[j] = output_feats[basic_ray_idx+j];
        }

        scalar_t d_final      = output_depths[basic_depth_idx+0];
        scalar_t ws_final     = output_ws[basic_depth_idx+0];
        scalar_t alpha_final  = output_alphas[basic_depth_idx+0];

        // rendering function: 
        // delta = d_{t+1} - d_t
        // alpha = 1 - exp(-sigma*delta)
        // T = prod(1 - alpha)
        // w = alpha * T
        // C = sum(w * c) c: rgb.
        // D = sum(w * d)
        // ws = sum(w)
        // given the output C, D, ws
        // d_C / d_rgb(i) = w_i = alpha_i * T_i
        // d_L / d_sigma(i) = d_D / d_alpha(i) * d_alpha(i) / d_sigma_i

        for (int i=0; i < num_sampled; i++) {
            // get current depths on ray.
            uint32_t sam_idx   = sampled_sort_idx[basic_sampled_idx+i];
            scalar_t sam_depth = sampled_depth[basic_sampled_idx+sam_idx];
            if (sam_depth < 0) continue; // -1 is invalid point.
            
            scalar_t sigma = sigmas[basic_sampled_idx+sam_idx];

            // get delta_t
            scalar_t sam_depth_nxt, delta_i;
            if (i < num_sampled - 1) { //  get the next sampled depth value.
                uint32_t sam_idx_nxt;
                for (int m=i+1; m < num_sampled; m++) {
                    sam_idx_nxt = sampled_sort_idx[basic_sampled_idx+m];
                    sam_depth_nxt = sampled_depth[basic_sampled_idx+sam_idx_nxt];
                    if (sam_depth_nxt > 0) break;
                }
                delta_i = sam_depth_nxt - sam_depth;
            } else {
                delta_i = (scalar_t) 1e10; // the last point, inf delta.
            }
            
            const scalar_t alpha  = (scalar_t) 1.0f - __expf(- sigma * delta_i);
            const scalar_t weight = alpha * T;
            uint32_t basic_feats_sam_idx = basic_feats_idx+sam_idx*FEATS_DIM;

            // output rgbd.
            // #pragma unroll 3
            for (int j=0; j < FEATS_DIM; j++) {
                feat[j] += weight * feats[basic_feats_sam_idx+j];
            }
            
            d += weight * sam_depth;
            ws += weight;
            alphas += weight * alpha;

            T *= (scalar_t) 1.0f - alpha;

            // minimal remained transmittence
            if (T < t_thres) break;

            // gradients of rgbs, only contains one term: (d_L / d_R)  * alpha_i * T_i;
            // #pragma unroll 3
            for (int j=0; j < FEATS_DIM; j++) {
                grad_feats[basic_feats_sam_idx+j] = grad_output_feats[basic_ray_idx+j] * weight;
            }

            // gradients of sigmas.
            // for d_L / d_D  * (d_D / d_simga) + d_L / d_R * (d_R / d_sigma) + d_L /d_ws * (d_ws / d_sigma) + d_L /d_alphas * (d_alphas / d_sigma)
            // term 1: for ws.
            scalar_t tmp = grad_weight_sums[basic_depth_idx] * (1 - ws_final);
            scalar_t grad_sigma_ws = delta_i * (grad_weight_sums[basic_depth_idx] * (T * 1 - (ws_final - ws)) + tmp);

            scalar_t grad_simga_d =  delta_i * (grad_depths[basic_depth_idx] * (T*sam_depth - (d_final - d)) + tmp);

            scalar_t grad_sigma_alpha = delta_i * (grad_output_alphas[basic_depth_idx] * (2*T*alpha - (alpha_final - alphas)) + tmp);

            // scalar_t grad_sigma_c = delta_i * ( grad_output_rgbs[basic_ray_idx+0] * (T * rgbs[basic_rgbs_sam_idx+0] - (rgb_final[0] - rgb[0]))
            //                                   + grad_output_rgbs[basic_ray_idx+1] * (T * rgbs[basic_rgbs_sam_idx+1] - (rgb_final[1] - rgb[1]))
            //                                   + grad_output_rgbs[basic_ray_idx+2] * (T * rgbs[basic_rgbs_sam_idx+2] - (rgb_final[2] - rgb[2]))
            //                                   );
            scalar_t grad_sigma_f = 0;
            // #pragma unroll 3
            for (int j=0; j < FEATS_DIM; j++) {
                grad_sigma_f += grad_output_feats[basic_ray_idx+j] * (T * feats[basic_feats_sam_idx+j] - (feat_final[j] - feat[j]));
            }
            grad_sigma_f = delta_i * (grad_sigma_f + tmp);

            // printf("grad_d: %f, %f, %f, %f \n", grad_simga_d, grad_depths[basic_depth_idx], delta_i, T*sam_depth - (d_final - d) );

            grad_sigmas[basic_sampled_idx+sam_idx*1+0] = grad_sigma_ws + grad_simga_d + grad_sigma_f + grad_sigma_alpha;
        }
        //
    }
}


template <typename scalar_t>
__global__ void volume_rendering_occ_training_backward_kernel(
    // gradients outputs.
    const scalar_t * __restrict__ grad_output_feats, // [B, N_rays, FEATS_DIM]
    const scalar_t * __restrict__ grad_depths, // [B, N_rays, 1]
    const scalar_t * __restrict__ grad_weight_sums, // [B, N_rays, 1]
    const scalar_t * __restrict__ grad_output_alphas, // [B, N_rays, 1]
    // inputs
    const scalar_t * sampled_depth, // [B, N_ray, N_sampled], the sampled depths.
    const int * sampled_sort_idx, // [B, N_ray, N_sampled], the sorted idx of the sampled z;
    // output of the ffmlp.
    const scalar_t * __restrict__ sigmas, // [B, N_ray, N_sampled, 1]
    const scalar_t * __restrict__ feats, // [B, N_ray, N_sampled, FEATS_DIM]
    // output of the Volume Rendering.
    const scalar_t * output_feats, // [B, N_rays, FEATS_DIM]
    const scalar_t * output_depths, // [B, N_rays, 1]
    const scalar_t * output_ws, // [B, N_rays, 1]
    const scalar_t * output_alphas, // [B, N_rays, 1]
    // output the values.
    scalar_t * grad_feats, // [B, N_ray, N_sampled, FEATS_DIM]
    scalar_t * grad_sigmas, // [B, N_ray, N_sampled, 1]
    const uint32_t batch_size, const uint32_t num_rays, const uint32_t num_sampled,
    const scalar_t t_thres
) {
    CUDA_KERNEL_LOOP(k, batch_size * num_rays) {
        uint32_t batch_idx = (uint32_t) (k / num_rays);
        uint32_t ray_idx   = k - batch_idx * num_rays;
        
        uint32_t basic_ray_idx   = batch_idx*num_rays*FEATS_DIM + ray_idx*FEATS_DIM;
        uint32_t basic_depth_idx = batch_idx*num_rays*1 + ray_idx*1;
        uint32_t basic_sampled_idx = batch_idx*num_rays*num_sampled + ray_idx*num_sampled + 0;
        uint32_t basic_feats_idx    = batch_idx*num_rays*num_sampled*FEATS_DIM + ray_idx*num_sampled*FEATS_DIM;
        
        // invalid ray (the largest sampled z < 0); no gradients for sigmas and color.
        if (sampled_depth[basic_sampled_idx + sampled_sort_idx[basic_sampled_idx+num_sampled-1]] < 0)
            continue;

        scalar_t T = (scalar_t) 1.0f; // initial T is 1.0f;
        scalar_t ws = 0, alphas = 0, d = 0, feat[FEATS_DIM] = {0};
        
        scalar_t feat_final[FEATS_DIM] = {0};
        // #pragma unroll 3
        for (int j=0; j < FEATS_DIM; j++) {
            feat_final[j] = output_feats[basic_ray_idx+j];
        }
        
        scalar_t d_final      = output_depths[basic_depth_idx+0];
        scalar_t ws_final     = output_ws[basic_depth_idx+0];
        scalar_t alpha_final  = output_alphas[basic_depth_idx+0];

        // rendering function: 
        // delta = d_{t+1} - d_t
        // alpha = 1 - exp(-sigma*delta)
        // T = prod(1 - alpha)
        // w = alpha * T
        // C = sum(w * c) c: rgb.
        // D = sum(w * d)
        // ws = sum(w)
        // given the output C, D, ws
        // d_C / d_rgb(i) = w_i = alpha_i * T_i
        // d_L / d_sigma(i) = d_D / d_alpha(i) * d_alpha(i) / d_sigma_i

        for (int i=0; i < num_sampled; i++) {
            // get current depths on ray.
            uint32_t sam_idx   = sampled_sort_idx[basic_sampled_idx+i];
            scalar_t sam_depth = sampled_depth[basic_sampled_idx+sam_idx];
            if (sam_depth < 0) continue; // -1 is invalid point.
            
            // scalar_t sigma = sigmas[basic_sampled_idx+sam_idx];

            // get delta_t
            // scalar_t sam_depth_nxt, delta_i;
            // if (i < num_sampled - 1) { //  get the next sampled depth value.
            //     uint32_t sam_idx_nxt;
            //     for (int m=i+1; m < num_sampled; m++) {
            //         sam_idx_nxt = sampled_sort_idx[basic_sampled_idx+m];
            //         sam_depth_nxt = sampled_depth[basic_sampled_idx+sam_idx_nxt];
            //         if (sam_depth_nxt > 0) break;
            //     }
            //     delta_i = sam_depth_nxt - sam_depth;
            // } else {
            //     delta_i = (scalar_t) 1e10; // the last point, inf delta.
            // }
            
            const scalar_t alpha  = sigmas[basic_sampled_idx+sam_idx]; // [0, 1]
            const scalar_t weight = alpha * T;
            uint32_t basic_feats_sam_idx = basic_feats_idx+sam_idx*FEATS_DIM;

            // output rgbd.
            // #pragma unroll 3
            for (int j=0; j < FEATS_DIM; j++) {
                feat[j] += weight * feats[basic_feats_sam_idx+j];
            }
            
            d += weight * sam_depth;
            ws += weight;
            alphas += weight * alpha;

            T *= (scalar_t) 1.0f - alpha;

            // minimal remained transmittence
            if (T < t_thres) break;

            // gradients of rgbs, only contains one term: (d_L / d_R)  * alpha_i * T_i;
            // #pragma unroll 3
            for (int j=0; j < FEATS_DIM; j++) {
                grad_feats[basic_feats_sam_idx+j] = grad_output_feats[basic_ray_idx+j] * weight; // will not meet the gradient problems.
            }

            // gradients of sigmas.
            // for d_L / d_D  * (d_D / d_simga) + d_L / d_R * (d_R / d_sigma) + d_L /d_ws * (d_ws / d_sigma) + d_L /d_alphas * (d_alphas / d_sigma)
            // term 1: for ws.
            // clip the gradients to valid values., the alpha is in [0, 1]: occupancy values, tmp \in [1, +inf]
            scalar_t tmp = 1 / (1 - alpha + 1e-7) <= (scalar_t) 15.0f ? 1 / (1 - alpha) : (scalar_t) 15.0f;
            
            scalar_t grad_sigma_ws = grad_weight_sums[basic_depth_idx] * (T * 1 - (ws_final - ws)) * tmp;

            scalar_t grad_simga_d = grad_depths[basic_depth_idx] * (T*sam_depth - (d_final - d)) * tmp;

            scalar_t grad_sigma_alpha = grad_output_alphas[basic_depth_idx] * (2*T*alpha - (alpha_final - alphas)) * tmp;

            scalar_t grad_sigma_f = 0;
            // #pragma unroll 3
            for (int j=0; j < FEATS_DIM; j++) {
                grad_sigma_f += grad_output_feats[basic_ray_idx+j] * (T * feats[basic_feats_sam_idx+j] - (feat_final[j] - feat[j]));
            }
            grad_sigma_f *= tmp;
            // scalar_t grad_sigma_c = tmp * ( grad_output_rgbs[basic_ray_idx+0] * (T * rgbs[basic_rgbs_sam_idx+0] - (rgb_final[0] - rgb[0]))
            //                               + grad_output_rgbs[basic_ray_idx+1] * (T * rgbs[basic_rgbs_sam_idx+1] - (rgb_final[1] - rgb[1]))
            //                               + grad_output_rgbs[basic_ray_idx+2] * (T * rgbs[basic_rgbs_sam_idx+2] - (rgb_final[2] - rgb[2]))
            //                               );

            // printf("grad_d: %f, %f, %f, %f \n", grad_simga_d, grad_depths[basic_depth_idx], delta_i, T*sam_depth - (d_final - d) );

            grad_sigmas[basic_sampled_idx+sam_idx*1+0] = grad_sigma_ws + grad_simga_d + grad_sigma_f + grad_sigma_alpha;
        }
        //
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> volume_rendering_training_forward(
    torch::Tensor sampled_depth, // [B, N_ray, N_sampled], the sampled depths are sorted values.
    torch::Tensor sampled_sort_idx, // [B, N_ray, N_sampled], the sorted idx of the sampled z;
    torch::Tensor sigmas, // [B, N_ray, N_sampled, 1]
    torch::Tensor feats, // [B, N_ray, N_sampled, feat_dim], can be rgb data or feats.
    const int device, const float t_thresh, const bool support_occ
) {
    cudaSetDevice(device); // on GPU device.
    CUDA_CHECK_ERRORS();

    const uint32_t batch_size  = sampled_depth.size(0);
    const uint32_t num_rays    = sampled_depth.size(1);
    const uint32_t num_sampled = sampled_depth.size(2);
    const uint32_t feats_dim   = feats.size(3);

    torch::Tensor output_feats  = torch::empty({batch_size, num_rays, feats_dim}, sampled_depth.options());
    torch::Tensor output_depths = torch::empty({batch_size, num_rays, 1}, sampled_depth.options());
    torch::Tensor weight_sum    = torch::empty({batch_size, num_rays, 1}, sampled_depth.options());
    torch::Tensor output_alphas = torch::empty({batch_size, num_rays, 1}, sampled_depth.options()); // the blended alpha maps.

    static constexpr uint32_t block_size = 128;
    uint32_t grid_size = (batch_size * num_rays + block_size - 1) / block_size;

    if (support_occ) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            sampled_depth.scalar_type(), "volume_rendering_training_forward", ([&] {
                volume_rendering_occ_training_forward_kernel<scalar_t> <<< grid_size, block_size >>> (
                    sampled_depth.contiguous().data_ptr<scalar_t>(),
                    sampled_sort_idx.contiguous().data_ptr<int>(),
                    sigmas.contiguous().data_ptr<scalar_t>(),
                    feats.contiguous().data_ptr<scalar_t>(),
                    output_feats.contiguous().data_ptr<scalar_t>(),
                    output_depths.contiguous().data_ptr<scalar_t>(),
                    weight_sum.contiguous().data_ptr<scalar_t>(),
                    output_alphas.contiguous().data_ptr<scalar_t>(),
                    batch_size, num_rays, num_sampled, t_thresh
                );
            })
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            sampled_depth.scalar_type(), "volume_rendering_training_forward", ([&] {
                volume_rendering_training_forward_kernel<scalar_t> <<< grid_size, block_size >>> (
                    sampled_depth.contiguous().data_ptr<scalar_t>(),
                    sampled_sort_idx.contiguous().data_ptr<int>(),
                    sigmas.contiguous().data_ptr<scalar_t>(),
                    feats.contiguous().data_ptr<scalar_t>(),
                    output_feats.contiguous().data_ptr<scalar_t>(),
                    output_depths.contiguous().data_ptr<scalar_t>(),
                    weight_sum.contiguous().data_ptr<scalar_t>(),
                    output_alphas.contiguous().data_ptr<scalar_t>(),
                    batch_size, num_rays, num_sampled, t_thresh
                );
            })
        );
    }

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>{output_feats, output_depths, weight_sum, output_alphas};
}

std::tuple<torch::Tensor, torch::Tensor> volume_rendering_training_backward(
    torch::Tensor grad_output_feats, torch::Tensor grad_output_depths, torch::Tensor grad_weight_sums, torch::Tensor grad_output_alphas, 
    torch::Tensor output_feats, torch::Tensor output_depths, torch::Tensor output_ws, torch::Tensor output_alphas,
    torch::Tensor sampled_depth, // [B, N_ray, N_sampled], the sampled depths are sorted values.
    torch::Tensor sampled_sort_idx, // [B, N_ray, N_sampled], the sorted idx of the sampled z;
    torch::Tensor sigmas, // [B, N_ray, N_sampled, 1]
    torch::Tensor feats, // [B, N_ray, N_sampled, 3]
    const int device, const float t_thresh, const bool support_occ
) {
    cudaSetDevice(device); // on GPU device.
    CUDA_CHECK_ERRORS();

    const uint32_t batch_size  = sampled_depth.size(0);
    const uint32_t num_rays    = sampled_depth.size(1);
    const uint32_t num_sampled = sampled_depth.size(2);
    const uint32_t feats_dim   = output_feats.size(2);

    torch::Tensor grad_feats  = torch::full({batch_size, num_rays, num_sampled, feats_dim}, 0, sampled_depth.options());
    torch::Tensor grad_sigmas = torch::full({batch_size, num_rays, num_sampled, 1}, 0, sampled_depth.options());

    static constexpr uint32_t block_size = 128;
    uint32_t grid_size = (batch_size * num_rays + block_size - 1) / block_size;

    if (support_occ) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            sampled_depth.scalar_type(), "volume_rendering_training_backward", ([&] {
                volume_rendering_occ_training_backward_kernel<scalar_t> <<< grid_size, block_size >>> (
                    grad_output_feats.contiguous().data_ptr<scalar_t>(),
                    grad_output_depths.contiguous().data_ptr<scalar_t>(),
                    grad_weight_sums.contiguous().data_ptr<scalar_t>(),
                    grad_output_alphas.contiguous().data_ptr<scalar_t>(),
                    //
                    sampled_depth.contiguous().data_ptr<scalar_t>(),
                    sampled_sort_idx.contiguous().data_ptr<int>(),
                    sigmas.contiguous().data_ptr<scalar_t>(),
                    feats.contiguous().data_ptr<scalar_t>(),
                    //
                    output_feats.contiguous().data_ptr<scalar_t>(),
                    output_depths.contiguous().data_ptr<scalar_t>(),
                    output_ws.contiguous().data_ptr<scalar_t>(),
                    output_alphas.contiguous().data_ptr<scalar_t>(),
                    //
                    grad_feats.contiguous().data_ptr<scalar_t>(),
                    grad_sigmas.contiguous().data_ptr<scalar_t>(),
                    //
                    batch_size, num_rays, num_sampled, t_thresh
                );
            })
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            sampled_depth.scalar_type(), "volume_rendering_training_backward", ([&] {
                volume_rendering_training_backward_kernel<scalar_t> <<< grid_size, block_size >>> (
                    grad_output_feats.contiguous().data_ptr<scalar_t>(),
                    grad_output_depths.contiguous().data_ptr<scalar_t>(),
                    grad_weight_sums.contiguous().data_ptr<scalar_t>(),
                    grad_output_alphas.contiguous().data_ptr<scalar_t>(),
                    //
                    sampled_depth.contiguous().data_ptr<scalar_t>(),
                    sampled_sort_idx.contiguous().data_ptr<int>(),
                    sigmas.contiguous().data_ptr<scalar_t>(),
                    feats.contiguous().data_ptr<scalar_t>(),
                    //
                    output_feats.contiguous().data_ptr<scalar_t>(),
                    output_depths.contiguous().data_ptr<scalar_t>(),
                    output_ws.contiguous().data_ptr<scalar_t>(),
                    output_alphas.contiguous().data_ptr<scalar_t>(),
                    //
                    grad_feats.contiguous().data_ptr<scalar_t>(),
                    grad_sigmas.contiguous().data_ptr<scalar_t>(),
                    //
                    batch_size, num_rays, num_sampled, t_thresh
                );
            })
        );
    }

    return std::tuple<torch::Tensor, torch::Tensor>{grad_feats, grad_sigmas};
}


template <typename scalar_t>
__global__ void volume_rendering_occ_kernel(
    // const scalar_t * ray_ori, // [B, N_ray, 3]
    // const scalar_t * ray_dir, // [B, N_ray, 3]
    // const scalar_t * max_depth, // [B, N_ray, N_max_hits.]
    const scalar_t * sampled_depth, // [B, N_ray, N_sampled], the sampled depths.
    // const int * sampled_sort_idx, // [B, N_ray, N_sampled], the sorted idx of the sampled z;
    // output of the ffmlp.
    const scalar_t * __restrict__ sigmas, // [B, N_ray, N_sampled, 1]
    const scalar_t * __restrict__ feats, // [B, N_ray, N_sampled, FEATS_DIM]
    // output the values.
    scalar_t * output_feats, // [B, N_ray, FEATS_DIM]
    scalar_t * output_depths, // [B, N_ray, 1]
    scalar_t * weight_sum, // [B, N_ray, 1] final alpha of the ray.
    scalar_t * output_alphas, // [B, N_ray, 1] the final alpha map value.
    const uint32_t batch_size, const uint32_t num_rays, const uint32_t num_sampled,
    const scalar_t t_thres
) {
    // parallel per ray.
    #pragma unroll
    CUDA_KERNEL_LOOP(k, batch_size * num_rays) {
        uint32_t batch_idx = (uint32_t) (k / num_rays);
        
        uint32_t ray_idx   = k - batch_idx * num_rays;
        
        uint32_t basic_ray_idx     = batch_idx*num_rays*FEATS_DIM + ray_idx*FEATS_DIM;
        uint32_t basic_depth_idx   = batch_idx*num_rays*1 + ray_idx*1;
        uint32_t basic_sampled_idx = batch_idx*num_rays*num_sampled + ray_idx*num_sampled + 0;
        uint32_t basic_feats_idx   = batch_idx*num_rays*num_sampled*FEATS_DIM + ray_idx*num_sampled*FEATS_DIM;

        // fast filtering the invalid rays, if the largest depth < 0, then directly return the zeros values.
        if ( sampled_depth[ basic_sampled_idx+num_sampled-1 ] < 0 ) {
            output_depths[basic_depth_idx] = (scalar_t) 0;
            weight_sum[basic_depth_idx]    = (scalar_t) 0;
            output_alphas[basic_depth_idx] = (scalar_t) 0;
            // update the features results;
            #pragma unroll
            for (int i=0; i < FEATS_DIM; i++) {
                output_feats[basic_ray_idx+i] = (scalar_t) 0; // rgb is assigned as 0;
            }
            continue;
        }
        
        scalar_t T = (scalar_t) 1.0f;
        scalar_t ws = 0, d=0, alphas = 0, feat[FEATS_DIM] = {0};
        
        #pragma unroll
        for (int i=0; i < num_sampled; i++) {
            // get current depths on rays;
            scalar_t sam_depth = sampled_depth[basic_sampled_idx+i];
            if ( sam_depth < 0 ) continue; // <0, invalid point, jumping continue;
            
            // for occupancy models; weight_i = T_i * alpha_i; T_i =\prod_{j=0}^i{1-alpha_j}
            const scalar_t alpha  = sigmas[basic_sampled_idx+i];
            const scalar_t weight = alpha * T; // w = alpha * T;
            // for occupancy values, alpha = sigma_value;
            #pragma unroll
            for (int j=0; j < FEATS_DIM; j++) {
                feat[j] += weight * feats[basic_feats_idx+i*FEATS_DIM+j];
            }
            d += weight * sam_depth;
            ws += weight;
            alphas += weight * alpha;

            T *= (scalar_t) 1.0f - alpha; // T = prod(1-alpha);

            // minimal remained transmittance;
            if (T < t_thres) break;
        }
        //
        output_depths[basic_depth_idx] = d;
        weight_sum[basic_depth_idx]    = ws;
        output_alphas[basic_depth_idx] = alphas;
        #pragma unroll
        for (int i=0; i < FEATS_DIM; i++) {
            output_feats[basic_ray_idx+i] = feat[i]; // rgb is assigned as 0;
        }
        // 
    }
}


// New forward occ volume rendering, OCC-based rendering;
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> volume_rendering_occ_forward(
    torch::Tensor sampled_depth, // [B, N_ray, N_sampled], the sampled depths are sorted values.
    torch::Tensor occs, // [B, N_ray, N_sampled, 1]
    torch::Tensor feats, // [B, N_ray, N_sampled, feat_dim], can be rgb data or feats.
    const int device, const float t_thresh
) {
    cudaSetDevice(device); // on GPU device.
    CUDA_CHECK_ERRORS();

    const uint32_t batch_size  = sampled_depth.size(0);
    const uint32_t num_rays    = sampled_depth.size(1);
    const uint32_t num_sampled = sampled_depth.size(2);
    const uint32_t feats_dim   = feats.size(3);

    torch::Tensor output_feats  = torch::empty({batch_size, num_rays, feats_dim}, sampled_depth.options());
    torch::Tensor output_depths = torch::empty({batch_size, num_rays, 1}, sampled_depth.options());
    torch::Tensor weight_sum    = torch::empty({batch_size, num_rays, 1}, sampled_depth.options());
    torch::Tensor output_alphas = torch::empty({batch_size, num_rays, 1}, sampled_depth.options()); // the blended alpha maps.

    static constexpr uint32_t block_size = 128;
    uint32_t grid_size = (batch_size * num_rays + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        sampled_depth.scalar_type(), "volume_rendering_occ_kernel", ([&] {
            volume_rendering_occ_kernel<scalar_t> <<< grid_size, block_size >>> (
                sampled_depth.contiguous().data_ptr<scalar_t>(),
                occs.contiguous().data_ptr<scalar_t>(),
                feats.contiguous().data_ptr<scalar_t>(),
                output_feats.contiguous().data_ptr<scalar_t>(),
                output_depths.contiguous().data_ptr<scalar_t>(),
                weight_sum.contiguous().data_ptr<scalar_t>(),
                output_alphas.contiguous().data_ptr<scalar_t>(),
                batch_size, num_rays, num_sampled, t_thresh
            );
        })
    );

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>{output_feats, output_depths, weight_sum, output_alphas};
}

