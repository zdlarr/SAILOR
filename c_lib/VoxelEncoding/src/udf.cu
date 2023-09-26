#include <math.h>
#include <vector>
#include <stdint.h>
#include <assert.h>

#include "../include/cuda_helper.h"
#include "../include/sampling.h"
#include <string>
#include <stdio.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#define MAX_UDF_VALUE 999999.9

// To calculate the UDF values between queried points and pcds.
// Similar to chamfer distance, reference to:
// https://github.com/chrdiller/pyTorchChamferDistance

template <typename scalar_t>
__device__ scalar_t unsigned_distance( const scalar_t* p0, const scalar_t* p1 ) {
    scalar_t d_pow[3] = {(scalar_t) 0, (scalar_t) 0, (scalar_t) 0};
    #pragma unroll 3
    for ( u_short k=0; k<3; k++ ) d_pow[k] = (p1[k] - p0[k]) * (p1[k] - p0[k]);

    return (scalar_t) sqrtf( (float) d_pow[0] + d_pow[1] + d_pow[2] );
}

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


__global__ void udf_calculating_kernel(
    const float * __restrict__ ray_ori, // [B, N_rays, 3]
    const float * __restrict__ ray_dir, // [B, N_rays, 3]
    const float * __restrict__ sampled_depths, // [B, N_rays, N_sampled, 1];
    const float * __restrict__ pcds, // [B, N_points, 3]
    // output data.
    float * output_udf, // [B, N_rays, N_sampled, 1];
    float * output_directs, // [B, N_rays, N_sampled, 3];
    float * output_ulabels, // [B, N_rays, N_sampled, 1];
    // propeties.
    const uint32_t batch_size,
    const uint32_t num_rays, 
    const uint32_t num_sampled,
    const uint32_t num_max_points // default as 6e5;
) {
    // parallel for per points * max_points.
    #pragma unroll
    CUDA_KERNEL_LOOP(k, batch_size*num_rays*num_sampled*num_max_points) {
        // for each point, we need to calculating the distance between point and other pcds.
        uint32_t batch_idx   = (uint32_t) k / (num_rays*num_sampled*num_max_points);
        uint32_t ray_idx     = (uint32_t) (k - batch_idx*num_rays*num_sampled*num_max_points) / (num_sampled*num_max_points);
        uint32_t sampled_idx = (uint32_t) (k - batch_idx*num_rays*num_sampled*num_max_points - ray_idx*num_sampled*num_max_points) / num_max_points;
        uint32_t pcds_idx    = (uint32_t) k - batch_idx*num_rays*num_sampled*num_max_points - ray_idx*num_sampled*num_max_points - sampled_idx*num_max_points;
        
        // get all index for rays.
        const uint32_t _idx_ray     = batch_idx*num_rays*3 + ray_idx*3;
        const uint32_t _idx_sampled = batch_idx*num_rays*num_sampled*1 + ray_idx*num_sampled*1 + sampled_idx;
        const uint32_t _idx_pcds    = batch_idx*num_max_points*3 + pcds_idx*3;
        const uint32_t _idx_directs = batch_idx*num_rays*num_sampled*3 + ray_idx*num_sampled*3 + sampled_idx*3;

        // get new ptr of the inputs
        const float * _ray_ori       = ray_ori + _idx_ray;
        const float * _ray_dir       = ray_dir + _idx_ray;
        const float * _sampled_depth = sampled_depths + _idx_sampled;
        const float * _pcds          = pcds + _idx_pcds;

        // when the sampled depth is -1, return invalid udf, dirs, ulabels (-1);
        if (_sampled_depth[0] < 0) {
            output_udf[ _idx_sampled ] = (float) -1; // assign the -1 value for invalid sampled depth.
            output_ulabels[ _idx_sampled ] = (float) 0;
            // #pragma unroll
            // for ( u_short i=0; i<3; i++ ) output_directs[ _idx_directs+i ] = -1;

            continue;
        }

        // get the ray ori, dir, and sampled_t;
        float p0[3] = {(float) -1, (float) -1, (float) -1};
        #pragma unroll 3
        for ( u_short i=0; i<3; i++ ) p0[i] = _ray_ori[i] + _ray_dir[i]*_sampled_depth[0];
        
        // get the distance between p0 and p1;
        float distance = unsigned_distance<float>(p0, _pcds); // no problem here.

        // comparing the distance & ptr value, the distance is right here;
        atomicMin<float>( output_udf + _idx_sampled , distance ); // update the output_udf ptr;
        // printf( "udf now: %f, distance now: %f \n", output_udf[_idx_sampled], distance );
        
        // when updated, output_direct is assigned as distance, update the directions and ulabels.
        if ( output_udf[ _idx_sampled ] == distance ) {
            #pragma unroll 3
            for ( u_short i=0; i<3; i++ ) output_directs[ _idx_directs+i ] = _pcds[i] - p0[i];
        }
    }
}


__global__ void unsigned_distance_kernel(
    const float * __restrict__ ray_ori, // [B, N_rays, 3]
    const float * __restrict__ ray_dir, // [B, N_rays, 3]
    const float * __restrict__ sampled_depths, // [B, N_rays, N_sampled, 1];
    const float * __restrict__ pcds, // [B, N_points, 3]
    // output data.
    float * output_udf, // [B, N_rays, N_sampled, 1];
    float * output_directs, // [B, N_rays, N_sampled, 3];
    float * output_ulabels, // [B, N_rays, N_sampled, 1];
     // propeties.
     const uint32_t batch_size,
     const uint32_t num_rays, 
     const uint32_t num_sampled,
     const uint32_t num_max_points // default as 6e5;
) {
    const int batch=1024;
    const int n = num_rays*num_sampled;
    const int m = num_max_points;

    __shared__ float buf[batch*3]; // shared buffer, save the target pcds to temp points
    for (int i=blockIdx.x; i<batch_size; i+=gridDim.x) { // loop in batch_size;
        for (int k2=0; k2 < m; k2+=batch) {
            int end_k = min(m, k2+batch) - k2;

            for (int j = threadIdx.x; j < end_k*3; j+= blockDim.x) {
                buf[j] = pcds[i*m*3 + k2*3 + j]; // save to buffers.
            }
            
            __syncthreads();
            for (int j = threadIdx.x + blockIdx.y*blockDim.x; j < n; j += blockDim.x*gridDim.y) {
                // get the idx of the ray, sampled points.
                int ray_idx      = (int) j / num_sampled;
                int sampled_idx  = (int) j - ray_idx*num_sampled;
                int _ray_idx     = i*num_rays*3 + ray_idx*3;
                int _sampled_idx = i*num_rays*num_sampled*1 + ray_idx*num_sampled*1 + sampled_idx*1;
                int _dir_idx     = i*num_rays*num_sampled*3 + ray_idx*num_sampled*3 + sampled_idx*3;
                // invalid depths.
                if (sampled_depths[_sampled_idx] < 0) { // update udf with values -1;
                    output_udf[ _sampled_idx ] = (float) -1;
                    output_ulabels[ _sampled_idx ] = (float) 0;
                    continue;
                }

                // in registers.
                float p0[3] = {0.0f}, p1[3] = {0.0f}, best_dir[3] = {0.0f};
                float best_dis = 0.0f; // best distance.
                int   best_i   = 0; // index.
                int   end_ka   = end_k - (end_k&3);
                
                // get the source point.
                #pragma unroll 3
                for (int k=0; k<3; k++) p0[k] = ray_ori[_ray_idx+k] + ray_dir[_ray_idx+k] * sampled_depths[_sampled_idx];
                // indeed, here we calculate 16 points, is not necessary.
                for (int k=0; k < end_ka; k+=4) { // every time, matching 16 points.
                    for (int u=0; u<4; u++) {
                        #pragma unroll 3
                        for (int t=0; t<3; t++) p1[t] = buf[k*3+t+3*u];
                        float distance = unsigned_distance<float>(p0, p1);
                        if ( (k==0 && u == 0) || distance < best_dis) {
                            best_dis = distance; best_i = k + k2 + u;
                            #pragma unroll 3
                            for (int t=0; t<3; t++) best_dir[t] = p1[t] - p0[t];
                        }
                    }
                }
                for (int k=end_ka;k<end_k;k++){
					#pragma unroll 3
                    for (int t=0; t<3; t++) p1[t] = buf[k*3+t];
                    float distance = unsigned_distance<float>(p0, p1);
                    if ( k==0 || distance < best_dis) {
                        best_dis = distance; best_i = k + k2;
                        #pragma unroll 3
                        for (int t=0; t<3; t++) best_dir[t] = p1[t] - p0[t];
                    }
                }
                if (k2==0 || best_dis < output_udf[ _sampled_idx ]) {
                    output_udf[ _sampled_idx ] = best_dis;
                    // normalized
                    float norm_ = sqrtf(best_dir[0]*best_dir[0] + best_dir[1]*best_dir[1] + best_dir[2]*best_dir[2]);
                    #pragma unroll 3
                    for (int t=0; t<3; t++) output_directs[_dir_idx+t] = best_dir[t] / norm_;
				}
                
            }
            __syncthreads();
        }
    }

}


__global__ void unsigned_distance_kernel_v2(
    const float * __restrict__ pcds0, // [B, N_points, 3]
    const float * __restrict__ pcds1, // [B, N_points, 3]
    // output data.
    float * output_udf, // [B, N_points, 1];
    float * output_directs, // [B, N_points, 3];
    float * output_ulabels, // [B, N_points, 1];
     // propeties.
     const uint32_t batch_size,
     const uint32_t num_pcds0, // default as 6e5;
     const uint32_t num_pcds1 // default as 6e5;
) {
    const int batch=1024;
    const int n = num_pcds0;
    const int m = num_pcds1;

    __shared__ float buf[batch*3]; // shared buffer, save the target pcds to temp points
    for (int i=blockIdx.x; i<batch_size; i+=gridDim.x) { // loop in batch_size;
        for (int k2=0; k2 < m; k2+=batch) {
            int end_k = min(m, k2+batch) - k2;

            for (int j = threadIdx.x; j < end_k*3; j+= blockDim.x) {
                buf[j] = pcds1[i*m*3 + k2*3 + j]; // save to buffers.
            }
            
            __syncthreads();
            for (int j = threadIdx.x + blockIdx.y*blockDim.x; j < n; j += blockDim.x*gridDim.y) {
                int _pcds0_idx = i*num_pcds0*3 + j*3;
                int _label_idx = i*num_pcds0*1 + j*1;

                // in registers.
                float p0[3] = {0.0f}, p1[3] = {0.0f}, best_dir[3] = {0.0f};
                float best_dis = 0.0f; // best distance.
                int   best_i   = 0; // index.
                int   end_ka   = end_k - (end_k&3);
                
                // get the source point.
                #pragma unroll 3
                for (int k=0; k<3; k++) p0[k] = pcds0[_pcds0_idx+k];
                // indeed, here we calculate 16 points, is not necessary.
                for (int k=0; k < end_ka; k+=4) { // every time, matching 16 points.
                    for (int u=0; u<4; u++) {
                        #pragma unroll 3
                        for (int t=0; t<3; t++) p1[t] = buf[k*3+t+3*u];
                        float distance = unsigned_distance<float>(p0, p1);
                        if ( (k==0 && u == 0) || distance < best_dis) {
                            best_dis = distance; best_i = k + k2 + u;
                            #pragma unroll 3
                            for (int t=0; t<3; t++) best_dir[t] = p1[t] - p0[t];
                        }
                    }
                }
                for (int k=end_ka;k<end_k;k++){
					#pragma unroll 3
                    for (int t=0; t<3; t++) p1[t] = buf[k*3+t];
                    float distance = unsigned_distance<float>(p0, p1);
                    if ( k==0 || distance < best_dis) {
                        best_dis = distance; best_i = k + k2;
                        #pragma unroll 3
                        for (int t=0; t<3; t++) best_dir[t] = p1[t] - p0[t];
                    }
                }
                if (k2==0 || best_dis < output_udf[ _label_idx ]) {
                    output_udf[ _label_idx ] = best_dis;
                    // normalized
                    float norm_ = sqrtf(best_dir[0]*best_dir[0] + best_dir[1]*best_dir[1] + best_dir[2]*best_dir[2]);
                    #pragma unroll 3
                    for (int t=0; t<3; t++) output_directs[ _pcds0_idx+t ] = best_dir[t] / norm_;
				}
                
            }
            __syncthreads();
        }
    }

}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> udf_calculating( // calculate the UDF values between queried points and pcds.
    torch::Tensor ray_ori,    // [B, N_rays, 3]
    torch::Tensor ray_dir,  // [B, N_rays, 3]
    torch::Tensor sampled_depths,     // [B, N_rays, N_sampled, 1]
    torch::Tensor pcds, // [B, N_points, 3];
    const int device
) {
    cudaSetDevice(device);
    CUDA_CHECK_ERRORS();

    const uint32_t batch_size     = ray_ori.size(0);
    const uint32_t num_rays       = ray_ori.size(1);
    const uint32_t num_sampled    = sampled_depths.size(2);
    const uint32_t num_max_points = pcds.size(1);

    // init all the properties.
    torch::Tensor output_udfs    = torch::full( {batch_size, num_rays, num_sampled, 1},  99999.0f, sampled_depths.options() );
    torch::Tensor output_ulabels = torch::full( {batch_size, num_rays, num_sampled, 1},  1, sampled_depths.options() );
    torch::Tensor output_udirs   = torch::full( {batch_size, num_rays, num_sampled, 3}, -1, sampled_depths.options() );

    // static constexpr int block_size = 1024; // the num of threads must be larger than 1024 * 1024
    // very large value here, we need a super large block size.
    // int num_all_items = num_rays*num_sampled*(num_max_points / block_size);
    // float grid_size = (float) num_all_items + ((float) block_size - 1) / (float) block_size;

    // udf_calculating_kernel <<< grid_size, block_size >>> (
    //     ray_ori.contiguous().data_ptr<float>(),
    //     ray_dir.contiguous().data_ptr<float>(),
    //     sampled_depths.contiguous().data_ptr<float>(),
    //     pcds.contiguous().data_ptr<float>(),
    //     //
    //     output_udfs.contiguous().data_ptr<float>(),
    //     output_udirs.contiguous().data_ptr<float>(),
    //     output_ulabels.contiguous().data_ptr<float>(),
    //     //
    //     batch_size, num_rays, num_sampled, num_max_points
    // );
    
    unsigned_distance_kernel <<< dim3(32,16,1), 512 >>>(
        ray_ori.contiguous().data_ptr<float>(),
        ray_dir.contiguous().data_ptr<float>(),
        sampled_depths.contiguous().data_ptr<float>(),
        pcds.contiguous().data_ptr<float>(),
        //
        output_udfs.contiguous().data_ptr<float>(),
        output_udirs.contiguous().data_ptr<float>(),
        output_ulabels.contiguous().data_ptr<float>(),
        //
        batch_size, num_rays, num_sampled, num_max_points
    );

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>(output_udfs, output_udirs, output_ulabels);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> udf_calculating_v2( // calculate the UDF values between queried points and pcds.
    torch::Tensor pcds0, // [B, N_p0, 3];
    torch::Tensor pcds1, // [B, N_p1, 3];
    const int device
) {
    cudaSetDevice(device);
    CUDA_CHECK_ERRORS();

    const uint32_t batch_size = pcds0.size(0);
    const uint32_t num_pcds0  = pcds0.size(1);
    const uint32_t num_pcds1  = pcds1.size(1);

    // init all the properties.
    torch::Tensor output_udfs    = torch::full( {batch_size, num_pcds0, 1},  99999.0f, pcds0.options() );
    torch::Tensor output_ulabels = torch::full( {batch_size, num_pcds0, 1},  1, pcds0.options() );
    torch::Tensor output_udirs   = torch::full( {batch_size, num_pcds0, 3}, -1, pcds0.options() );

    unsigned_distance_kernel_v2 <<< dim3(32,16,1), 512 >>>(
        pcds0.contiguous().data_ptr<float>(),
        pcds1.contiguous().data_ptr<float>(),
        //
        output_udfs.contiguous().data_ptr<float>(),
        output_udirs.contiguous().data_ptr<float>(),
        output_ulabels.contiguous().data_ptr<float>(),
        //
        batch_size, num_pcds0, num_pcds1
    );

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>(output_udfs, output_udirs, output_ulabels);
}