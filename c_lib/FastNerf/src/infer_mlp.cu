// Fully-fused MLP and Hydra-Attention.

#include <math.h>
#include <vector>
#include <stdint.h>
#include <assert.h>

// #include <omp.h>
#include "../include/cuda_helper.h"
#include <ATen/cuda/CUDAContext.h>
#include "../include/infer_mlp.h"
#include "cuda_fp16.h"
#include <string>
#include <stdio.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>

#define FEATS_DIM 30
#define NUM_VIEWS 4
#define NUM_HYDRA_STACKS 2
#define GEO_FEATS_DIM 16
#define RGB_FEATS_DIM 16
#define OUTPUT_GEO_FEATS_DIM 8
#define DENSITY_FEATS_DIM 64
#define MAX_DENSITY_HIDDEN_DIM 64
#define MAX_GEO_FEAT_HIDDEN_DIM 18
#define MAX_COLOR_FEAT_HIDDEN_DIM 32
#define COLOR_FEATS_DIM 30
#define NUM_FEAT_LAYERS 4
#define NUM_NERF_STREAMS 2

// utils functions;
// sqrt func;
__device__ __forceinline__ half sqrt_half(half val) {
    // sqrt(half)
    return __float2half( sqrtf( __half2float(val) ) );
}

// relu:
__device__ __forceinline__ half relu(half val) {
    // each relu's max(val, 0.0f);
	return __float2half( max( __half2float(val), 0.0f) );
}

// sigmoid: 1/(1+e^(-x))
__device__ __forceinline__ half sigmoid(half val) {
    return __float2half( 1.0f / ( 1.0f + expf(-1.0f *  __half2float(val)) ) );
}

template <const uint32_t size>
__device__ __forceinline__
void relu_tensor(half * vals) {
    #pragma unroll size
    for (int i=0; i < size; i++) {
        vals[i] = relu(vals[i]);
    }
}

template <const uint32_t size>
__device__ __forceinline__
void sigmoid_tensor(half * vals) {
    #pragma unroll size
    for (int i=0; i < size; i++) {
        vals[i] = sigmoid(vals[i]);
    }
}

template <const uint32_t size>
__device__ __forceinline__
void sqrt_tensor(half * vals) {
    #pragma unroll size
    for (int i=0; i < size; i++) {
        vals[i] = sqrt_half(vals[i]);
    }
}

// fc function;
template <const uint32_t in_dim, const uint32_t out_dim, const uint32_t offset, const uint32_t max_hidden_dim>
__device__ __forceinline__
void fc_layer(half *layer_data, const half *net_params, int &param_index) {
    // return W @ X_in + B; e.g., W: [64, 18] X: [18, 1] B: [64, 1]
    // net_params : W,B,W,B ..., W,B;
    // e.g., max_hidden_dim 64;
    
    #pragma unroll out_dim
    for (int i=0; i<out_dim; i++) { // since the bias dim is outdim;
        layer_data[offset+i] = net_params[param_index + in_dim*out_dim + i]; // load the bias to layer_data;
    }
    #pragma unroll out_dim
    for (int i=0; i<out_dim; i++) {
        #pragma unroll in_dim
        for (int j=0; j<in_dim; j++) { // save to the last 64 dim or first 64 dim;
            layer_data[offset+i] = __hfma( net_params[param_index++], layer_data[max_hidden_dim+j-offset], layer_data[offset+i] );
        }
    }

    param_index += out_dim;
}

template <const uint32_t in_dim, const uint32_t out_dim, const uint32_t offset, const uint32_t max_hidden_dim>
__device__ __forceinline__
void fc_layer_no_bias(half *layer_data, const half *net_params, int &param_index) {
    // return W @ X + 0;
    #pragma unroll out_dim
    for (int i=0; i<out_dim; i++) {
        layer_data[offset+i] = __float2half( 0.0f ); // load the 0 to layer_data;
    }
    #pragma unroll out_dim
    for (int i=0; i<out_dim; i++) {
        #pragma unroll in_dim
        for (int j=0; j<in_dim; j++) {
            layer_data[offset+i] = __hfma( net_params[param_index++], layer_data[max_hidden_dim+j-offset], layer_data[offset+i] );
        }
    }
}

// density MLP kernel;
__global__ void density_feat_kernel(
    const __half * mlp_params, // one dim's tensor;
    const float * p_z, // [N_points, 1];
    const float * p_tpsdf, // [N_points, 1];
    const float * feats_geo, // [N_points, 16];
    const int n_points, // e.g., 1, 1024, 64
    float * outputs_density_feats, // [N_points, 64]
    float * outputs_geo_feats // [N_points, 8]
) {
    constexpr int param_size = 13696; // 18 * 64 + 64 + (64 * 64 + 64) * 3, 4 layers' MLP;
    constexpr int param_size_geo_feat = 440; // 16 * 18 + 16 + 8 * 16 + 8, 2 layers' MLP;
    constexpr int geo_feat_param_offset = 22081; // total density mlp's params' size;
    
    __shared__ half net_params[param_size]; // shared memory params;
    __shared__ half geo_net_params[param_size_geo_feat];
    
    // load mlp params to shared memory, each thread in the block shares this memory;
    // when all threads in a block run finished, then network_params are filled;
    int cached_idx = threadIdx.x;
    while (cached_idx < param_size) {
        net_params[cached_idx] = mlp_params[cached_idx];
        cached_idx += blockDim.x; // add by num of threads in a block;
    }
    cached_idx = threadIdx.x;
    while (cached_idx < param_size_geo_feat) {
        geo_net_params[cached_idx] = mlp_params[geo_feat_param_offset+cached_idx];
        cached_idx += blockDim.x;
    }
    __syncthreads();

    // reduced layer data, the max size is max_hidden_dim * 2;
    half layer_data[MAX_DENSITY_HIDDEN_DIM*2];
    half layer_data_geo_feat[MAX_GEO_FEAT_HIDDEN_DIM*2];
    
    // for each thread;
    CUDA_KERNEL_LOOP(k, n_points) {

        // per-view data;
        int param_index = 0, param_index_geo = 0;

        // first layers' input: feed inputs to layer data's first 64 dim;
        layer_data[0] = __float2half( p_z[k] );
        layer_data[1] = __float2half( p_tpsdf[k] );
        layer_data_geo_feat[0] = layer_data[0];
        layer_data_geo_feat[1] = layer_data[1];

        #pragma unroll 16
        for(int j=0; j<GEO_FEATS_DIM; j++) {
            layer_data[2+j] = __float2half( feats_geo[GEO_FEATS_DIM*k+j] );
            layer_data_geo_feat[2+j] = layer_data[2+j];
        }

        // parse feature MLP's each layers to obtain output features; totally four layers;
        fc_layer<18, 64, 64, MAX_DENSITY_HIDDEN_DIM>(layer_data, net_params, param_index);
        relu_tensor<MAX_DENSITY_HIDDEN_DIM>(layer_data+64);
        fc_layer<64, 64, 0, MAX_DENSITY_HIDDEN_DIM>(layer_data, net_params, param_index);
        relu_tensor<MAX_DENSITY_HIDDEN_DIM>(layer_data);
        fc_layer<64, 64, 64, MAX_DENSITY_HIDDEN_DIM>(layer_data, net_params, param_index);
        relu_tensor<MAX_DENSITY_HIDDEN_DIM>(layer_data+64);
        fc_layer<64, 64, 0, MAX_DENSITY_HIDDEN_DIM>(layer_data, net_params, param_index);
        relu_tensor<MAX_DENSITY_HIDDEN_DIM>(layer_data);

        // parse geo_feat mlp;
        fc_layer<18, 16, 18, MAX_GEO_FEAT_HIDDEN_DIM>(layer_data_geo_feat, geo_net_params, param_index_geo);
        relu_tensor<16>(layer_data_geo_feat+18);
        fc_layer<16, 8, 0, MAX_GEO_FEAT_HIDDEN_DIM>(layer_data_geo_feat, geo_net_params, param_index_geo);
        relu_tensor<8>(layer_data_geo_feat); // final activation relu;

        // save to global memory;
        #pragma unroll 64
        for (int j=0; j<DENSITY_FEATS_DIM; j++) {
            outputs_density_feats[DENSITY_FEATS_DIM*k+j] = __half2float( layer_data[j] );
        }
        #pragma unroll 8
        for (int j=0; j<OUTPUT_GEO_FEATS_DIM; j++) {
            outputs_geo_feats[OUTPUT_GEO_FEATS_DIM*k+j] = __half2float( layer_data_geo_feat[j] );
        }
        
    }
}

// density sigma MLP;
__global__ void density_sigma_kernel(
    const __half * mlp_params, // one dim's tensor;
    const float * feats_geo, // [N_points, 64];
    const int n_points, // e.g., 1, 1024, 64
    float * outputs_density // [N_points, 1]
) {
    constexpr int param_offset = 13696; // 18 * 64 + 64 + (64 * 64 + 64) * 3, 4 layers' MLP; 
    constexpr int param_size   = 22081 - param_offset; // get the offset for the fused mlp;
    
    __shared__ half net_params[param_size];
    
    int cached_idx = threadIdx.x;
    while (cached_idx < param_size) {
        net_params[cached_idx] = mlp_params[cached_idx+param_offset];
        cached_idx += blockDim.x;
    }
    __syncthreads();

    half layer_data[MAX_DENSITY_HIDDEN_DIM*2];

    CUDA_KERNEL_LOOP(k, n_points) {
        int param_index = 0;
        // set input data;
        #pragma unroll 64
        for(int j=0; j<DENSITY_FEATS_DIM; j++) {
            layer_data[j] = __float2half( feats_geo[DENSITY_FEATS_DIM*k+j] );
        }
        // forward MLP;
        fc_layer<64, 64, 64, MAX_DENSITY_HIDDEN_DIM>(layer_data, net_params, param_index);
        relu_tensor<MAX_DENSITY_HIDDEN_DIM>(layer_data+64);
        fc_layer<64, 64, 0, MAX_DENSITY_HIDDEN_DIM>(layer_data, net_params, param_index);
        relu_tensor<MAX_DENSITY_HIDDEN_DIM>(layer_data);
        fc_layer<64, 1, 64, MAX_DENSITY_HIDDEN_DIM>(layer_data, net_params, param_index);
        sigmoid_tensor<1>(layer_data+64);
        // save to global memory;
        outputs_density[k] = __half2float( layer_data[MAX_DENSITY_HIDDEN_DIM] );
    }
}

// color MLP kernel;
__global__ void hydra_attention_kernel(
    const __half * hydra_params, // one dim's tensor;
    // const float * rgb_feats, // [B, N_v, N_points, 16];
    // const float * geo_feats, // [B, N_v, N_points, 8]
    // const float * rgbs, // [B, N_v, N_points, 3];
    // const float * dirs, // [B, N_v, N_points, 3];
    const float * feats, // [B, N_v, N_points, 30];
    const int n_points, // e.g., N_v_points = N_view * N_points
    const int batch_size, const int num_views, // num of texture views;
    float * outputs_color_feats, // [B, N_points, 30];
    float * outputs_color // [B, N_points, 30];
) {
    int n_pts_per_batch = (int) n_points / batch_size;
    constexpr int param_size = 11644 + 4163; // the param size of the hydra attention and final mlp;
    __shared__ half net_params[param_size];

    int cached_idx = threadIdx.x;
    while (cached_idx < param_size) {
        net_params[cached_idx] = hydra_params[cached_idx];
        cached_idx += blockDim.x;
    }
    __syncthreads();

    // n_points = B * n_pts_per_batch;
    CUDA_KERNEL_LOOP(k, n_points) {
        // get the properties of the inputs' data, b_id, pts_id;
        int batch_idx = (int) (k / n_pts_per_batch);
        int pts_idx   = (int) (k - batch_idx * n_pts_per_batch);
        
        int param_offset = 0;
        int param_index_per_view = 0, param_index_per_view_ = 0, b_rgb_feats_idx;
        // build temp data (1). save inputs and outputs values for each view;
        half multi_view_data[MAX_COLOR_FEAT_HIDDEN_DIM*2 * NUM_VIEWS]; // [DIM,DIM ; DIM,DIM; ...]
        half multi_view_data_[64*2]; // [DIM,DIM ; DIM,DIM; ...]
        half multi_view_norm = __float2half(0.0f); // save q's and k's norm; [1,1; 1,1; ...]

        // feed into the inputs data; [rgb_feats(16), rgbs(3), geo_feats(8), dirs(3)].
        #pragma unroll 4
        for (int i=0; i < num_views; i++) {
            b_rgb_feats_idx = batch_idx*num_views*n_pts_per_batch*FEATS_DIM
                            + i*n_pts_per_batch*FEATS_DIM
                            + pts_idx*FEATS_DIM;
            // b_geo_feats_idx = batch_idx*num_views*n_pts_per_batch*OUTPUT_GEO_FEATS_DIM
            //                 + i*n_pts_per_batch*OUTPUT_GEO_FEATS_DIM
            //                 + pts_idx*OUTPUT_GEO_FEATS_DIM;
            // b_rgbs_idx      = batch_idx*num_views*n_pts_per_batch*3
            //                 + i*n_pts_per_batch*3
            //                 + pts_idx*3;
            // feed into the inputs data; [rgb_feats(16), rgbs(3), geo_feats(8), dirs(3)] 30 dim.
            #pragma unroll 30
            for (int j=0; j<FEATS_DIM; j++) { // feed rgb feats, totally 16 dim;
                multi_view_data[i*MAX_COLOR_FEAT_HIDDEN_DIM*2 + j] = __float2half( feats[b_rgb_feats_idx + j] );
            }
            // #pragma unroll 3
            // for (int j=0; j<3; j++) { // feed rgbs data, totally 3 dim;
            //     multi_view_data[i*MAX_COLOR_FEAT_HIDDEN_DIM*2 + RGB_FEATS_DIM + j] = __float2half( rgbs[b_rgbs_idx + j] );
            // }
            // #pragma unroll 8
            // for (int j=0; j<OUTPUT_GEO_FEATS_DIM; j++) { // feed geo feats, totally 8 dim;
            //     multi_view_data[i*MAX_COLOR_FEAT_HIDDEN_DIM*2 + RGB_FEATS_DIM + 3 + j] = __float2half( geo_feats[b_geo_feats_idx + j] );
            // }
            // #pragma unroll 3
            // for (int j=0; j<3; j++) { // feed geo feats, totally 3 dim;
            //     multi_view_data[i*MAX_COLOR_FEAT_HIDDEN_DIM*2 + RGB_FEATS_DIM + 3 + OUTPUT_GEO_FEATS_DIM + j] = __float2half( dirs[b_rgbs_idx + j] );
            // }
        }
        
        // totally two stack for Hydra-attention-encoder;
        #pragma unroll 2
        for (int s=0; s<NUM_HYDRA_STACKS; s++) {
            // step 1. attention operation;
            half q[MAX_COLOR_FEAT_HIDDEN_DIM*NUM_VIEWS], k[MAX_COLOR_FEAT_HIDDEN_DIM*NUM_VIEWS], kv[MAX_COLOR_FEAT_HIDDEN_DIM]= { __float2half(0.0f) };
            #pragma unroll 4
            for (int i=0; i<num_views; i++) { // using 3 mlp to encode the features;
                param_index_per_view = param_offset; // reset the net param's index;
                
                // w_q, with no bias; save the q right side;
                fc_layer_no_bias<30, 32, 32, MAX_COLOR_FEAT_HIDDEN_DIM>( multi_view_data + i*MAX_COLOR_FEAT_HIDDEN_DIM*2, net_params, param_index_per_view );
                // normalize q;
                #pragma unroll 32
                for (int j=0; j<MAX_COLOR_FEAT_HIDDEN_DIM; j++) { // output **2 sum save to norm-data;
                    multi_view_norm = __hfma( multi_view_data[i*MAX_COLOR_FEAT_HIDDEN_DIM*2 + MAX_COLOR_FEAT_HIDDEN_DIM + j],
                                              multi_view_data[i*MAX_COLOR_FEAT_HIDDEN_DIM*2 + MAX_COLOR_FEAT_HIDDEN_DIM + j],
                                              multi_view_norm);
                }
                // write to q;
                #pragma unroll 32
                for (int j=0; j<MAX_COLOR_FEAT_HIDDEN_DIM; j++) { // data / norm(data)
                    q[i*MAX_COLOR_FEAT_HIDDEN_DIM+j] = __hdiv( multi_view_data[i*MAX_COLOR_FEAT_HIDDEN_DIM*2 + MAX_COLOR_FEAT_HIDDEN_DIM + j], sqrt_half(multi_view_norm) );
                }
                // clear norm;
                multi_view_norm = __float2half(0.0f);

                // w_ks;
                fc_layer_no_bias<30, 32, 32, MAX_COLOR_FEAT_HIDDEN_DIM>( multi_view_data + i*MAX_COLOR_FEAT_HIDDEN_DIM*2, net_params, param_index_per_view );
                // normalize k;
                #pragma unroll 32
                for (int j=0; j<MAX_COLOR_FEAT_HIDDEN_DIM; j++) { // output **2 sum save to norm-data; calculating norm
                    multi_view_norm = __hfma( multi_view_data[i*MAX_COLOR_FEAT_HIDDEN_DIM*2 + MAX_COLOR_FEAT_HIDDEN_DIM + j],
                                              multi_view_data[i*MAX_COLOR_FEAT_HIDDEN_DIM*2 + MAX_COLOR_FEAT_HIDDEN_DIM + j],
                                              multi_view_norm);
                }
                // write to k;
                #pragma unroll 32
                for (int j=0; j<MAX_COLOR_FEAT_HIDDEN_DIM; j++) { // data / norm(data)
                    k[i*MAX_COLOR_FEAT_HIDDEN_DIM+j] = __hdiv( multi_view_data[i*MAX_COLOR_FEAT_HIDDEN_DIM*2 + MAX_COLOR_FEAT_HIDDEN_DIM + j], sqrt_half(multi_view_norm) );
                }
                // clear norm;
                multi_view_norm = __float2half(0.0f);
                
                // w_vs, now v saved in multi_view_data;
                fc_layer_no_bias<30, 32, 32, MAX_COLOR_FEAT_HIDDEN_DIM>( multi_view_data + i*MAX_COLOR_FEAT_HIDDEN_DIM*2, net_params, param_index_per_view );
            }
            // (k * v).sum()
            #pragma unroll 32
            for (int i=0; i<MAX_COLOR_FEAT_HIDDEN_DIM; i++) {
                #pragma unroll 4
                for (int j=0; j<num_views; j++) { // sum the data
                    kv[i] = __hfma(multi_view_data[j*MAX_COLOR_FEAT_HIDDEN_DIM*2 + MAX_COLOR_FEAT_HIDDEN_DIM + i],
                                   k[j*MAX_COLOR_FEAT_HIDDEN_DIM+i], kv[i]);
                }
                // q * kv, saved in q;
                #pragma unroll 4
                for (int j=0; j<num_views; j++) {
                    q[j*MAX_COLOR_FEAT_HIDDEN_DIM+i] = __hmul(q[j*MAX_COLOR_FEAT_HIDDEN_DIM+i], kv[i]);
                }
            }
            
            // fc (without bias); rewrite to multi_view_data (offset with MAX_COLOR_FEAT_HIDDEN_DIM)
            half res[COLOR_FEATS_DIM]; // tmp
            #pragma unroll 4
            for (int n=0; n<num_views; n++) {
                // fc results saved in right side, without bias;
                param_index_per_view_ = param_index_per_view;
                #pragma unroll 30
                for (int i=0; i<30; i++) {
                    multi_view_data[n*MAX_COLOR_FEAT_HIDDEN_DIM*2+MAX_COLOR_FEAT_HIDDEN_DIM + i] = __float2half( 0.0f ); // load the 0 to last ;
                }
                #pragma unroll 30
                for (int i=0; i<30; i++) {
                    #pragma unroll 32
                    for (int j=0; j<32; j++) {
                        multi_view_data[n*MAX_COLOR_FEAT_HIDDEN_DIM*2+MAX_COLOR_FEAT_HIDDEN_DIM + i] = 
                            __hfma( net_params[param_index_per_view_++], q[n*MAX_COLOR_FEAT_HIDDEN_DIM+j],
                                    multi_view_data[n*MAX_COLOR_FEAT_HIDDEN_DIM*2+MAX_COLOR_FEAT_HIDDEN_DIM + i] );
                    }
                }

                // add the residual, add to res here; res = out [..., right] + input [left, ...]
                #pragma unroll 30
                for (int i=0; i<COLOR_FEATS_DIM; i++) {
                    res[i] = __hadd(multi_view_data[n*MAX_COLOR_FEAT_HIDDEN_DIM*2 + i],
                                    multi_view_data[n*MAX_COLOR_FEAT_HIDDEN_DIM*2+MAX_COLOR_FEAT_HIDDEN_DIM + i]);
                    multi_view_data[n*MAX_COLOR_FEAT_HIDDEN_DIM*2 + i] = res[i]; // also copy to left;
                }

                // w1, save in the right;
                fc_layer<30, 32, 32, MAX_COLOR_FEAT_HIDDEN_DIM>( multi_view_data + n*MAX_COLOR_FEAT_HIDDEN_DIM*2, net_params, param_index_per_view_ );
                relu_tensor<MAX_COLOR_FEAT_HIDDEN_DIM>(multi_view_data + n*MAX_COLOR_FEAT_HIDDEN_DIM*2 + MAX_COLOR_FEAT_HIDDEN_DIM);
                // w2, save in the left;
                fc_layer<32, 30, 0, MAX_COLOR_FEAT_HIDDEN_DIM>( multi_view_data + n*MAX_COLOR_FEAT_HIDDEN_DIM*2, net_params, param_index_per_view_ );
                // add the residual before;
                #pragma unroll 30
                for (int i=0; i<COLOR_FEATS_DIM; i++) { // add the previous residual inputs;
                    multi_view_data[n*MAX_COLOR_FEAT_HIDDEN_DIM*2 + i] = __hadd(multi_view_data[n*MAX_COLOR_FEAT_HIDDEN_DIM*2 + i],
                                                                                res[i]);
                }
            }
            
            param_offset += 5822;
        }
        // now, the index is 11644;
        // printf("the index : %d", param_index_per_view_);

        // write to global memory, output_color_feats: [B, N_points, 30], rgbs : [B, N_points, 3];

        #pragma unroll 30
        for (int i=0; i<COLOR_FEATS_DIM; i++ ) { // copy from left, first view info, start from 0;
            outputs_color_feats[batch_idx*n_pts_per_batch*COLOR_FEATS_DIM+pts_idx*COLOR_FEATS_DIM + i] = __half2float( multi_view_data[i] );
            multi_view_data_[i] = multi_view_data[i]; // copy to new register;
        }
        
        // printf("the index : %d\n", param_index_per_view_);

        // rgb mlp; 3 layers[1.w_c, 2.relu, 3.wc, 4.relu, 5.sigmoid];
        fc_layer<30, 64, 64, 64>( multi_view_data_, net_params, param_index_per_view_ );
        relu_tensor<64>(multi_view_data_ + 64);
        fc_layer<64, 32, 0, 64>( multi_view_data_, net_params, param_index_per_view_ );
        relu_tensor<32>(multi_view_data_);
        fc_layer<32, 3, 64, 64>( multi_view_data_, net_params, param_index_per_view_ );
        sigmoid_tensor<3>(multi_view_data_ + 64);
        
        // param_index_per_view_ is now 15807 here;
        // printf("the index new: %d\n", param_index_per_view_);

        // to global memory;
        #pragma unroll 3
        for (int i=0; i<3; i++ ) {
            outputs_color[batch_idx*n_pts_per_batch*3+pts_idx*3+i] = __half2float( multi_view_data_[64+i] );
        }

        // printf("write to global memory\n");
    }
}


// upsampling MKP kernel;
__global__ void upsampling_kernel(
    const __half * mlp_params, // one dim's tensor;
    const float * feats_neighbors, // [N, 40];
    const float * ray_feats, // [N, 30];
    const int n_points, // N
    float * outputs_weights // [N_points, 3]
) {
    constexpr int param_size = 15075; // the param size of the hydra attention and final mlp;
    __shared__ half net_params[param_size];

    int cached_idx = threadIdx.x;
    while (cached_idx < param_size) {
        net_params[cached_idx] = mlp_params[cached_idx];
        cached_idx += blockDim.x;
    }
    __syncthreads();

    half layer_data[64 * 2];

    CUDA_KERNEL_LOOP(k, n_points) {
        // per-view data;
        int param_index = 0;

        // first layers' input: feed inputs to layer data's first 64 dim;
        #pragma unroll 40
        for (int i=0; i<40; i++) {
            layer_data[i] = __float2half( feats_neighbors[k*40+i] ); // first 40 dim as inputs;
        }

        // parse feature MLP's each layers to obtain output features; totally four layers;
        fc_layer<40, 64, 64, 64>(layer_data, net_params, param_index);
        relu_tensor<64>(layer_data+64);
        fc_layer<64, 64, 0, 64>(layer_data, net_params, param_index);
        relu_tensor<64>(layer_data);
        fc_layer<64, 32, 64, 64>(layer_data, net_params, param_index);
        relu_tensor<32>(layer_data+64); // save the data in the right;

        // concat the ray_feats;
        #pragma unroll 30
        for (int i=0; i<30; i++) {
            layer_data[64+32+i] = __float2half( ray_feats[k*30+i] );
        }
        
        fc_layer<62, 64, 0, 64>(layer_data, net_params, param_index);
        relu_tensor<64>(layer_data);

        fc_layer<64, 32, 64, 64>(layer_data, net_params, param_index);
        relu_tensor<32>(layer_data+64);
        fc_layer<32, 3, 0, 64>(layer_data, net_params, param_index);
        sigmoid_tensor<3>(layer_data); // final activation sigmoid;

        // save to global memory;
        #pragma unroll 3
        for (int j=0; j<3; j++) {
            outputs_weights[3*k+j] = __half2float( layer_data[j] );
        }
    }
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
        #pragma unroll
        for (int i=0; i < FEATS_DIM; i++) {
            output_feats[basic_ray_idx+i] = feat[i]; // rgb is assigned as 0;
        }
        // 
    }
}

torch::Tensor infer_density_mlp(
    // now all data are [N_points, C]
    torch::Tensor points_z, // [B, N_view, N_rays, N_points, 1]
    torch::Tensor points_tpsdf, // [B, N_view, N_rays, N_points, 1]
    torch::Tensor sampled_feats_geo, // [B, N_view, N_rays, N_points, 16]
    // mlp
    torch::Tensor net_params_density,
    const int batch_size, const int num_points, const int num_views,
    const int device
) {
    cudaSetDevice( device ); // on GPU device.
    CUDA_CHECK_ERRORS();

    const int N_v_points = batch_size * num_points * num_views;
    torch::Tensor output_density = torch::empty({batch_size*num_points, 1}, points_z.options()); //
    torch::Tensor output_density_feats = torch::empty({N_v_points, DENSITY_FEATS_DIM}, points_z.options());
    torch::Tensor output_geo_feats     = torch::empty({N_v_points, OUTPUT_GEO_FEATS_DIM}, points_z.options()); // original num of points;

    static constexpr uint32_t block_size = 256;
    uint32_t grid_size_feat = (uint32_t) (N_v_points + block_size - 1) / block_size;
    cudaStream_t curr_stream = at::cuda::getCurrentCUDAStream();

    density_feat_kernel <<< grid_size_feat, block_size, 0, curr_stream>>> (
        (__half*)net_params_density.contiguous().data_ptr<torch::Half>(),
        points_z.contiguous().data_ptr<float>(),
        points_tpsdf.contiguous().data_ptr<float>(),
        sampled_feats_geo.contiguous().data_ptr<float>(),
        N_v_points,
        output_density_feats.contiguous().data_ptr<float>(),
        output_geo_feats.contiguous().data_ptr<float>()
    );
    
    // ***** mean features ######
    output_density_feats = output_density_feats.view({batch_size, num_views, num_points, DENSITY_FEATS_DIM});
    output_density_feats = torch::mean(output_density_feats, 1).view({batch_size*num_points, DENSITY_FEATS_DIM});
    // ##########################
    uint32_t grid_size_density = (uint32_t) (batch_size*num_points + block_size - 1) / block_size;

    density_sigma_kernel <<< grid_size_density, block_size, 0, curr_stream>>> (
        (__half*)net_params_density.contiguous().data_ptr<torch::Half>(),
        output_density_feats.contiguous().data_ptr<float>(),
        batch_size*num_points, 
        output_density.contiguous().data_ptr<float>()
    );
    // output occ
    output_density = output_density.view({batch_size, 1, num_points}); // [B, N_pts, 1];

    cudaDeviceSynchronize();
    CUDA_CHECK_ERRORS();

    return output_density;
}

// nerf function;
// speed up x2.5 (more) 
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> infer_fast_rendering(
    // inputs, now are all [N_points, C]
    torch::Tensor sorted_dists, // [B, N_rays, N_points]
    //
    torch::Tensor points_z, // [B, N_view, N_rays, N_points, 1]
    torch::Tensor points_tpsdf, // [B, N_view, N_rays, N_points, 1]
    torch::Tensor sampled_feats_geo, // [B, N_view, N_rays, N_points, 16]
    torch::Tensor sampled_rgbs, // [B, N_view', N_rays, N_points, 3]
    torch::Tensor sampled_feats_rgb, // [B, N_view', N_rays, N_points, 16]
    torch::Tensor rays_dir, // [B, N_view', N_rays, N_points, 3]
    // the parameters of the MLP;
    torch::Tensor net_params_density, // [N_inputs*W0+B0 + W0*W1+B1 + ....] one dim tensor;
    torch::Tensor net_params_color, // [N_inputs*W0+B0 + W0*W1+B1 + ....] one dim tensor;
    // properties;
    const int batch_size, const int n_rays, const int n_sampled, // e.g., 1, 1024, 64
    const int geo_num_views, const int tex_num_views, // e.g., 8, 4
    const int device
) {
    cudaSetDevice( device ); // on GPU device.
    CUDA_CHECK_ERRORS();
    
    const int N_v_points = points_z.size(0);
    const int N_v_points_ = sampled_rgbs.size(0);
    const int N_points = (const int) N_v_points / (geo_num_views * batch_size); // n_rays * n_sampled
    // hidden data;
    torch::Tensor output_density_feats = torch::empty({N_v_points, DENSITY_FEATS_DIM}, points_z.options());
    torch::Tensor output_geo_feats     = torch::empty({N_v_points, OUTPUT_GEO_FEATS_DIM}, points_z.options()); // original num of points;
    // the output data, we output the color features & density, density features; [B, N_points, ]
    torch::Tensor output_color_feats   = torch::empty({batch_size, n_rays, COLOR_FEATS_DIM}, sampled_feats_rgb.options()); //
    torch::Tensor output_density       = torch::empty({batch_size*N_points, 1}, points_z.options()); // 
    torch::Tensor output_color         = torch::empty({batch_size, n_rays, 3}, sampled_rgbs.options()); //
    // tmp output;
    torch::Tensor output_dists         = torch::empty({batch_size, n_rays, 1}, points_z.options()); // 
    torch::Tensor output_feats         = torch::empty({batch_size, tex_num_views, n_rays, FEATS_DIM}, points_z.options()); // 

    // torch::Tensor output_density_feats_new = torch::empty({N_v_points, DENSITY_FEATS_DIM}, points_z.options());
    // float * density_feats_ptr = output_density_feats.contiguous().data_ptr<float>();

    // seperate two streams to execute kernels;
    cudaStream_t streams[NUM_NERF_STREAMS+NUM_VIEWS];
    // s1. stream for density and color
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    
    // int block_size, min_grid_size;
    // cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, density_feat_kernel, 0, 0);
    static constexpr uint32_t block_size = 256;
    uint32_t grid_size_feat = (uint32_t) (N_v_points + block_size - 1) / block_size;

    density_feat_kernel <<< grid_size_feat, block_size, 0, streams[0] >>> (
        (__half*)net_params_density.contiguous().data_ptr<torch::Half>(),
        points_z.contiguous().data_ptr<float>(),
        points_tpsdf.contiguous().data_ptr<float>(),
        sampled_feats_geo.contiguous().data_ptr<float>(),
        N_v_points, 
        output_density_feats.contiguous().data_ptr<float>(),
        output_geo_feats.contiguous().data_ptr<float>()
    );

    /* reduce features and reshape to [N_points, 64] */
    output_density_feats = output_density_feats.view({batch_size, geo_num_views, n_rays, n_sampled, DENSITY_FEATS_DIM});
    output_density_feats = torch::mean(output_density_feats, 1).view({batch_size*N_points, DENSITY_FEATS_DIM});
    // select the first N' views' geometry features; [..., :4] and reshape to [B, N', n_rays*n_sampled, 8];
    output_geo_feats     = output_geo_feats.view({batch_size, geo_num_views, n_rays, n_sampled, OUTPUT_GEO_FEATS_DIM});
    output_geo_feats     = output_geo_feats.slice(1, 0, tex_num_views).view({batch_size, tex_num_views, n_rays, n_sampled, OUTPUT_GEO_FEATS_DIM});
    /*******************/

    // cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, density_sigma_kernel, 0, 0);
    uint32_t grid_size_density = (uint32_t) (batch_size*N_points + block_size - 1) / block_size;

    density_sigma_kernel <<< grid_size_density, block_size, 0, streams[0] >>> (
        (__half*)net_params_density.contiguous().data_ptr<torch::Half>(),
        output_density_feats.contiguous().data_ptr<float>(),
        batch_size*N_points, 
        output_density.contiguous().data_ptr<float>()
    );
    // output occ
    output_density = output_density.view({batch_size, n_rays, n_sampled, 1}); // [B, N_pts, 1];

    // reshape the input data, to [B, N_views, N_rays*N_sampled, C];
    sampled_feats_rgb = sampled_feats_rgb.view({batch_size, tex_num_views, n_rays, n_sampled, RGB_FEATS_DIM});
    sampled_rgbs      = sampled_rgbs.view({batch_size, tex_num_views, n_rays, n_sampled, 3});
    rays_dir          = rays_dir.view({batch_size, tex_num_views, n_rays, n_sampled, 3}); 

    uint32_t grid_size_vr = (uint32_t) (batch_size*n_rays + block_size - 1) / block_size;

    // PRE volume_rendering int, to get the features; rgb_feats(16), rgbs(3), geo_feats(8), dirs(3)
    torch::Tensor feats_v, feats;
    #pragma unroll
    for(int i=0; i<tex_num_views; i++) {
        feats_v = torch::cat({ sampled_feats_rgb.select(1,i), 
                               sampled_rgbs.select(1,i),
                               output_geo_feats.select(1,i),
                               rays_dir.select(1,i)}, 3 ); // [B, N_rays, N_sampled, C=30];
        feats = torch::empty({batch_size, n_rays, FEATS_DIM}, points_z.options());
        cudaStreamCreate(&streams[2+i]);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            sorted_dists.scalar_type(), "volume_rendering_occ_kernel", ([&] {
                volume_rendering_occ_kernel<scalar_t> <<< grid_size_vr, block_size, 0, streams[2+i] >>> (
                    sorted_dists.contiguous().data_ptr<scalar_t>(),
                    output_density.contiguous().data_ptr<scalar_t>(),
                    feats_v.contiguous().data_ptr<scalar_t>(), // [B, N_ray, N_sampled, FEATS_DIM]
                    feats.contiguous().data_ptr<scalar_t>(),
                    output_dists.contiguous().data_ptr<scalar_t>(),
                    batch_size, n_rays, n_sampled, 1e-4
                );
            })
        );
        output_feats.select(1, i) = feats;
    }

    // s2. stream for color, inputs are {geo_features, rgb_features, ray_dirs, sampled_rgbs};
    
    // allocate block size;
    // cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, hydra_attention_kernel, 0, 0);
    uint32_t grid_size_hydra = (uint32_t) (batch_size*n_rays + block_size - 1) / block_size;
    
    // hydra attention.
    // wait the output_geo_feats finished;
    hydra_attention_kernel <<< grid_size_hydra, block_size, 0, streams[1] >>> (
        (__half*)net_params_color.contiguous().data_ptr<torch::Half>(), // one dimension tensor;
        // sampled_feats_rgb.contiguous().data_ptr<float>(), // [B,N_views,N_points,16];
        // output_geo_feats.contiguous().data_ptr<float>(), // [B,N_views,N_points,8];
        // sampled_rgbs.contiguous().data_ptr<float>(), // [B,N_view,N_points,3];
        // rays_dir.contiguous().data_ptr<float>(), // [B, N_v, N_points, 3];
        output_feats.contiguous().data_ptr<float>(), // [B, N_v, N_points, 30];
        // properties and output;
        batch_size*n_rays, batch_size, tex_num_views, 
        output_color_feats.contiguous().data_ptr<float>(), // [B,N_points,30];
        output_color.contiguous().data_ptr<float>() // [B,N_points,3]
    );
    // printf( "the shape : %d, %d", output_color_feats.size(1), output_color.size(1) ); // 
    
    cudaDeviceSynchronize();

    // reshape the output data, [B, N_rays, N_sampled, 1 or 3 or C];
    // output_density = output_density.view({batch_size, n_rays, n_sampled, 1}); // [B, N_pts, 1];
    // output_color_feats = output_color_feats.view({batch_size, n_rays, n_sampled, COLOR_FEATS_DIM}); // [B, N_rays, N_pts_sampled, C=30];
    // output_color = output_color.view({batch_size, n_rays, n_sampled, 3}); // [B, N_rays, N_pts_sampled, C=3];
    
    // cudaFree(output_density_feats.contiguous().data_ptr<float>());
    // cudaFree(output_geo_feats.contiguous().data_ptr<float>());
    // cudaFree(output_density.contiguous().data_ptr<float>());
    // cudaFree(output_feats.contiguous().data_ptr<float>());

    for(int i=0; i<NUM_NERF_STREAMS+NUM_VIEWS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    CUDA_CHECK_ERRORS();
    
    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>{output_dists, output_color, output_color_feats};
}

// predict the upsampling fusion weights;
// speed up x3 (more) 
torch::Tensor infer_upsampling(
    torch::Tensor feats_neighbors, // [16 + 3 + 1] * 2; 40 dim, [N_rays, 40]
    torch::Tensor ray_rgb_feats, // [N_rays, 30]
    torch::Tensor net_params, // [N_inputs*W0+B0 + W0*W1+B1 + ....] one dim tensor;
    const int device
) {
    cudaSetDevice( device ); // on GPU device
    CUDA_CHECK_ERRORS();

    const int N_points = feats_neighbors.size(0);
    // returned properties;
    torch::Tensor output_weights = torch::empty({N_points, 3}, feats_neighbors.options());

    // int block_size, min_grid_size;
    // cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, upsampling_kernel, 0, 0);
    static constexpr uint32_t block_size = 256;
    uint32_t grid_size_up = (uint32_t) (N_points + block_size - 1) / block_size;

    upsampling_kernel <<< grid_size_up, block_size >>>(
        (__half*)net_params.contiguous().data_ptr<torch::Half>(), // one dimension tensor;
        feats_neighbors.contiguous().data_ptr<float>(),
        ray_rgb_feats.contiguous().data_ptr<float>(),
        N_points, // N
        output_weights.contiguous().data_ptr<float>()
    );

    CUDA_CHECK_ERRORS();
    return output_weights;
}