#include <math.h>
#include <vector>
#include <stdint.h>
#include <assert.h>

// #include <omp.h>
#include "../include/cuda_helper.h"
#include "../include/octree.h"
#include "../include/sampling.h"
#include <string>
#include <stdio.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>

template <typename scalar_t>
__global__ void inverse_mat3x3(
    const scalar_t* Ms, // [B, N, 3, 3]
    const uint32_t batch_num, 
    const uint32_t num_views,
    scalar_t* inv_Ms
) {
    CUDA_KERNEL_LOOP(i, batch_num*num_views) {
        uint32_t batch_id = (uint32_t) (i / num_views);
        uint32_t view_id  = i - batch_id * num_views;
        uint32_t basic_id = batch_id * num_views * 9 + view_id * 9;
        scalar_t a1 = Ms[basic_id+0],
              b1 = Ms[basic_id+1],
              c1 = Ms[basic_id+2],
              a2 = Ms[basic_id+3],
              b2 = Ms[basic_id+4],
              c2 = Ms[basic_id+5],
              a3 = Ms[basic_id+6],
              b3 = Ms[basic_id+7],
              c3 = Ms[basic_id+8];
        scalar_t det = a1*(b2*c3-c2*b3)-a2*(b1*c3-c1*b3)+a3*(b1*c2-c1*b2);
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

template <typename T>
inline __device__ T lerp(const T &l0, const T &l1, float t) {
    return l0 * (1 - t) + l1 * t;
}

__global__ void project_xyz( // return the projected xy [0, original_x]; [0, original_y]
    OcTree *octree_nodes, // all nodes.
    const float * ray_ori, // [B, N_rays, 3]
    const float * ray_dir, // [B, N_rays, 3]
    const float * Ks, //  [B, N_views, 3, 3]
    const float * RTs, // [B, N_views, 3, 4]
    const int * sampled_idx, // [B, N_ray, N_sampled]
    const float * sampled_depths, // [B, N_ray, N_sampled]
    float * proj_xy, // output projected xy position. [B, N_view, N_ray, N_sampled, 9, 2]
    float * proj_z, // output projected xy position. [B, N_view, N_ray, N_sampled, 9, 1]
    int ori_w, int ori_h,
    int batch_size,
    int num_rays,
    int num_sampled,
    int num_views,
    bool record_corners
) {
    #pragma unroll
    CUDA_KERNEL_LOOP(k, batch_size * num_views * num_rays * num_sampled) {
        // get idx.
        int batch_idx   = (int) (k / (num_views * num_rays * num_sampled));
        int view_idx    = (int) ((k - batch_idx * num_views * num_rays * num_sampled) / (num_rays * num_sampled));
        int ray_idx     = (int) ((k - batch_idx*num_views*num_rays*num_sampled - view_idx*num_rays*num_sampled) / num_sampled);
        int sam_idx     = (int) (k - batch_idx*num_views*num_rays*num_sampled - view_idx*num_rays*num_sampled - ray_idx*num_sampled);
        
        size_t basic_sampled_idx = batch_idx*num_rays*num_sampled + ray_idx*num_sampled + sam_idx;
        int basic_ray_idx = batch_idx*num_rays*3 + ray_idx*3;
        // get sampled idx & z.
        int voxel_idx = sampled_idx[basic_sampled_idx];
        if (voxel_idx == -1) continue; // not valid voxel's idx for point sampling.

        size_t idx_RT = batch_idx*num_views*12 + view_idx*12;
        size_t idx_K  = batch_idx*num_views*9  + view_idx*9;
        float sampled_z = sampled_depths[basic_sampled_idx];
        float3 ray_ori_pz = make_float3(ray_ori[basic_ray_idx+0], ray_ori[basic_ray_idx+1], ray_ori[basic_ray_idx+2]);
        float3 ray_dir_pz = make_float3(ray_dir[basic_ray_idx+0], ray_dir[basic_ray_idx+1], ray_dir[basic_ray_idx+2]);
        float3 ray_sampled_pz = make_float3(ray_ori_pz.x + ray_dir_pz.x * sampled_z, 
                                            ray_ori_pz.y + ray_dir_pz.y * sampled_z,
                                            ray_ori_pz.z + ray_dir_pz.z * sampled_z);
        // B, N_view, N_ray, N_sampled, 3;
        size_t project_xy_basic_idx, project_z_basic_idx;
        if (record_corners) {
            project_xy_basic_idx = batch_idx*num_views*num_rays*num_sampled*18 + view_idx*num_rays*num_sampled*18
                                 + ray_idx*num_sampled*18 + sam_idx*18;
            project_z_basic_idx = batch_idx*num_views*num_rays*num_sampled*9 + view_idx*num_rays*num_sampled*9
                                + ray_idx*num_sampled*9 + sam_idx*9;
        } else {
            project_xy_basic_idx = batch_idx*num_views*num_rays*num_sampled*2 + view_idx*num_rays*num_sampled*2
                                 + ray_idx*num_sampled*2 + sam_idx*2;
            project_z_basic_idx = batch_idx*num_views*num_rays*num_sampled*1 + view_idx*num_rays*num_sampled*1
                                + ray_idx*num_sampled*1 + sam_idx*1;
        }
                                 
        // 1. get the pixel location
        float pixel_x, pixel_y, pixel_z, cam_pt_x, cam_pt_y, cam_pt_z;
        cam_pt_x = RTs[0+idx_RT]*ray_sampled_pz.x + RTs[1+idx_RT]*ray_sampled_pz.y + RTs[2+idx_RT] *ray_sampled_pz.z + RTs[3+idx_RT];
        cam_pt_y = RTs[4+idx_RT]*ray_sampled_pz.x + RTs[5+idx_RT]*ray_sampled_pz.y + RTs[6+idx_RT] *ray_sampled_pz.z + RTs[7+idx_RT];
        cam_pt_z = RTs[8+idx_RT]*ray_sampled_pz.x + RTs[9+idx_RT]*ray_sampled_pz.y + RTs[10+idx_RT]*ray_sampled_pz.z + RTs[11+idx_RT];
        
        pixel_x = Ks[0+idx_K]*cam_pt_x + Ks[1+idx_K]*cam_pt_y + Ks[2+idx_K]*cam_pt_z;
        pixel_y = Ks[3+idx_K]*cam_pt_x + Ks[4+idx_K]*cam_pt_y + Ks[5+idx_K]*cam_pt_z;
        pixel_z = Ks[6+idx_K]*cam_pt_x + Ks[7+idx_K]*cam_pt_y + Ks[8+idx_K]*cam_pt_z;
        pixel_x = pixel_x / pixel_z;
        pixel_y = pixel_y / pixel_z;
        // to [-1,1] coordinates.
        pixel_x = (pixel_x - ori_w / 2.0) / (ori_w / 2.0);
        pixel_y = (pixel_y - ori_h / 2.0) / (ori_h / 2.0);

        // the first xyz is the projected corners' positions.
        proj_xy[project_xy_basic_idx+0] = pixel_x;
        proj_xy[project_xy_basic_idx+1] = pixel_y;
        proj_z[project_z_basic_idx+0]   = cam_pt_z;

        if (record_corners) {
            // 2. get the pixel location of the eight corners.
            OcTree curr_node = octree_nodes[voxel_idx];
            float3 xyz_corner;
            
            #pragma unroll 8
                for (int i=0; i < 8; i++) { // for all leaves.
                    xyz_corner = curr_node.xyz_corners[i];
                    // printf("x, y, z: %f, %f, %f \n", xyz_corner.x, xyz_corner.y, xyz_corner.z);

                    cam_pt_x = RTs[0+idx_RT]*xyz_corner.x + RTs[1+idx_RT]*xyz_corner.y + RTs[2+idx_RT] *xyz_corner.z + RTs[3+idx_RT];
                    cam_pt_y = RTs[4+idx_RT]*xyz_corner.x + RTs[5+idx_RT]*xyz_corner.y + RTs[6+idx_RT] *xyz_corner.z + RTs[7+idx_RT];
                    cam_pt_z = RTs[8+idx_RT]*xyz_corner.x + RTs[9+idx_RT]*xyz_corner.y + RTs[10+idx_RT]*xyz_corner.z + RTs[11+idx_RT];
                    
                    pixel_x = Ks[0+idx_K]*cam_pt_x + Ks[1+idx_K]*cam_pt_y + Ks[2+idx_K]*cam_pt_z;
                    pixel_y = Ks[3+idx_K]*cam_pt_x + Ks[4+idx_K]*cam_pt_y + Ks[5+idx_K]*cam_pt_z;
                    pixel_z = Ks[6+idx_K]*cam_pt_x + Ks[7+idx_K]*cam_pt_y + Ks[8+idx_K]*cam_pt_z;
                    pixel_x = pixel_x / pixel_z;
                    pixel_y = pixel_y / pixel_z;
                    // to [-1,1] coordinates.
                    pixel_x = (pixel_x - ori_w / 2.0) / (ori_w / 2.0);
                    pixel_y = (pixel_y - ori_h / 2.0) / (ori_h / 2.0);

                    proj_xy[project_xy_basic_idx+(i+1)*2+0] = pixel_x;
                    proj_xy[project_xy_basic_idx+(i+1)*2+1] = pixel_y;
                    proj_z[project_z_basic_idx+i+1]         = cam_pt_z;
                }
        }
        
        // if (pixel_x < 0 || pixel_x >= ori_w || pixel_y < 0 || pixel_y >= ori_h || cam_pt_z < 0) continue;
    }
}


__global__ void trilinear_aggregate_features_kernel( // aggregate 8 corners's features.
    OcTree *octree_nodes, // all nodes.
    const float * ray_ori, // [B, N_rays, 3]
    const float * ray_dir, // [B, N_rays, 3]
    // const float * Ks, //  [B, N_views, 3, 3]
    // const float * RTs, // [B, N_views, 3, 4]
    const int * sampled_idx, // [B, N_ray, N_sampled]
    const float * sampled_depths, // [B, N_ray, N_sampled]
    const float * input_feats, // [B, C, N_ray, N_sampled, 8]
    float * output_feats, // [B, C, N_ray, N_sampled]
    // int ori_w, int ori_h,
    int batch_size,
    int num_rays,
    int num_sampled,
    int dim_feats
) {
    #pragma unroll
    CUDA_KERNEL_LOOP(k, batch_size * num_rays * num_sampled * dim_feats) {
        // get idx.
        int batch_idx = (int) (k / (num_rays * num_sampled * dim_feats));
        int ray_idx   = (int) ((k - batch_idx*num_rays*num_sampled*dim_feats) / (num_sampled*dim_feats));
        int sam_idx   = (int) ((k - batch_idx*num_rays*num_sampled*dim_feats - ray_idx*num_sampled*dim_feats) / dim_feats);
        int feat_idx  = (int) (k - batch_idx*num_rays*num_sampled*dim_feats - ray_idx*num_sampled*dim_feats - sam_idx*dim_feats);
        
        size_t basic_sampled_idx = batch_idx*num_rays*num_sampled + ray_idx*num_sampled + sam_idx;
        int basic_ray_idx = batch_idx*num_rays*3 + ray_idx*3;
        // get sampled idx & z.
        int voxel_idx = sampled_idx[basic_sampled_idx];
        if (voxel_idx == -1) continue; // not valid voxel's idx for point sampling.
        
        float sampled_z = sampled_depths[basic_sampled_idx];
        float3 ray_ori_pz = make_float3(ray_ori[basic_ray_idx+0], ray_ori[basic_ray_idx+1], ray_ori[basic_ray_idx+2]);
        float3 ray_dir_pz = make_float3(ray_dir[basic_ray_idx+0], ray_dir[basic_ray_idx+1], ray_dir[basic_ray_idx+2]);
        float3 ray_sampled_pz = make_float3(ray_ori_pz.x + ray_dir_pz.x * sampled_z, 
                                            ray_ori_pz.y + ray_dir_pz.y * sampled_z,
                                            ray_ori_pz.z + ray_dir_pz.z * sampled_z);
        // get 8 corners, positions.
        OcTree curr_node = octree_nodes[voxel_idx];
        
        // to registers.
        const float3 *cx = &curr_node.xyz_corners[0];
        
        // for (int i=0; i < 8; i++) {
        //     printf("x,y,z: %f, %f, %f \n", all_corners[i].x, all_corners[i].y, all_corners[i].z);
        // }
        // printf("x0,x0,x0,x0: %f, %f, %f, %f \n", cx[0].x, cx[1].x, cx[2].x, cx[3].x);
        // printf("x1,x1,x1,x1: %f, %f, %f, %f \n", cx[4].x, cx[5].x, cx[6].x, cx[7].x);
        // 1. get features of the feat_idx.
        const float *fts = &input_feats[batch_idx*dim_feats*num_rays*num_sampled*MAX_LEAVES
                                       +feat_idx*num_rays*num_sampled*MAX_LEAVES
                                       +ray_idx*num_sampled*MAX_LEAVES
                                       +sam_idx*MAX_LEAVES];
        
        // printf("x0,x1,x2,x3: %f, %f, %f, %f \n", feat_corners[0], feat_corners[1], feat_corners[2], feat_corners[7]);
        // 2. lerp the feature along x axis. [x0, x4], [x1, x5], [x2, x6], [x3, x7];
        const float x04 = lerp(fts[0], fts[4], (ray_sampled_pz.x - cx[0].x) / (cx[4].x - cx[0].x));
        const float x15 = lerp(fts[1], fts[5], (ray_sampled_pz.x - cx[1].x) / (cx[5].x - cx[1].x));
        const float x26 = lerp(fts[2], fts[6], (ray_sampled_pz.x - cx[2].x) / (cx[6].x - cx[2].x));
        const float x37 = lerp(fts[3], fts[7], (ray_sampled_pz.x - cx[3].x) / (cx[7].x - cx[3].x));
        // printf("x0,x0,x0,x0: %f, %f, %f, %f \n", 
        //     (ray_sampled_pz.x - cx[0].x) / (cx[4].x - cx[0].x), 
        //     (ray_sampled_pz.x - cx[1].x) / (cx[5].x - cx[1].x), 
        //     (ray_sampled_pz.x - cx[2].x) / (cx[6].x - cx[2].x),
        //     (ray_sampled_pz.x - cx[3].x) / (cx[7].x - cx[3].x));
            
        const float y04_26 = lerp(x04, x26, (ray_sampled_pz.y - (cx[0].y + cx[4].y) / 2.0) / ((cx[2].y + cx[6].y) / 2.0 - (cx[0].y + cx[4].y) / 2.0));
        const float y15_37 = lerp(x15, x37, (ray_sampled_pz.y - (cx[1].y + cx[5].y) / 2.0) / ((cx[3].y + cx[7].y) / 2.0 - (cx[1].y + cx[5].y) / 2.0));
        const float z0426_1537 = lerp(y04_26, y15_37, (ray_sampled_pz.z - (cx[0].z + cx[4].z + cx[2].z + cx[6].z) / 4.0) / ((cx[1].z + cx[5].z + cx[3].z + cx[7].z) / 4.0 - (cx[0].z + cx[4].z + cx[2].z + cx[6].z) / 4.0));
        // printf("x0,x0,x0,x0: %f, %f, %f, %f, %f \n", (cx[0].y + cx[4].y) / 2.0, cx[4].y, cx[0].y, (cx[0].z + cx[4].z + cx[2].z + cx[6].z) / 4.0, cx[4].z);
        // the outputed features.
        output_feats[batch_idx*dim_feats*num_rays*num_sampled + feat_idx*num_rays*num_sampled + ray_idx*num_sampled + sam_idx] = z0426_1537;
    }
}


__global__ void trilinear_aggregate_features_backward_kernel( // aggregate 8 corners's features.
    OcTree *octree_nodes, // all nodes.
    const float * ray_ori, // [B, N_rays, 3]
    const float * ray_dir, // [B, N_rays, 3]
    // const float * Ks, //  [B, N_views, 3, 3]
    // const float * RTs, // [B, N_views, 3, 4]
    const int * sampled_idx, // [B, N_ray, N_sampled]
    const float * sampled_depths, // [B, N_ray, N_sampled]
    const float * grad_out, // d_Loss / d_{output_feats}: [B, C, N_ray, N_sampled]
    float *grad_data, // output gradients: [B, C, N_ray, N_sampled, 8]
    // int ori_w, int ori_h,
    int batch_size,
    int num_rays,
    int num_sampled,
    int dim_feats
) {
    // backward the output data's gradients to inputs' tensor.
    #pragma unroll
    CUDA_KERNEL_LOOP(k, batch_size * num_rays * num_sampled * dim_feats) {
        // get idx。
        int batch_idx = (int) (k / (num_rays * num_sampled * dim_feats));
        int ray_idx   = (int) ((k - batch_idx*num_rays*num_sampled*dim_feats) / (num_sampled*dim_feats));
        int sam_idx   = (int) ((k - batch_idx*num_rays*num_sampled*dim_feats - ray_idx*num_sampled*dim_feats) / dim_feats);
        int feat_idx  = (int) (k - batch_idx*num_rays*num_sampled*dim_feats - ray_idx*num_sampled*dim_feats - sam_idx*dim_feats);

        size_t basic_sampled_idx = batch_idx*num_rays*num_sampled + ray_idx*num_sampled + sam_idx;
        int basic_ray_idx = batch_idx*num_rays*3 + ray_idx*3;
        // get sampled idx & z.
        int voxel_idx = sampled_idx[basic_sampled_idx];
        if (voxel_idx == -1) continue; // not valid voxel's idx for point sampling, the gradients are set to 0;

        size_t grad_out_idx = batch_idx*dim_feats*num_rays*num_sampled
                            + feat_idx*num_rays*num_sampled
                            + ray_idx*num_sampled
                            + sam_idx;
        size_t grad_data_basic_idx = batch_idx*dim_feats*num_rays*num_sampled*MAX_LEAVES
                                   + feat_idx*num_rays*num_sampled*MAX_LEAVES
                                   + ray_idx*num_sampled*MAX_LEAVES
                                   + sam_idx*MAX_LEAVES;
                            
        const float grad_out_curr = grad_out[grad_out_idx];
        
        float sampled_z = sampled_depths[basic_sampled_idx];
        float3 ray_ori_pz = make_float3(ray_ori[basic_ray_idx+0], ray_ori[basic_ray_idx+1], ray_ori[basic_ray_idx+2]);
        float3 ray_dir_pz = make_float3(ray_dir[basic_ray_idx+0], ray_dir[basic_ray_idx+1], ray_dir[basic_ray_idx+2]);
        float3 ray_sampled_pz = make_float3(ray_ori_pz.x + ray_dir_pz.x * sampled_z, 
                                            ray_ori_pz.y + ray_dir_pz.y * sampled_z,
                                            ray_ori_pz.z + ray_dir_pz.z * sampled_z);
        // get 8 corners, positions.
        OcTree curr_node = octree_nodes[voxel_idx];
        
        // to registers.
        const float3 *cx = &curr_node.xyz_corners[0];
        
        // calculate the gradients for 8 tensors.
        float dz_dy15_37 = (ray_sampled_pz.z - (cx[0].z + cx[4].z + cx[2].z + cx[6].z) / 4.0) / ((cx[1].z + cx[5].z + cx[3].z + cx[7].z) / 4.0 - (cx[0].z + cx[4].z + cx[2].z + cx[6].z) / 4.0); // t;
        float dz_dy04_26 = 1 - dz_dy15_37;
        float dy04_26_dx26 = (ray_sampled_pz.y - (cx[0].y + cx[4].y) / 2.0) / ((cx[2].y + cx[6].y) / 2.0 - (cx[0].y + cx[4].y) / 2.0); // t;
        float dy04_26_dx04 = 1 - dy04_26_dx26;
        float dy15_37_dx37 = (ray_sampled_pz.y - (cx[1].y + cx[5].y) / 2.0) / ((cx[3].y + cx[7].y) / 2.0 - (cx[1].y + cx[5].y) / 2.0);
        float dy15_37_dx15 = 1 - dy15_37_dx37;
        float dx04_dfts4 = (ray_sampled_pz.x - cx[0].x) / (cx[4].x - cx[0].x);
        float dx04_dfts0 = 1 - dx04_dfts4;
        float dx26_dfts6 = (ray_sampled_pz.x - cx[2].x) / (cx[6].x - cx[2].x);
        float dx26_dfts2 = 1 - dx26_dfts6;
        float dx15_dfts5 = (ray_sampled_pz.x - cx[1].x) / (cx[5].x - cx[1].x);
        float dx15_dfts1 = 1 - dx15_dfts5;
        float dx37_dfts7 = (ray_sampled_pz.x - cx[3].x) / (cx[7].x - cx[3].x);
        float dx37_dfts3 = 1 - dx37_dfts7;
        
        // gradients accumulations, gradients channels.
        float dz_dfts0 = dx04_dfts0 * dy04_26_dx04 * dz_dy04_26;
        float dz_dfts1 = dx15_dfts1 * dy15_37_dx15 * dz_dy15_37;
        float dz_dfts2 = dx26_dfts2 * dy04_26_dx26 * dz_dy04_26;
        float dz_dfts3 = dx37_dfts3 * dy15_37_dx37 * dz_dy15_37;
        float dz_dfts4 = dx04_dfts4 * dy04_26_dx04 * dz_dy04_26;
        float dz_dfts5 = dx15_dfts5 * dy15_37_dx15 * dz_dy15_37;
        float dz_dfts6 = dx26_dfts6 * dy04_26_dx26 * dz_dy04_26;
        float dz_dfts7 = dx37_dfts7 * dy15_37_dx37 * dz_dy15_37;
        // update the output gradients.
        grad_data[grad_data_basic_idx+0] = grad_out_curr * dz_dfts0;
        grad_data[grad_data_basic_idx+1] = grad_out_curr * dz_dfts1;
        grad_data[grad_data_basic_idx+2] = grad_out_curr * dz_dfts2;
        grad_data[grad_data_basic_idx+3] = grad_out_curr * dz_dfts3;
        grad_data[grad_data_basic_idx+4] = grad_out_curr * dz_dfts4;
        grad_data[grad_data_basic_idx+5] = grad_out_curr * dz_dfts5;
        grad_data[grad_data_basic_idx+6] = grad_out_curr * dz_dfts6;
        grad_data[grad_data_basic_idx+7] = grad_out_curr * dz_dfts7;
        
    }
}

template<typename scalar_t>
__global__ void rays_calculating_parallel_kernel(
    const scalar_t * __restrict__ cam_intr_inv, // [B, N_v, 3, 3]
    const scalar_t * __restrict__ cam_R_inv, // [B, N_v, 3, 3]
    const scalar_t * __restrict__ cam_T, // [B, N_v, 3, 1]
    scalar_t * output_rays_ori_dir, // [B, N_v, H, W, 6(ori,dir)]
    const uint32_t batch_size,
    const uint32_t num_views,
    const uint32_t im_h, // half of the resolution, 256
    const uint32_t im_w, // e.g., 512
    const int device,
    const int num_gpus // e.g., 2
) {
    CUDA_KERNEL_LOOP(i, batch_size*num_views*im_h*im_w) {
        const uint32_t batch_idx = (uint32_t) i / (num_views*im_h*im_w);
        const uint32_t view_idx  = (uint32_t) (i - batch_idx*num_views*im_h*im_w) / (im_h*im_w);
        const uint32_t h_idx     = (uint32_t) (i - batch_idx*num_views*im_h*im_w - view_idx*im_h*im_w) / im_w;
        const uint32_t w_idx     = (uint32_t) (i - batch_idx*num_views*im_h*im_w - view_idx*im_h*im_w - h_idx*im_w);
        
        const uint32_t h_idx_e   = h_idx * num_gpus + device; // 0 -> 0*2+0, 0*2+1; 1 -> 1*2+0, 1*2+1;

        // the idx of K is the same as the matrix R.
        const uint32_t idx_K_inv = batch_idx*num_views*9 + view_idx*9;
        const uint32_t idx_T     = batch_idx*num_views*3 + view_idx*3;
        const uint32_t idx_output = batch_idx*num_views*im_h*im_w*6
                                  + view_idx*im_h*im_w*6
                                  + h_idx*im_w*6
                                  + w_idx*6;
        scalar_t cam_dir[3] = {0}, world_dir[3] = {0}, cam_center[3] = {0};
        
        // to register.
        const scalar_t * cam_R_inv_    = cam_R_inv + idx_K_inv;
        const scalar_t * cam_intr_inv_ = cam_intr_inv + idx_K_inv;
        const scalar_t * cam_T_        = cam_T + idx_T;

        #pragma unroll 3
        // get camera's center position : -R^T * T;
        for (u_short k = 0; k < 3; k++) {
            cam_center[k] = - cam_R_inv_[3*k+0] * cam_T_[0]
                            - cam_R_inv_[3*k+1] * cam_T_[1]
                            - cam_R_inv_[3*k+2] * cam_T_[2];
        } 
        
        #pragma unroll 3
        for (u_short k = 0; k < 3; k++) {
            cam_dir[k] = (scalar_t) w_idx  *cam_intr_inv_[3*k+0]
                       + (scalar_t) h_idx_e*cam_intr_inv_[3*k+1]
                       + (scalar_t) 1      *cam_intr_inv_[3*k+2];
        }

        #pragma unroll 3
        for (u_short k = 0; k < 3; k++) {
            world_dir[k] = cam_dir[0] * cam_R_inv_[3*k+0]
                         + cam_dir[1] * cam_R_inv_[3*k+1]
                         + cam_dir[2] * cam_R_inv_[3*k+2];
        }
        // normalization, in three dims;
        scalar_t norm_dir = (scalar_t) sqrtf((float) world_dir[0]*world_dir[0] + world_dir[1]*world_dir[1] + world_dir[2]*world_dir[2]);
        // the z axis must larger than 0;
        // scalar_t scale_z = world_dir[2] > 0 ? world_dir[2] : - world_dir[2];
        // Make the z axis's length to be 1.0 ;
        // #pragma unroll 3
        // for (u_short k = 0; k < 3; k++) {
        //     world_dir[k] /= scale_z;
        // }

        // normalization, normal dim;
        #pragma unroll 3
        for (u_short k = 0; k < 3; k++) {
            world_dir[k] /= (norm_dir + 1e-10);
        }

        #pragma unroll 3
        for (u_short k = 0; k < 3; k++) {
            output_rays_ori_dir[idx_output+k]   = cam_center[k];   
            output_rays_ori_dir[idx_output+3+k] = world_dir[k];
        }
    }
}


template<typename scalar_t>
__global__ void rays_calculating_kernel(
    const scalar_t * __restrict__ cam_intr_inv, // [B, N_v, 3, 3]
    const scalar_t * __restrict__ cam_R_inv, // [B, N_v, 3, 3]
    const scalar_t * __restrict__ cam_T, // [B, N_v, 3, 1]
    scalar_t * output_rays_ori_dir, // [B, N_v, H, W, 6(ori,dir)]
    const uint32_t batch_size,
    const uint32_t num_views,
    const uint32_t im_h,
    const uint32_t im_w
) {
    // get the rays' dir : R^{-1} K^{-1} [x,y,1] -> the original ray in world coordinates.
    // parallel in each pixels.
    CUDA_KERNEL_LOOP(i, batch_size*num_views*im_h*im_w) {
        const uint32_t batch_idx = (uint32_t) i / (num_views*im_h*im_w);
        const uint32_t view_idx  = (uint32_t) (i - batch_idx*num_views*im_h*im_w) / (im_h*im_w);
        const uint32_t h_idx     = (uint32_t) (i - batch_idx*num_views*im_h*im_w - view_idx*im_h*im_w) / im_w;
        const uint32_t w_idx     = (uint32_t) (i - batch_idx*num_views*im_h*im_w - view_idx*im_h*im_w - h_idx*im_w);

        // the idx of K is the same as the matrix R.
        const uint32_t idx_K_inv = batch_idx*num_views*9 + view_idx*9;
        const uint32_t idx_T     = batch_idx*num_views*3 + view_idx*3;
        const uint32_t idx_output = batch_idx*num_views*im_h*im_w*6
                                  + view_idx*im_h*im_w*6
                                  + h_idx*im_w*6
                                  + w_idx*6;
        scalar_t cam_dir[3] = {0}, world_dir[3] = {0}, cam_center[3] = {0};
        
        // to register.
        const scalar_t * cam_R_inv_    = cam_R_inv + idx_K_inv;
        const scalar_t * cam_intr_inv_ = cam_intr_inv + idx_K_inv;
        const scalar_t * cam_T_        = cam_T + idx_T;

        #pragma unroll 3
        // get camera's center position : -R^T * T;
        for (u_short k = 0; k < 3; k++) {
            cam_center[k] = - cam_R_inv_[3*k+0] * cam_T_[0]
                            - cam_R_inv_[3*k+1] * cam_T_[1]
                            - cam_R_inv_[3*k+2] * cam_T_[2];
        } 
        
        #pragma unroll 3
        for (u_short k = 0; k < 3; k++) {
            cam_dir[k] = (scalar_t) w_idx*cam_intr_inv_[3*k+0]
                       + (scalar_t) h_idx*cam_intr_inv_[3*k+1]
                       + (scalar_t) 1    *cam_intr_inv_[3*k+2];
        }

        #pragma unroll 3
        for (u_short k = 0; k < 3; k++) {
            world_dir[k] = cam_dir[0] * cam_R_inv_[3*k+0]
                         + cam_dir[1] * cam_R_inv_[3*k+1]
                         + cam_dir[2] * cam_R_inv_[3*k+2];
        }
        // normalization, in three dims;
        scalar_t norm_dir = (scalar_t) sqrtf((float) world_dir[0]*world_dir[0] + world_dir[1]*world_dir[1] + world_dir[2]*world_dir[2]);
        // the z axis must larger than 0;
        // scalar_t scale_z = world_dir[2] > 0 ? world_dir[2] : - world_dir[2];
        // Make the z axis's length to be 1.0 ;
        // #pragma unroll 3
        // for (u_short k = 0; k < 3; k++) {
        //     world_dir[k] /= scale_z;
        // }

        // normalization, normal dim;
        #pragma unroll 3
        for (u_short k = 0; k < 3; k++) {
            world_dir[k] /= (norm_dir + 1e-10);
        }

        #pragma unroll 3
        for (u_short k = 0; k < 3; k++) {
            output_rays_ori_dir[idx_output+k]   = cam_center[k];   
            output_rays_ori_dir[idx_output+3+k] = world_dir[k];
        }
    }
}


template<typename scalar_t>
__global__ void rays_selecting_kernel(
    const scalar_t * __restrict__ ray_ori_dir, // [B, N_v, H, W, 6]
    const int * __restrict__ sampled_xy, // [B, N_v, N_rays, 2]
    scalar_t * output_sampled_rays, // [B, N_v, N_rays, 6]
    const uint32_t batch_size,
    const uint32_t num_views,
    const uint32_t num_rays,
    const uint32_t im_h, // default as 512.
    const uint32_t im_w
) {
    // parallel in each ray.
    CUDA_KERNEL_LOOP(i, batch_size*num_views*num_rays) {
        const uint32_t batch_idx = (uint32_t) i / (num_views*num_rays);
        const uint32_t view_idx  = (uint32_t) (i - batch_idx*num_views*num_rays) / num_rays;
        const uint32_t ray_idx   = (uint32_t) (i - batch_idx*num_views*num_rays - view_idx*num_rays);

        // get the x, y idx.
        const uint32_t basic_xy_idx = batch_idx*num_views*num_rays*2 + view_idx*num_rays*2 + ray_idx*2;
        
        const int * sampled_xy_ = sampled_xy + basic_xy_idx;

        const uint32_t x_idx = (uint32_t) sampled_xy_[0], y_idx = (uint32_t) sampled_xy_[1];

        const uint32_t ray_ori_dir_basic_idx = batch_idx*num_views*im_h*im_w*6
                                             + view_idx*im_h*im_w*6
                                             + y_idx*im_w*6
                                             + x_idx*6;

        const scalar_t * ray_ori_dir_ = ray_ori_dir + ray_ori_dir_basic_idx;
        
        const uint32_t output_idx     = batch_idx*num_views*num_rays*6
                                      + view_idx*num_rays*6
                                      + ray_idx*6;
        #pragma unroll 6
        for (u_short k = 0; k < 6; k++) {
            output_sampled_rays[output_idx + k] = ray_ori_dir_[k];
        }                 
    }
}


template<typename scalar_t>
__global__ void rgbdv_selecting_kernel(
    const scalar_t * __restrict__ target_rgbs,   // [B, N_v, H, W, 3 or 12]
    const scalar_t * __restrict__ target_normals,   // [B, N_v, H, W, 3]
    const scalar_t * __restrict__ target_depths, // [B, N_v, H, W, 1]
    const int * __restrict__ sampled_xy, // [B, N_v, N_rays, 2]
    scalar_t * output_sampled_rgb, // [B, N_v, N_rays, 3 or 12]
    scalar_t * output_sampled_normals, // [B, N_v, N_rays, 3 or 12]
    scalar_t * output_sampled_depth, // [B, N_v, N_rays, 1]
    scalar_t * output_sampled_label, // [B, N_v, N_rays, 1]
    const uint32_t batch_size,
    const uint32_t num_views,
    const uint32_t num_rays,
    const uint32_t rgb_channel,
    const uint32_t im_h, // default as 512.
    const uint32_t im_w
) {
    // parallel in each ray.
    CUDA_KERNEL_LOOP(i, batch_size*num_views*num_rays) {
        const uint32_t batch_idx = (uint32_t) i / (num_views*num_rays);
        const uint32_t view_idx  = (uint32_t) (i - batch_idx*num_views*num_rays) / num_rays;
        const uint32_t ray_idx   = (uint32_t) (i - batch_idx*num_views*num_rays - view_idx*num_rays);

        // get the x, y idx.
        const uint32_t basic_xy_idx = batch_idx*num_views*num_rays*2 + view_idx*num_rays*2 + ray_idx*2;
        
        const int * sampled_xy_ = sampled_xy + basic_xy_idx;

        const uint32_t x_idx = (uint32_t) sampled_xy_[0], y_idx = (uint32_t) sampled_xy_[1];

        const uint32_t ray_rgb_basic_idx = batch_idx*num_views*im_h*im_w*rgb_channel
                                         + view_idx*im_h*im_w*rgb_channel
                                         + y_idx*im_w*rgb_channel
                                         + x_idx*rgb_channel;
        
        const scalar_t * target_rgbs_ = target_rgbs + ray_rgb_basic_idx;

        const uint32_t ray_normals_basic_idx = batch_idx*num_views*im_h*im_w*3
                                             + view_idx*im_h*im_w*3
                                             + y_idx*im_w*3
                                             + x_idx*3;

        const scalar_t * target_normals_ = target_normals + ray_normals_basic_idx;

        const uint32_t ray_depth_basic_idx = batch_idx*num_views*im_h*im_w*1
                                           + view_idx*im_h*im_w*1
                                           + y_idx*im_w*1
                                           + x_idx*1;

        const scalar_t * target_depths_ = target_depths + ray_depth_basic_idx;                                   
        
        const uint32_t output_idx_rgb = batch_idx*num_views*num_rays*rgb_channel
                                      + view_idx*num_rays*rgb_channel
                                      + ray_idx*rgb_channel;

        const uint32_t output_idx_nml = batch_idx*num_views*num_rays*3
                                      + view_idx*num_rays*3
                                      + ray_idx*3;
        
        const uint32_t output_idx_d   = batch_idx*num_views*num_rays*1
                                      + view_idx*num_rays*1
                                      + ray_idx*1;
        
        #pragma unroll
        for (uint32_t k = 0; k < rgb_channel; k++) {
            output_sampled_rgb[output_idx_rgb + k] = target_rgbs_[k];
        }
        output_sampled_depth[output_idx_d] = target_depths_[0];
        #pragma unroll
        for (uint32_t k = 0; k < 3; k++) {
            output_sampled_normals[output_idx_nml + k] = target_normals_[k];
        }
        
        if (target_depths_[0] > 0) output_sampled_label[output_idx_d] = 1;
        else output_sampled_label[output_idx_d] = 0; 
    }
}

torch::Tensor rays_calculating_parallel(
    torch::Tensor cam_intr, // [B, N_v, 3, 3]
    torch::Tensor cam_R, // [B, N_v, 3, 3]
    torch::Tensor cam_T, // [B, N_v, 3]
    const int load_size, // 512
    const int device,
    const int num_gpus
) {
    cudaSetDevice(0); // on default GPU device;
    CUDA_CHECK_ERRORS();
    
    const uint32_t batch_size = cam_intr.size(0);
    const uint32_t num_views  = cam_intr.size(1);
    const uint32_t load_size_h = load_size / num_gpus; // e.g., 512 / 2
    // new tensors.
    torch::Tensor cam_R_inv_torch    = torch::full({batch_size, num_views, 3, 3}, 0, cam_R.options());
    torch::Tensor cam_intr_inv_torch = torch::full({batch_size, num_views, 3, 3}, 0, cam_intr.options());
    torch::Tensor rays_ori_dir       = torch::full({batch_size, num_views, load_size_h, load_size, 6}, 0, cam_intr.options());
    // rays are send into half data;
    
    // inverse_mat3x3
    static constexpr uint32_t block_size = 128;
    uint32_t grid_size = (batch_size * num_views + block_size - 1) / block_size;

    // get inverse matrix.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        cam_intr.scalar_type(), "inverse_mat3x3_kernel", ([&] {
            inverse_mat3x3<scalar_t> <<< grid_size, block_size >>> (
                cam_intr.contiguous().data_ptr<scalar_t>(),
                batch_size, num_views,
                cam_intr_inv_torch.contiguous().data_ptr<scalar_t>()
            );
        })
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        cam_R.scalar_type(), "inverse_mat3x3_kernel", ([&] {
            inverse_mat3x3<scalar_t> <<< grid_size, block_size >>> (
                cam_R.contiguous().data_ptr<scalar_t>(),
                batch_size, num_views,
                cam_R_inv_torch.contiguous().data_ptr<scalar_t>()
            );
        })
    );

    grid_size = (batch_size*num_views*load_size_h*load_size + block_size - 1) / block_size;
    // calculate all rays for original & dirs.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        cam_intr.scalar_type(), "rays_calculating_parallel_kernel", ([&] {
            rays_calculating_parallel_kernel<scalar_t> <<< grid_size, block_size >>> (
                cam_intr_inv_torch.contiguous().data_ptr<scalar_t>(),
                cam_R_inv_torch.contiguous().data_ptr<scalar_t>(),
                cam_T.contiguous().data_ptr<scalar_t>(),
                rays_ori_dir.contiguous().data_ptr<scalar_t>(),
                batch_size, num_views, load_size_h, load_size, device, num_gpus
            );
        })
    );

    // select N_rays (N_pixels: position) in the dilated masks.
    return rays_ori_dir;
}

// sampling rays : [B, N_views, N_rays(e.g., 1024), 6(ori, dir)]
// sampled idx : [B, N_views, N_rays, 2(x,y)]
torch::Tensor rays_calculating(
    torch::Tensor cam_intr, // [B, N_v, 3, 3]
    torch::Tensor cam_R, // [B, N_v, 3, 3]
    torch::Tensor cam_T, // [B, N_v, 3]
    const int load_size, // 512
    const int device
) {
    cudaSetDevice(device); // on GPU device.
    CUDA_CHECK_ERRORS();
    
    const uint32_t batch_size = cam_intr.size(0);
    const uint32_t num_views  = cam_intr.size(1);

    // new tensors.
    torch::Tensor cam_R_inv_torch    = torch::full({batch_size, num_views, 3, 3}, 0, cam_R.options());
    torch::Tensor cam_intr_inv_torch = torch::full({batch_size, num_views, 3, 3}, 0, cam_intr.options());
    torch::Tensor rays_ori_dir       = torch::full({batch_size, num_views, load_size, load_size, 6}, 0, cam_intr.options());

    // inverse_mat3x3
    static constexpr uint32_t block_size = 128;
    uint32_t grid_size = (batch_size * num_views + block_size - 1) / block_size;

    // get inverse matrix.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        cam_intr.scalar_type(), "inverse_mat3x3_kernel", ([&] {
            inverse_mat3x3<scalar_t> <<< grid_size, block_size >>> (
                cam_intr.contiguous().data_ptr<scalar_t>(),
                batch_size, num_views,
                cam_intr_inv_torch.contiguous().data_ptr<scalar_t>()
            );
        })
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        cam_R.scalar_type(), "inverse_mat3x3_kernel", ([&] {
            inverse_mat3x3<scalar_t> <<< grid_size, block_size >>> (
                cam_R.contiguous().data_ptr<scalar_t>(),
                batch_size, num_views,
                cam_R_inv_torch.contiguous().data_ptr<scalar_t>()
            );
        })
    );

    grid_size = (batch_size*num_views*load_size*load_size + block_size - 1) / block_size;
    // calculate all rays for original & dirs.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        cam_intr.scalar_type(), "rays_calculating_kernel", ([&] {
            rays_calculating_kernel<scalar_t> <<< grid_size, block_size >>> (
                cam_intr_inv_torch.contiguous().data_ptr<scalar_t>(),
                cam_R_inv_torch.contiguous().data_ptr<scalar_t>(),
                cam_T.contiguous().data_ptr<scalar_t>(),
                rays_ori_dir.contiguous().data_ptr<scalar_t>(),
                batch_size, num_views, load_size, load_size
            );
        })
    );

    // select N_rays (N_pixels: position) in the dilated masks.
    return rays_ori_dir;
}


torch::Tensor rays_selecting( // sample N_rays per view from the ray_ori_dir.
    torch::Tensor ray_ori_dir,  // [B, N_v, H, W, 6]
    torch::Tensor sampled_xy, // [B, N_v, N_rays, 2]
    const int device
) {
    cudaSetDevice(device); // on GPU device.
    CUDA_CHECK_ERRORS();
    
    const uint32_t batch_size = sampled_xy.size(0);
    const uint32_t num_views  = sampled_xy.size(1);
    const uint32_t num_rays   = sampled_xy.size(2);
    const uint32_t im_h       = ray_ori_dir.size(2);
    const uint32_t im_w       = ray_ori_dir.size(3);

    // new tensors.
    torch::Tensor rays_selected = torch::full({batch_size, num_views, num_rays, 6}, 0, ray_ori_dir.options());
    
    static constexpr uint32_t block_size = 128;
    uint32_t grid_size = (batch_size*num_views*num_rays + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        ray_ori_dir.scalar_type(), "rays_selecting_kernel", ([&] {
            rays_selecting_kernel<scalar_t> <<< grid_size, block_size >>> (
                ray_ori_dir.contiguous().data_ptr<scalar_t>(),
                sampled_xy.contiguous().data_ptr<int>(),
                rays_selected.contiguous().data_ptr<scalar_t>(),
                batch_size, num_views, num_rays, im_h, im_w
            );
        })
    );

    return rays_selected;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rgbdv_selecting( // sample RGBDs per view from the target rgbds, when depth is 0 (invalid.)
    torch::Tensor target_rgbs,    // [B, N_v, H, W, 3 or 12]
    torch::Tensor target_normals, // [B, N_v, H, W, 3]
    torch::Tensor target_depths,  // [B, N_v, H, W, 1]
    torch::Tensor sampled_xy,     // [B, N_v, N_rays, 2]
    const int device
) {
    cudaSetDevice(device); // on GPU device.
    CUDA_CHECK_ERRORS();
    
    const uint32_t batch_size = sampled_xy.size(0);
    const uint32_t num_views  = sampled_xy.size(1);
    const uint32_t num_rays   = sampled_xy.size(2);
    const uint32_t im_h       = target_rgbs.size(2);
    const uint32_t im_w       = target_rgbs.size(3);
    const uint32_t rgbs_chael = target_rgbs.size(4); // rgb's channel is 3 or 12 here.

    // new tensors.
    torch::Tensor rgb_selected     = torch::full({batch_size, num_views, num_rays, rgbs_chael}, 0, target_rgbs.options());
    torch::Tensor normals_selected = torch::full({batch_size, num_views, num_rays, 3}, 0, target_normals.options());
    torch::Tensor depth_selected   = torch::full({batch_size, num_views, num_rays, 1}, 0, target_depths.options());
    // the label to label whether the ray's depth is valid.
    torch::Tensor rays_d_valid_label = torch::full({batch_size, num_views, num_rays, 1}, 1, target_rgbs.options());
    
    static constexpr uint32_t block_size = 128;
    uint32_t grid_size = (batch_size*num_views*num_rays + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        target_rgbs.scalar_type(), "rgbdv_selecting_kernel", ([&] {
            rgbdv_selecting_kernel<scalar_t> <<< grid_size, block_size >>> (
                target_rgbs.contiguous().data_ptr<scalar_t>(),
                target_normals.contiguous().data_ptr<scalar_t>(),
                target_depths.contiguous().data_ptr<scalar_t>(),
                sampled_xy.contiguous().data_ptr<int>(),
                rgb_selected.contiguous().data_ptr<scalar_t>(),
                normals_selected.contiguous().data_ptr<scalar_t>(),
                depth_selected.contiguous().data_ptr<scalar_t>(),
                rays_d_valid_label.contiguous().data_ptr<scalar_t>(),
                batch_size, num_views, num_rays, rgbs_chael, im_h, im_w
            );
        })
    );

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>{rgb_selected, normals_selected, depth_selected, rays_d_valid_label};
}

// __global__ void voxel_feature_sampling_2d_forward_kernel(
//     OcTree *octree_nodes, // all nodes.
//     // input ray info.
//     const float * ray_ori, // [B, N_rays, 3]
//     const float * ray_dir, // [B, N_rays, 3]
//     const float * Ks, // [B, N_views, 3, 3]
//     const float * RTs, // [B, N_views, 3, 4]
//     float * input_feature, // [B, N_views, C, H, W]
//     // output sampled (idx, z);
//     const int * sampled_idx, // [B, N_ray, N_sampled]
//     const float * sampled_depths, // [B, N_ray, N_sampled]
//     float * sampled_features, // [B, N_views, N_ray, N_sampled, C]
//     // the parameters.
//     int im_w, int im_h,
//     int batch_size,
//     int num_rays,
//     int num_sampled, // default as 128;
//     int num_views // default as 3.
// ) {
//     #pragma unroll
//     CUDA_KERNEL_LOOP(k, batch_size * num_views * num_rays * num_sampled) {
//         // get idx.
//         int batch_idx   = (int) (k / (num_views * num_rays * num_sampled));
//         int view_idx    = (int) ((k - batch_idx * num_views * num_rays * num_sampled) / (num_rays * num_sampled));
//         int ray_idx     = (int) ((k - batch_idx*num_views*num_rays*num_sampled - view_idx*num_rays*num_sampled) / num_sampled);
//         int sam_idx     = (int) (k - batch_idx*num_views*num_rays*num_sampled - view_idx*num_rays*num_sampled - ray_idx*num_sampled);

//         int basic_sampled_idx = batch_idx*num_rays*num_sampled + ray_idx*num_sampled + sam_idx;
//         int basic_ray_idx = batch_idx*num_rays*3 + ray_idx*3;
//         // get sampled idx & z.
//         int voxel_idx = sampled_idx[basic_sampled_idx];
//         if (voxel_idx == -1) continue; // not valid voxel's idx.

//         size_t idx_RT = batch_idx*num_views*12 + view_idx*12;
//         size_t idx_K  = batch_idx*num_views*9  + view_idx*9;
//         float sampled_z = sampled_depths[basic_sampled_idx];
//         float3 ray_ori_pz = make_float3(ray_ori[basic_ray_idx+0], ray_ori[basic_ray_idx+1], ray_ori[basic_ray_idx+2]);
//         float3 ray_dir_pz = make_float3(ray_dir[basic_ray_idx+0], ray_dir[basic_ray_idx+1], ray_dir[basic_ray_idx+2]);
//         float3 ray_sampled_pz = make_float3(ray_ori_pz.x + ray_dir_pz.x * sampled_z, 
//                                             ray_ori_pz.y + ray_dir_pz.y * sampled_z,
//                                             ray_ori_pz.z + ray_dir_pz.z * sampled_z);
//         // 1. get the pixel location
//         float cam_pt_x = RTs[0+idx_RT]*ray_sampled_pz.x + RTs[1+idx_RT]*ray_sampled_pz.y + RTs[2+idx_RT] *ray_sampled_pz.z + RTs[3+idx_RT];
//         float cam_pt_y = RTs[4+idx_RT]*ray_sampled_pz.x + RTs[5+idx_RT]*ray_sampled_pz.y + RTs[6+idx_RT] *ray_sampled_pz.z + RTs[7+idx_RT];
//         float cam_pt_z = RTs[8+idx_RT]*ray_sampled_pz.x + RTs[9+idx_RT]*ray_sampled_pz.y + RTs[10+idx_RT]*ray_sampled_pz.z + RTs[11+idx_RT];

//         float pixel_x, pixel_y, pixel_z;
//         pixel_x = Ks[0+idx_K]*cam_pt_x + Ks[1+idx_K]*cam_pt_y + Ks[2+idx_K]*cam_pt_z;
//         pixel_y = Ks[3+idx_K]*cam_pt_x + Ks[4+idx_K]*cam_pt_y + Ks[5+idx_K]*cam_pt_z;
//         pixel_z = Ks[6+idx_K]*cam_pt_x + Ks[7+idx_K]*cam_pt_y + Ks[8+idx_K]*cam_pt_z;
//         pixel_x = pixel_x / pixel_z;
//         pixel_y = pixel_y / pixel_z;
//         // down sample to int range.
//         int pixel_x_ = (int) ::floor(pixel_x), pixel_y_ = (int) ::floor(pixel_y);
//         // when sampled range exceed the range, continue. (default as padding mode='zero');
//         if (pixel_x_ < 0 || pixel_x_ >= im_w || pixel_y_ < 0 || pixel_y_ >= im_h || cam_pt_z < 0) continue;
//         // printf("x, y : %d, %d\n", pixel_x_, pixel_y_);
        
//         // if (align_corners) {
//         //     // unnormalize coord from [-1, 1] to [0, size - 1]
//         //     return ((coord + 1.f) / 2) * (size - 1);
//         // } else {
//         //     // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
//         //     return ((coord + 1.f) * size - 1) / 2;
//         // }
        
        

//         // 3. concat all features.
//     }
// }

// __global__ void voxel_feature_sampling_2d_backward_kernel(
//     OcTree *octree_nodes, // all
//     // input ray info.
//     const int * ray_ori,
//     const int * ray_dir,
//     const float * ks,
//     const float * RTs,
//     // sampled (idx, z);
//     const int * sampled_idx, // [B, N_ray, N_sampled]
//     const int * sampled_depths,
//     // cornder points.
//     float * sampled_feature_grads, // [B, N_views, N_ray, N_sampled, C]
//     // the parameters.
//     int batch_size,
//     int num_rays,
//     int num_sampled, // default as 128;
//     int num_views
// ) {
    
// }


__global__ void ray_voxels_sampling_kernel(
    // inputed infomation (hited indexs).
    OcTree * octree_nodes, // all node list.
    const int * hit_idxs, // the hitted voxels' index, [B, N, n_max_hits.]
    const float * min_depths, // the inserted min z, [B, N, n_max_hits.]
    const float * max_depths, // the inserted max z, [B, N, n_max_hits.]
    // const int * index_queue, // the controled index, [B, N, 1], num of sampled voxels.
    const float * uniform_rands, // [B, N, n_sampled] (\in [0,1), for sampling. )
    const float * rays_valid_len, // [B, N, 1]
    // output idx, depth.
    int * sampled_idx, // [B, N, n_sampled]
    int * sampled_queue_idx, // [B, N, 1] max_num=N.
    float * sampled_depths, // [B, N, n_sampled] max_num=N;
    // float * sampled_corner_pos, // [B, N, n_sampled, 8]
    // the parameters.
    int batch_size,
    int num_rays,
    int num_max_hits,
    int num_sampled, // default as 64;
    int num_max_intersected_voxels, // e.g., 18 
    const int start_level, // the start level for the travelsal.
    const int num_levels, // total num levels for the octree.
    float * voxel_size // the mini voxel size;
) {
    #pragma unroll
    CUDA_KERNEL_LOOP(k, batch_size * num_rays * num_max_intersected_voxels) {
        // get the ray_idx, batch_idx.
        int batch_idx = (int) (k / (num_rays * num_max_intersected_voxels));
        int ray_idx = (int)(k - batch_idx * num_rays * num_max_intersected_voxels) / num_max_intersected_voxels;
        int voxel_local_id = k - batch_idx*num_rays*num_max_intersected_voxels - ray_idx*num_max_intersected_voxels;
        // the output idx.
        int queue_idx = batch_idx*num_rays*1 + ray_idx*1; // get num of the intersected voxels of the curr ray.
        int output_basic_idx   = batch_idx*num_rays*num_max_hits + ray_idx*num_max_hits;
        int output_sampled_basic_idx = batch_idx*num_rays*num_sampled + ray_idx*num_sampled;

        int voxel_idx = hit_idxs[output_basic_idx+voxel_local_id];
        
        if (voxel_idx == -1) continue; // not valid voxel.
        
        // get the properties.
        float min_depth = min_depths[output_basic_idx+voxel_local_id];
        float max_depth = max_depths[output_basic_idx+voxel_local_id];
        // int num_intersected_voxels = index_queue[queue_idx];
        float valid_len = rays_valid_len[queue_idx];
        OcTree node = octree_nodes[voxel_idx]; // get the current node.
        float curr_voxel_size  = node.voxel_size;
        float start_voxel_size = voxel_size[batch_idx] * powf(2.0, num_levels - start_level - 1);
        float voxel_size_rate  = (float) start_voxel_size / curr_voxel_size;
        // printf("s vsize rate: %f, %f, %f \n", voxel_size_rate, start_voxel_size, curr_voxel_size); // the three voxel sizes.
        
        // int num_sampled_voxels = (int) floorf( (max_depth - min_depth) * (float) num_sampled / valid_len );
        int num_sampled_voxels = (int) ( (max_depth - min_depth) * voxel_size_rate * (float) num_sampled / valid_len );
        // at least given 1 points.
        num_sampled_voxels   = num_sampled_voxels >= 1 ? num_sampled_voxels : 1;
        float sampled_offset = (float) (max_depth - min_depth) / (num_sampled_voxels + 1);
        
        // sample N points in [min_depth, max_depth];
        for (int i=0; i < num_sampled_voxels; i++) {
            // generate random z, and sampled z in [min, max];
            int rid = atomicAdd(&(sampled_queue_idx[queue_idx]), 1);
            // In case exceeding;
            if (rid >= num_sampled) {
                atomicSub(&(sampled_queue_idx[queue_idx]), 1);
                break;
            };
            
            float rand_val  = uniform_rands[rid + output_sampled_basic_idx];
            float basic_sampled_z = min_depth + sampled_offset * (i + 1);
            float sampled_z = basic_sampled_z - sampled_offset / 2.0 + sampled_offset * rand_val; // [min, max];
            // printf("sampled val, z: %f, %f", rand_val, sampled_z);
            // update the sampled_depths & idx;
            
            sampled_depths[rid + output_sampled_basic_idx] = sampled_z;
            sampled_idx[rid + output_sampled_basic_idx]    = voxel_idx;
        }

    }
}

__global__ void ray_voxels_further_sampling_kernel(
    OcTree * octree_nodes, // all node list.
    const int * hit_idxs, // the hitted voxels' index, [B, N, n_max_hits.]
    const float * min_depths, // the inserted min z, [B, N, n_max_hits.]
    const float * max_depths, // the inserted max z, [B, N, n_max_hits.]
    // const int * index_queue, // the controled index, [B, N, 1], num of sampled voxels.
    const float * uniform_rands, // [B, N, n_sampled] (\in [0,1), for sampling. )
    // const float * rays_valid_len, // [B, N, 1]
    // output idx, depth.
    int * sampled_idx, // [B, N, n_sampled]
    int * sampled_queue_idx, // [B, N, 1] max_num=N.
    float * sampled_depths, // [B, N, n_sampled] max_num=N;
    // the parameters.
    int batch_size,
    int num_rays,
    int num_max_hits,
    int num_sampled // default as 64;
) {
    #pragma unroll
    CUDA_KERNEL_LOOP(k, batch_size * num_rays) {
        // get the ray_idx, batch_idx.
        int batch_idx = (int) (k / num_rays);
        int ray_idx = k - batch_idx*num_rays;
        int curr_sampled_num = sampled_queue_idx[batch_idx*num_rays+ray_idx];
        int basic_input_idx  = batch_idx*num_rays*num_max_hits + ray_idx*num_max_hits;
        int basic_queue_idx  = batch_idx*num_rays+ray_idx;
        int basic_output_idx = batch_idx*num_rays*num_sampled + ray_idx*num_sampled;

        float min_intersected_z_mid = 99999, min_voxel_size = 99999;
        int min_intersected_idx = -1;
        float min_sampled_z0 = -1.0, min_sampled_z1 = -1.0;
        
        // get the min voxel_size (and min_z);
        for (int i=0; i < num_max_hits; i++) {
            int voxel_idx = hit_idxs[basic_input_idx+i];
            if (voxel_idx == -1) break; // when meet an invalid voxel, the latter voxels are invalid.
            OcTree node = octree_nodes[voxel_idx]; // get the node.
            if (node.voxel_size <= min_voxel_size) { // if voxel size <= min voxel size;
                float z0 = min_depths[basic_input_idx+i];
                float z1 = max_depths[basic_input_idx+i];
                float mid_sampled_z = (z0 + z1) / 2.0;
                if (node.voxel_size == min_voxel_size) { // case1: equal size, select the min depth.
                    if (mid_sampled_z < min_intersected_z_mid) {
                        min_intersected_z_mid = mid_sampled_z;
                        min_sampled_z0 = z0; min_sampled_z1 = z1;
                        min_intersected_idx = voxel_idx;
                    }
                } else { // case 2: smaller size, directly set the min depth.
                    min_intersected_z_mid = mid_sampled_z;
                    min_sampled_z0 = z0; min_sampled_z1 = z1;
                    min_intersected_idx = voxel_idx;
                }
                min_voxel_size = node.voxel_size;
            }
        }
        // printf("sampled idx, min_z, max_z: %d, %f, %f \n", min_intersected_idx, min_sampled_z0, min_sampled_z1);
        if (min_intersected_idx < 0) continue; // when 
        
        int num_sampled_new = num_sampled - curr_sampled_num;
        if (num_sampled_new <= 0) continue;
        float sampled_offset = (float) (min_sampled_z1 - min_sampled_z0) / (num_sampled_new + 1);
        // printf("idx, min_z, num: %d, %f, %d \n", min_intersected_idx, min_sampled_z0, num_sampled_new);
        int j = 0;
        for (int i=curr_sampled_num; i<num_sampled; i++) {
            float basic_sampled_z = min_sampled_z0 + sampled_offset * (j + 1);
            float rand_val  = uniform_rands[basic_output_idx + i];
            float sampled_z = basic_sampled_z - sampled_offset / 2.0 + sampled_offset * rand_val; // [min, max];

            // printf("sampled_z: %f\n", sampled_z);
            sampled_depths[basic_output_idx + i] = sampled_z;
            sampled_idx[basic_output_idx + i]    = min_intersected_idx;
            // update sampled idx.
            sampled_queue_idx[basic_queue_idx] += 1;
            j++;
        }
        //
        
    }
}

__global__ void ray_voxels_sampling_uniform_kernel(
    // inputed infomation (hited indexs).
    OcTree * octree_nodes, // all node list.
    const int * hit_idxs, // the hitted voxels' index, [B, N, n_max_hits.]
    const float * min_depths, // the inserted min z, [B, N, n_max_hits.]
    const float * max_depths, // the inserted max z, [B, N, n_max_hits.]
    // const int * index_queue, // the controled index, [B, N, 1], num of sampled voxels.
    const float * uniform_rands, // [B, N, n_sampled] (\in [0,1), for sampling. )
    const float * rays_valid_len, // [B, N, 1]
    // output idx, depth.
    int * sampled_idx, // [B, N, n_sampled]
    // int * sampled_queue_idx, // [B, N, 1] max_num=N.
    float * sampled_depths, // [B, N, n_sampled] max_num=N;
    // float * sampled_corner_pos, // [B, N, n_sampled, 8]
    // the parameters.
    int batch_size,
    int num_rays,
    int num_max_hits, // default as volume_dim * 3;
    int num_sampled, // default as 64;
    const int num_max_intersected_voxels, // e.g., 18 
    const int start_level, // the start level for the travelsal.
    const int num_levels, // total num levels for the octree.
    float * voxel_size // the mini voxel size;
) {
    // for each ray, the num of intersected voxels is not equal.
    #pragma unroll
    CUDA_KERNEL_LOOP(k, batch_size * num_rays) {
        // get the ray_idx, batch_idx, get the idxs.
        int batch_idx = (int) (k / num_rays);
        int ray_idx = k - batch_idx*num_rays;
        int basic_input_idx  = batch_idx*num_rays*num_max_hits + ray_idx*num_max_hits;
        int basic_output_idx = batch_idx*num_rays*num_sampled + ray_idx*num_sampled;
        int queue_idx        = batch_idx*num_rays*1 + ray_idx*1; // get num of the intersected voxels of the curr ray.
        // the basic info.
        int num_intersected_voxels = 0, _num_sampled = 0, _has_sampled = 0, left_sampled = 0;
        // bool flag = true;

        // for (int i=0; i < num_max_intersected_voxels; i++) {
        //     // valid voxel, the intersected_idx > 0;
        //     if ( hit_idxs[basic_input_idx + i] != -1) num_intersected_voxels ++;
        // }

        // this divide must left some points, where no points are sampled.
        if ( hit_idxs[basic_input_idx] < 0 ) continue;
        // at least one voxel is intersected.
        // num_sampled_per_voxel = (int) num_sampled / num_intersected_voxels;
        float valid_len = rays_valid_len[queue_idx];
        float start_voxel_size = voxel_size[batch_idx] * powf(4.0, num_levels - start_level - 1);
        // printf("s vsize rate: %f, %f, %f \n", voxel_size_rate, start_voxel_size, curr_voxel_size); // the three voxel sizes.
        
        // int num_sampled_voxels = (int) floorf( (max_depth - min_depth) * (float) num_sampled / valid_len );
        // at least one point is intersected.
        // num_sampled_per_voxel = num_sampled_per_voxel >= 1 ? num_sampled_per_voxel : 1;
        // sampling points.
        // #pragma unroll
        for (int i=0; i < num_max_intersected_voxels; i++) {
            // get the properties.
            int voxel_idx   = hit_idxs[basic_input_idx + i];
            float min_depth = min_depths[basic_input_idx + i];
            float max_depth = max_depths[basic_input_idx + i];
            if (voxel_idx < 0 || min_depth < 0 || max_depth < 0) break;

            OcTree node = octree_nodes[voxel_idx]; // get the current node.
            float curr_voxel_size  = node.voxel_size;
            float voxel_size_rate  = (float) start_voxel_size / curr_voxel_size;
            // if (voxel_size_rate == 4.0) voxel_size_rate = 2.0; // set the weight to 2.0 for large voxels.
            int _num_sampled = (int) ( (max_depth - min_depth) * voxel_size_rate * (float) num_sampled / valid_len );
            _num_sampled = _num_sampled >= 1 ? _num_sampled : 1; // at least sample 1 point.

            left_sampled = num_sampled - _has_sampled;
            if (left_sampled < _num_sampled) _num_sampled = left_sampled;

            float sampled_offset = (float) (max_depth - min_depth) / (_num_sampled + 1);

            for (int j=0; j < _num_sampled; j++) {
                if (_has_sampled >= num_sampled) break;
                float rand_val = uniform_rands[basic_output_idx + _has_sampled];
                float basic_sampled_z = min_depth + sampled_offset * (j + 1);
                float sampled_z = basic_sampled_z - sampled_offset / 2.0 + sampled_offset * rand_val; // [min, max];
                
                sampled_depths[basic_output_idx + _has_sampled] = sampled_z;
                sampled_idx[basic_output_idx + _has_sampled]    = voxel_idx;
                ++_has_sampled;
            }
        }

        left_sampled = num_sampled - _has_sampled;
        if (left_sampled > num_max_intersected_voxels) left_sampled = num_max_intersected_voxels;
        if (left_sampled > 0) { // when left sampled for sampling.
            for (int i=0; i < left_sampled; i++) { // each voxel sampled 1 point. here
                if (_has_sampled >= num_sampled) break;
                int voxel_idx   = hit_idxs[basic_input_idx + i];
                float min_depth = min_depths[basic_input_idx + i];
                float max_depth = max_depths[basic_input_idx + i];
                if (voxel_idx < 0 || min_depth < 0 || max_depth < 0) break; // incase meet invalid voxel

                float sampled_offset = (max_depth - min_depth) / 2.0; // sample one point here.
                float rand_val = uniform_rands[basic_output_idx + _has_sampled];
                float basic_sampled_z = min_depth + sampled_offset;
                float sampled_z = basic_sampled_z - sampled_offset / 2.0 + sampled_offset * rand_val; // [min, max];
                sampled_depths[basic_output_idx + _has_sampled] = sampled_z;
                sampled_idx[basic_output_idx + _has_sampled]    = voxel_idx;
                ++_has_sampled;
            }
        }

    }
}


std::tuple<torch::Tensor, torch::Tensor> MultiLayerOctree::ray_voxels_points_sampling(
    torch::Tensor hits_idxs, // [B, N, num_max_hits.] (N=1024)
    torch::Tensor min_depths, // [B, N, num_max_hits]
    torch::Tensor max_depths, // [B, N, num_max_hits]
    // torch::Tensor index_q_control, // [B, N, 1]
    // torch::Tensor sampled_index_q_control, // [B, N, 1]
    torch::Tensor rays_valid_len, // [B, N, 1]
    torch::Tensor uniform_rand_values, // [B, N, num_sampled].
    // torch::Tensor sampled_idx, torch::Tensor sampled_depths, // [B, N, num_sampled].
    const int start_level, const int num_sampled
) {
    cudaSetDevice(this->device); // on GPU device.
    CUDA_CHECK_ERRORS();

    int num_rays    = hits_idxs.size(1); // default as 1024.
    int num_batches = hits_idxs.size(0); // batch size.
    int num_max_hits = hits_idxs.size(2);
    // assert(num_batches == batch_size);
    // int num_sampled = sampled_idx.size(2); // default as 64.
    // printf("num sampled : %d \n", num_sampled);

    // init the tensors : the sampled index control queue, idxs & depths.
    torch::Tensor sampled_index_q_control = torch::full({batch_size, num_rays, 1}, 0, hits_idxs.options()).to(torch::kInt);
    torch::Tensor sampled_idx = torch::full({batch_size, num_rays, num_sampled}, -1, hits_idxs.options());
    torch::Tensor sampled_depths = torch::full({batch_size, num_rays, num_sampled}, -1, min_depths.options());
    // torch::Tensor sort_idx = torch::full({batch_size, num_rays, num_sampled}, -1, hits_idxs.options());
    
    // properties.
    int *hits_idxs_ptr = hits_idxs.contiguous().data_ptr<int>();
    float *min_depths_ptr  = min_depths.contiguous().data_ptr<float>();
    float *max_depths_ptr  = max_depths.contiguous().data_ptr<float>();
    // int * queue_idx_ptr = index_q_control.contiguous().data_ptr<int>();
    float * uniform_rands_ptr = uniform_rand_values.contiguous().data_ptr<float>();
    int * sampled_idx_ptr = sampled_idx.contiguous().data_ptr<int>();
    int * sampled_queue_idx_ptr = sampled_index_q_control.contiguous().data_ptr<int>();
    float * sampled_depths_ptr = sampled_depths.contiguous().data_ptr<float>();
    float * rays_valid_len_ptr = rays_valid_len.contiguous().data_ptr<float>();
    
    if (this->num_max_intersected_voxels_fine[0] == 0)
        return std::tuple<torch::Tensor, torch::Tensor>{sampled_idx, sampled_depths}; // when no voxels intersected, return;

    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, ray_voxels_sampling_kernel, 0, 0);
    grid_size = (int) (batch_size * num_rays * this->num_max_intersected_voxels_fine[0] + block_size - 1) / block_size;
    // sampling points in voxels, the voxels are sampled in all voxels.
    ray_voxels_sampling_kernel <<< grid_size, block_size >>> (
        RAW_PTR(this->octree_nodes),
        hits_idxs_ptr, min_depths_ptr, max_depths_ptr, uniform_rands_ptr, rays_valid_len_ptr,
        sampled_idx_ptr, sampled_queue_idx_ptr, sampled_depths_ptr, 
        batch_size, num_rays, num_max_hits, num_sampled, this->num_max_intersected_voxels_fine[0],
        start_level, this->num_levels, this->voxel_size
    );
    
    // refine the sampling points.
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, ray_voxels_further_sampling_kernel, 0, 0);
    grid_size = (int) (batch_size * num_rays + block_size - 1) / block_size;
    // fill in all the sampled points
    ray_voxels_further_sampling_kernel <<< grid_size, block_size >>> (
        RAW_PTR(this->octree_nodes),
        hits_idxs_ptr, min_depths_ptr, max_depths_ptr, uniform_rands_ptr,
        sampled_idx_ptr, sampled_queue_idx_ptr, sampled_depths_ptr, 
        batch_size, num_rays, num_max_hits, num_sampled
    );

    // sort the sampled z;
    // std::tuple<torch::Tensor, torch::Tensor> sort_ret = torch::sort(sampled_depths, 2, 0); // sort the sampled depth values in undescending.
    // sort_idx = std::get<1>(sort_ret); // get he sorted idxs.
    // return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>{sampled_idx, sampled_depths, sort_idx.to(torch::kInt)};
    return std::tuple<torch::Tensor, torch::Tensor>{sampled_idx, sampled_depths};
}

// uniform points sampling in coarse voxels.
std::tuple<torch::Tensor, torch::Tensor> MultiLayerOctree::ray_voxels_points_sampling_coarse(
    torch::Tensor hits_idxs, // [B, N, num_max_hits.] (eg. N=1024)
    torch::Tensor min_depths, // [B, N, num_max_hits]
    torch::Tensor max_depths, // [B, N, num_max_hits]
    // torch::Tensor index_q_control, // [B, N, 1]
    // torch::Tensor sampled_index_q_control, // [B, N, 1]
    torch::Tensor uniform_rand_values, // [B, N, num_sampled].
    torch::Tensor rays_valid_len, // [B, N, 1]
    // torch::Tensor sampled_idx, torch::Tensor sampled_depths, // [B, N, num_sampled].
    const int start_level, const int num_sampled
) {
    cudaSetDevice(this->device); // on GPU device.
    CUDA_CHECK_ERRORS();

    int num_rays    = hits_idxs.size(1); // default as 1024.
    int num_batches = hits_idxs.size(0); // i.e., batch size.
    int num_max_hits = hits_idxs.size(2); // coarse level, max hit voxels.

    // properties. 
    // torch::Tensor sampled_index_q_control = torch::full({batch_size, num_rays, 1}, 0, hits_idxs.options()).to(torch::kInt);
    torch::Tensor sampled_idx = torch::full({batch_size, num_rays, num_sampled}, -1, hits_idxs.options());
    torch::Tensor sampled_depths = torch::full({batch_size, num_rays, num_sampled}, -1, min_depths.options());

    // properties.
    int *hits_idxs_ptr = hits_idxs.contiguous().data_ptr<int>();
    float *min_depths_ptr  = min_depths.contiguous().data_ptr<float>();
    float *max_depths_ptr  = max_depths.contiguous().data_ptr<float>();
    // int * queue_idx_ptr = index_q_control.contiguous().data_ptr<int>();
    float * uniform_rands_ptr = uniform_rand_values.contiguous().data_ptr<float>();
    int * sampled_idx_ptr = sampled_idx.contiguous().data_ptr<int>();
    // int * sampled_queue_idx_ptr = sampled_index_q_control.contiguous().data_ptr<int>();
    float * sampled_depths_ptr = sampled_depths.contiguous().data_ptr<float>();
    float * rays_valid_len_ptr = rays_valid_len.contiguous().data_ptr<float>();
    
    if (this->num_max_intersected_voxels[0] == 0)
        return std::tuple<torch::Tensor, torch::Tensor>{sampled_idx, sampled_depths}; // when no voxels intersected, return;

    CUDA_CHECK_ERRORS();

    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, ray_voxels_sampling_uniform_kernel, 0, 0);
    grid_size = (int) (batch_size * num_rays + block_size - 1) / block_size;
    
    // sampling all points uniformly through all voxels, parallel in rays.
    ray_voxels_sampling_uniform_kernel <<< grid_size, block_size >>> (
        RAW_PTR(this->octree_nodes),
        hits_idxs_ptr, min_depths_ptr, max_depths_ptr, uniform_rands_ptr, rays_valid_len_ptr,
        sampled_idx_ptr, sampled_depths_ptr, 
        batch_size, num_rays, num_max_hits, num_sampled, this->num_max_intersected_voxels[0],
        start_level, this->num_levels, this->voxel_size
    );
    CUDA_CHECK_ERRORS();
    
    return std::tuple<torch::Tensor, torch::Tensor>{sampled_idx, sampled_depths};
}


// the function used for points sampling.
std::tuple<torch::Tensor, torch::Tensor> MultiLayerOctree::sort_samplings(
    torch::Tensor sampled_depths
) {
    cudaSetDevice(this->device); // on GPU device.
    CUDA_CHECK_ERRORS();

    int num_batches = sampled_depths.size(0); // batch size.
    int num_rays    = sampled_depths.size(1); // default as 1024.
    int num_sampled = sampled_depths.size(2); // num of sampled depths.

    torch::Tensor sort_idx = torch::full( {num_batches, num_rays, num_sampled}, -1, sampled_depths.options() );
    torch::Tensor sort_depths = torch::full( {num_batches, num_rays, num_sampled}, -1, sampled_depths.options() );
    // sort the sampled depth values in undescending, the sorting dim is 2;
    std::tuple<torch::Tensor, torch::Tensor> sort_ret = torch::sort(sampled_depths, 2, 0);
    // get the sorted results.
    sort_depths = std::get<0>(sort_ret);
    sort_idx = std::get<1>(sort_ret); // get he sorted idxs.

    CUDA_CHECK_ERRORS();

    return std::tuple<torch::Tensor, torch::Tensor>{sort_depths, sort_idx.to(torch::kInt)};
}


// calculate the projected xy-z coordinates in space.
std::tuple<torch::Tensor, torch::Tensor> MultiLayerOctree::project_sampled_xyz(
    torch::Tensor ray_ori, torch::Tensor ray_dir,
    torch::Tensor sampled_z, torch::Tensor sampled_idx,
    torch::Tensor Ks, torch::Tensor RTs, // [B, N_view, 3, 3] and [B, N_view, 3, 4]
    // torch::Tensor proj_xy, torch::Tensor proj_z, // output: [B, N_view, N_ray, N_sampled, 9, 2+1] (x,y, z_local)
    const int ori_w, const int ori_h, const int num_views, const bool record_corners
) {
    cudaSetDevice(this->device); // on GPU device.
    CUDA_CHECK_ERRORS();

    int batch_size    = sampled_z.size(0);
    int num_rays      = sampled_z.size(1);
    int num_sampled_z = sampled_z.size(2); // [B, N_ray, N_z];
    torch::Tensor proj_xy, proj_z;
    if (record_corners) {
        proj_xy = torch::full({batch_size, num_views, num_rays, num_sampled_z, 9, 2}, -1, ray_ori.options());
        proj_z  = torch::full({batch_size, num_views, num_rays, num_sampled_z, 9, 1}, -1, ray_ori.options());
    } else {
        proj_xy = torch::full({batch_size, num_views, num_rays, num_sampled_z, 2}, -1, ray_ori.options());
        proj_z  = torch::full({batch_size, num_views, num_rays, num_sampled_z, 1}, -1, ray_ori.options());
    }

    // printf("num_views, batch_size : %d, %d \n", num_views, batch_size);
    float * ray_ori_ptr = ray_ori.contiguous().data_ptr<float>(); // [B, N_ray, 3];
    float * ray_dir_ptr = ray_dir.contiguous().data_ptr<float>(); // [B, N_ray, 3];
    float * cam_int = Ks.contiguous().data_ptr<float>();
    float * cam_ext = RTs.contiguous().data_ptr<float>();
    int * sampled_idx_ptr = sampled_idx.contiguous().data_ptr<int>();
    float * sampled_z_ptr = sampled_z.contiguous().data_ptr<float>();
    float * proj_xy_ptr  = proj_xy.contiguous().data_ptr<float>();
    float * proj_z_ptr   = proj_z.contiguous().data_ptr<float>();

    // update grid size : B * N_rays * N_views * N_sampled.
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, project_xyz, 0, 0);
    grid_size = (int) (batch_size * num_rays * num_views * num_sampled_z + block_size - 1) / block_size;
    // get the 2D grids.
    project_xyz <<< grid_size, block_size >>> (
        RAW_PTR(this->octree_nodes), 
        ray_ori_ptr, ray_dir_ptr, 
        cam_int, cam_ext, 
        sampled_idx_ptr, sampled_z_ptr, 
        proj_xy_ptr, proj_z_ptr,
        ori_w, ori_h, 
        batch_size, num_rays, num_sampled_z, num_views, 
        record_corners
    );

    return std::tuple<torch::Tensor, torch::Tensor>{proj_xy, proj_z};
}


torch::Tensor MultiLayerOctree::trilinear_aggregate_features(
    torch::Tensor ray_ori, torch::Tensor ray_dir,
    torch::Tensor sampled_z, torch::Tensor sampled_idx,
    torch::Tensor input_feats
) {
    cudaSetDevice(this->device); // on GPU device.
    CUDA_CHECK_ERRORS();
    
    int batch_size    = sampled_z.size(0);
    int num_rays      = sampled_z.size(1);
    int num_sampled_z = sampled_z.size(2); // [B, N_ray, N_z];
    int dim_feat      = input_feats.size(1);

    torch::Tensor output_feats = torch::full({batch_size, dim_feat, num_rays, num_sampled_z}, 0, input_feats.options());

    // printf("num_views, batch_size : %d, %d \n", num_views, batch_size);
    float * ray_ori_ptr = ray_ori.contiguous().data_ptr<float>(); // [B, N_ray, 3];
    float * ray_dir_ptr = ray_dir.contiguous().data_ptr<float>(); // [B, N_ray, 3];
    int * sampled_idx_ptr = sampled_idx.contiguous().data_ptr<int>();
    float * sampled_z_ptr = sampled_z.contiguous().data_ptr<float>();
    float * input_feats_ptr  = input_feats.contiguous().data_ptr<float>(); // [B, C, N_ray, N_sampled, 8]
    float * output_feats_ptr = output_feats.contiguous().data_ptr<float>(); // [B, C, N_ray, N_sampled]

    // update grid size : B * N_rays * N_views * N_sampled * dim_feat;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, trilinear_aggregate_features_kernel, 0, 0);
    grid_size = (int) (batch_size * num_rays * num_sampled_z * dim_feat + block_size - 1) / block_size;

    trilinear_aggregate_features_kernel <<< grid_size, block_size >>> (
        RAW_PTR(this->octree_nodes), 
        ray_ori_ptr, ray_dir_ptr, 
        sampled_idx_ptr, sampled_z_ptr, 
        input_feats_ptr, output_feats_ptr,
        batch_size, num_rays, num_sampled_z, 
        dim_feat
    );
    return output_feats;
}


torch::Tensor MultiLayerOctree::trilinear_aggregate_features_backward(
    torch::Tensor ray_ori, torch::Tensor ray_dir,
    torch::Tensor sampled_z, torch::Tensor sampled_idx,
    torch::Tensor input_feats, torch::Tensor grad_output
) {
    cudaSetDevice(this->device); // on GPU device.
    CUDA_CHECK_ERRORS();
    
    int batch_size    = sampled_z.size(0);
    int num_rays      = sampled_z.size(1);
    int num_sampled_z = sampled_z.size(2); // [B, N_ray, N_z];
    int dim_feat = input_feats.size(1);

    // [B, C, N_ray, N_sampled, 8]
    torch::Tensor grad_data = torch::full({batch_size, dim_feat, num_rays, num_sampled_z, 8}, 0, input_feats.options());
    
    // printf("num_views, batch_size : %d, %d \n", num_views, batch_size);
    float * ray_ori_ptr = ray_ori.contiguous().data_ptr<float>(); // [B, N_ray, 3];
    float * ray_dir_ptr = ray_dir.contiguous().data_ptr<float>(); // [B, N_ray, 3];
    int * sampled_idx_ptr = sampled_idx.contiguous().data_ptr<int>();
    float * sampled_z_ptr = sampled_z.contiguous().data_ptr<float>();
    float * input_feats_ptr = input_feats.contiguous().data_ptr<float>(); // [B, C, N_ray, N_sampled, 8]
    float * grad_output_ptr = grad_output.contiguous().data_ptr<float>(); // [B, C, N_ray, N_sampled]
    float * grad_data_ptr   = grad_data.contiguous().data_ptr<float>(); // [B, C, N_ray, N_sampled]

    // update grid size : B * N_rays * N_views * N_sampled * dim_feat;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, trilinear_aggregate_features_backward_kernel, 0, 0);
    grid_size = (int) (batch_size * num_rays * num_sampled_z * dim_feat + block_size - 1) / block_size;

    trilinear_aggregate_features_backward_kernel <<< grid_size, block_size >>> (
        RAW_PTR(this->octree_nodes), 
        ray_ori_ptr, ray_dir_ptr, 
        sampled_idx_ptr, sampled_z_ptr, 
        grad_output_ptr, grad_data_ptr,
        batch_size, num_rays, num_sampled_z, 
        dim_feat
    );

    return grad_data;
}


// void MultiLayerOctree::grid_sample_features(
//     std::vector<torch::Tensor> input_feature_maps, // [(B*N_views, C, H, W)]
//     std::vector<torch::Tensor> output_pc_features,
//     torch::Tensor ray_ori, torch::Tensor ray_dir,
//     torch::Tensor sampled_z, torch::Tensor sampled_idx, 
//     torch::Tensor Ks, torch::Tensor RTs, // [B, N_view, 3, 3] and [B, N_view, 3, 4]
//     torch::Tensor proj_xy, torch::Tensor proj_z, // output: [B, N_view, N_ray, N_sampled, 3] (x,y, z_local)
//     const int ori_w, const int ori_h
// ) {
//     // grid_sample the features from the feature maps.
//     int num_features = input_feature_maps.size();
//     int batch_size    = sampled_z.size(0);
//     int num_rays      = sampled_z.size(1);
//     int num_sampled_z = sampled_z.size(2); // [B, N_ray, N_z];
//     int num_views = input_feature_maps[0].size(0) / batch_size;

//     // printf("num_views, batch_size : %d, %d \n", num_views, batch_size);
//     float * ray_ori_ptr = ray_ori.contiguous().data_ptr<float>(); // [B, N_ray, 3];
//     float * ray_dir_ptr = ray_dir.contiguous().data_ptr<float>(); // [B, N_ray, 3];
//     float * cam_int = Ks.contiguous().data_ptr<float>();
//     float * cam_ext = RTs.contiguous().data_ptr<float>();
//     int * sampled_idx_ptr = sampled_idx.contiguous().data_ptr<int>();
//     float * sampled_z_ptr = sampled_z.contiguous().data_ptr<float>();
//     float * proj_xy_ptr  = proj_xy.contiguous().data_ptr<float>();
//     float * proj_z_ptr   = proj_z.contiguous().data_ptr<float>();

//     // update grid size : B * N_rays * N_views * N_sampled.
//     grid_size = (int) (batch_size * num_rays * num_views * num_sampled_z + block_size - 1) / block_size;
//     cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, project_xyz, 0, 0);
//     // get the 2D grids.
//     project_xyz <<< grid_size, block_size >>> (
//         RAW_PTR(this->octree_nodes),
//         ray_ori_ptr, ray_dir_ptr, 
//         cam_int, cam_ext, 
//         sampled_idx_ptr, sampled_z_ptr, proj_xy_ptr, proj_z_ptr,
//         ori_w, ori_h, 
//         batch_size, num_rays, num_sampled_z, num_views
//     );

//     // // update grid size : B * N_rays * N_views * N_sampled.
//     // // grid_size = (int) (batch_size * num_rays * num_views * num_sampled_z + block_size - 1) / block_size;
//     // cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, voxel_feature_sampling_2d_forward_kernel, 0, 0);

//     // for (int i=0; i < num_features; i++) { // sampling features for each map.
//     //     torch::Tensor ft_map = input_feature_maps[i]; // [B, C, H, W]
//     //     torch::Tensor output_sampled_ft = output_pc_features[i];
//     //     float * ft_map_ptr = ft_map.contiguous().data_ptr<float>();
//     //     float * output_sampled_ft_ptr = output_sampled_ft.contiguous().data_ptr<float>();
//     //     int im_w = ft_map.size(3), im_h = ft_map.size(2);

//     //     voxel_feature_sampling_2d_forward_kernel <<< grid_size, block_size >>> (
//     //         RAW_PTR(this->octree_nodes), ft_map_ptr, 
//     //         sampled_idx_ptr, sampled_z_ptr, output_sampled_ft_ptr,
//     //         im_w, im_h,
//     //         batch_size, num_rays, num_sampled_z, num_views
//     //     );
//     // }
// }