#include <math.h>
#include <vector>
#include <stdint.h>
#include <assert.h>

// #include <omp.h>
#include "../include/cuda_helper.h"
#include "../include/octree.h"
#include <string>
#include <stdio.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#define N_STACK_MAX_NODES 8
#define DBL_MAX 9999999

__device__ static float atomicAddFloat(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(val + __int_as_float(assumed)) );
    } while (assumed != old);
    return __int_as_float(old);
}

template<typename vec_t>
__device__ bool judge_equal(const vec_t a, const vec_t b) {
    return (a.x == b.x && a.y == b.y && a.z == b.z);
}

// for a ray intersection with 
__device__ float2 RayVoxelIntersection(
    const float3 & ray_ori,
    const float3 & ray_dir,
    const float3 & center,
    const float voxel_size
) {
    float f_low = 0;
    float f_high = 100000.f;
    float f_dim_low, f_dim_high, temp, inv_ray_dir, start, aabb;
    float half_voxel_size = voxel_size / 2.0f;
    #pragma unroll
    for (int i=0; i < 3; i++) {
        if (i==0) { // for x;
            inv_ray_dir = __fdividef(1.0f, ray_dir.x); start = ray_ori.x; aabb = center.x;
        } else if (i==1) { // for y;
            inv_ray_dir = __fdividef(1.0f, ray_dir.y); start = ray_ori.y; aabb = center.y;
        } else { // for z;
            inv_ray_dir = __fdividef(1.0f, ray_dir.z); start = ray_ori.z; aabb = center.z;
        }
        // get the near & far t;
        f_dim_low  = (aabb - half_voxel_size - start) * inv_ray_dir;
        f_dim_high = (aabb + half_voxel_size - start) * inv_ray_dir;
        
        if (f_dim_low > f_dim_high) { // switch
            temp = f_dim_low; f_dim_low = f_dim_high; f_dim_high = temp;
        }

        if (f_dim_high < f_low) return make_float2(-1.0f,-1.0f);
        if (f_dim_low > f_high) return make_float2(-1.0f,-1.0f);

        f_low = (f_dim_low > f_low) ? f_dim_low : f_low;
        f_high = (f_dim_high < f_high) ? f_dim_high : f_high;
        if (f_low > f_high) return make_float2(-1.0f,-1.0f);
    }
    return make_float2(f_low,f_high);
}

__device__ int RayVoxelIntersectionNodeList(
    OcTree * octree_nodes, // all octree nodes.
    const int volume_basic_idx,
    int3 st_pos,
    int volume_dim,
    const int * volume_ptr, // e.g., [B,8,8,8] 
    int queue_basic_idx,
    int * hit_idxs,
    float * min_depth,
    float * max_depth,
    int * index_queue, // the controled index, [B, N, 1]
    // float * ray_total_valid_len, // [B,N,1]
    int output_basic_idx,
    int ray_basic_idx,
    const float3 ray_start, // [B, N, 3], (N=1024)
    const float3 ray_dir // [B, N, 3], (N=1024)
) {
    // for the first intersected voxel.
    int intersected_voxel_id = volume_basic_idx + st_pos.x*volume_dim*volume_dim + st_pos.y*volume_dim + st_pos.z;
    int node_idx = volume_ptr[intersected_voxel_id];
    // when intersect with the node, append to output list.
    if (node_idx > -1) { // not empy node.
        OcTree node = octree_nodes[node_idx]; // get the node.

        float2 voxel_depths = RayVoxelIntersection( // get the intersected info.
            ray_start, ray_dir,
            make_float3(node.xyz_center_.x, node.xyz_center_.y, node.xyz_center_.z), node.voxel_size
        );
        
        // assert(voxel_depths.x > -1 && voxel_depths.y > -1); // must intersected with the ray.
        // TODO.
        if (voxel_depths.x < 0 || voxel_depths.y < 0) return -1; // may not intersected with the voxel !!!
        
        // if (voxel_depths.x > -1) {
        //     int rid = atomicAdd(&(index_queue[queue_basic_idx]), 1);
        //     hit_idxs[output_basic_idx + rid]  = node_idx;
        //     min_depth[output_basic_idx + rid] = voxel_depths.x;
        //     max_depth[output_basic_idx + rid] = voxel_depths.y;
        // }
        if (node.valid) { // since the ray idx in parallel, no needed for atomic operation, and no min/max will repeated;
            // int rid = atomicAdd(&(index_queue[queue_basic_idx]), 1);
            int rid = index_queue[queue_basic_idx]; // no needed for atomicAdd (No Synth.).
            hit_idxs[output_basic_idx + rid]  = node_idx;
            min_depth[output_basic_idx + rid] = voxel_depths.x;
            max_depth[output_basic_idx + rid] = voxel_depths.y;
            index_queue[queue_basic_idx] += 1;
            // the valid len += (max_depth - min_depth.)
            // ray_total_valid_len[queue_basic_idx] += (voxel_depths.y - voxel_depths.x);
            
            return rid;
        }
    }
    return -1;
}

__device__ int RayVoxelIntersectionNodeListNew(
    OcTree * octree_nodes, // all octree nodes.
    const int volume_basic_idx,
    int3 st_pos,
    int volume_dim,
    const int * volume_ptr, // e.g., [B,8,8,8] 
    int queue_basic_idx,
    int * hit_idxs,
    float * min_depth,
    float * max_depth,
    int * index_queue, // the controled index, [B, N, 1]
    // float * ray_total_valid_len, // [B,N,1]
    int output_basic_idx,
    int ray_basic_idx,
    const float3 ray_start_, // [B, N, 3], (N=1024)
    const float3 ray_dir_, // [B, N, 3], (N=1024)
    float * ray_total_valid_len,
    bool is_training
) {
    // for the first intersected voxel.
    int intersected_voxel_id = volume_basic_idx + st_pos.x*volume_dim*volume_dim + st_pos.y*volume_dim + st_pos.z;
    int node_idx = volume_ptr[intersected_voxel_id];
    // when intersect with the node, append to output list.
    if (node_idx > -1) { // not empy node.
        OcTree node = octree_nodes[node_idx]; // get the node.
        float3 v_center = make_float3(node.xyz_center_.x, node.xyz_center_.y, node.xyz_center_.z);

        float2 voxel_depths = RayVoxelIntersection( // get the intersected info.
            ray_start_, ray_dir_,
            v_center, node.voxel_size
        );
        
        if (voxel_depths.x < 0 || voxel_depths.y < 0) return -1; // may not intersected with the voxel !!!
        
        if (node.valid) { // since the ray idx in parallel, no needed for atomic operation, and no min/max will repeated;
            /* new voxel traversal */
            // int rate = 4;
            // float v_size = node.voxel_size;
            // float child_v_size = v_size / rate;
            // bool has_child = false;
            // int rid = index_queue[queue_basic_idx]; // get previous id.

            // float3 start_pos = make_float3(ray_start_.x + voxel_depths.x * ray_dir_.x, ray_start_.y + voxel_depths.x * ray_dir_.y, ray_start_.z + voxel_depths.x * ray_dir_.z);
            // float3 st_offset = make_float3(start_pos.x-v_center.x+v_size/2.0f, start_pos.y-v_center.y+v_size/2.0f, start_pos.z-v_center.z+v_size/2.0f);
            
            // int st_x = (int)(st_offset.x / child_v_size), st_y = (int)(st_offset.y / child_v_size), st_z = (int)(st_offset.z / child_v_size);
            // st_x = ((st_x < 0) ? 0 : st_x); st_y = ((st_y < 0) ? 0 : st_y); st_z = ((st_z < 0) ? 0 : st_z);
            // int3 st_pos = make_int3((st_x < rate) ? st_x : rate - 1,
            //                         (st_y < rate) ? st_y : rate - 1,
            //                         (st_z < rate) ? st_z : rate - 1);
            // // the intersection kernel.
            // float step_x = (ray_dir_.x >= 0) ? 1 : -1, step_y = (ray_dir_.y >= 0) ? 1 : -1, step_z = (ray_dir_.z >= 0) ? 1 : -1;
            // // get the first intersected point in boundaries.
            // float nxt_v_bd_x = (step_x > 0) ? (st_pos.x + step_x) * child_v_size : st_pos.x * child_v_size,
            //       nxt_v_bd_y = (step_y > 0) ? (st_pos.y + step_y) * child_v_size : st_pos.y * child_v_size,
            //       nxt_v_bd_z = (step_z > 0) ? (st_pos.z + step_z) * child_v_size : st_pos.z * child_v_size;

            // float t_max_x = (ray_dir_.x != 0) ? (nxt_v_bd_x - st_offset.x)/ray_dir_.x : DBL_MAX, // for example, MAX here.
            //       t_max_y = (ray_dir_.y != 0) ? (nxt_v_bd_y - st_offset.y)/ray_dir_.y : DBL_MAX,
            //       t_max_z = (ray_dir_.z != 0) ? (nxt_v_bd_z - st_offset.z)/ray_dir_.z : DBL_MAX;
            
            // float t_delta_x = (ray_dir_.x != 0) ? child_v_size/ray_dir_.x * step_x : DBL_MAX,
            //       t_delta_y = (ray_dir_.y != 0) ? child_v_size/ray_dir_.y * step_y : DBL_MAX,
            //       t_delta_z = (ray_dir_.z != 0) ? child_v_size/ray_dir_.z * step_z : DBL_MAX;
            
            // float tx = t_max_x, ty = t_max_y, tz = t_max_z;
            // int j=0, J=rate*rate*rate; // max iteration=8, e.g.;
            
            // while ( j < J ) {
            //     int i = st_pos.x*rate*rate + st_pos.y*rate + st_pos.z;
            //     int child_idx = node.children_gindex[i];
            //     if (child_idx != -1) { // the child is ok.
            //         // get child.
            //         OcTree child_node = octree_nodes[child_idx]; 
            //         float2 child_depths = RayVoxelIntersection( // get the intersected info.
            //             ray_start_, ray_dir_,
            //             make_float3(child_node.xyz_center_.x, child_node.xyz_center_.y, child_node.xyz_center_.z), 
            //             child_node.voxel_size
            //         );
            //         if (child_depths.x >= 0 && child_depths.y >= 0 && child_node.valid) {
            //             hit_idxs[output_basic_idx + rid]  = child_idx;
            //             min_depth[output_basic_idx + rid] = child_depths.x;
            //             max_depth[output_basic_idx + rid] = child_depths.y;
            //             rid += 1;
            //             // weighted points sampling.
            //             // atomicAddFloat(&(ray_total_valid_len[queue_basic_idx]), (child_depths.y - child_depths.x) * 2.0);
            //             ray_total_valid_len[queue_basic_idx] += (child_depths.y - child_depths.x) * 4.0;
            //             has_child = true;
            //             printf("child min, max depth: %f, %f \n", child_depths.x, child_depths.y);
            //         }
            //     }
                
            //     if (tx < ty) {
            //         if (tx < tz) {st_pos.x += step_x; tx += t_delta_x;}
            //         else {st_pos.z += step_z; tz += t_delta_z;}
            //     }
            //     else {
            //         if (ty < tz) {st_pos.y += step_y; ty += t_delta_y;}
            //         else {st_pos.z += step_z; tz += t_delta_z;}
            //     }
            //     if (st_pos.x < 0 || st_pos.x >= rate || st_pos.y < 0 || st_pos.y >= rate || st_pos.z < 0 || st_pos.z >= rate) break;
            //     j++;
            // }
            /* ----------------------------- */

            int rid = index_queue[queue_basic_idx]; // get previous id.
            if (!is_training) { // when inference, adaptive sampling.
                // bool has_child = false;
                #pragma unroll
                for (int i=0;i<MAX_LEAVES;i++) { // judge wether the child node is intersected.
                    int child_idx = node.children_gindex[i];
                    if (child_idx != -1) { // valid child.
                        // get child.
                        OcTree child_node = octree_nodes[child_idx]; 
                        float2 child_depths = RayVoxelIntersection( // get the intersected info.
                            ray_start_, ray_dir_,
                            make_float3(child_node.xyz_center_.x, child_node.xyz_center_.y, child_node.xyz_center_.z), 
                            child_node.voxel_size
                        );
                        if (child_depths.x < 0 || child_depths.y < 0 || (!child_node.valid)) continue; // no intersected with the child.
        
                        hit_idxs[output_basic_idx + rid]  = child_idx;
                        min_depth[output_basic_idx + rid] = child_depths.x;
                        max_depth[output_basic_idx + rid] = child_depths.y;
                        ray_total_valid_len[queue_basic_idx] += (child_depths.y - child_depths.x) * 4.0; // weighted sampling (distance)
                        
                        rid += 1;
                        // printf("child min, max depth: %f, %f \n", child_depths.x, child_depths.y);
                        // has_child = true;
                    } 
                    // else break; // after -1, all children nodes' idxs are -1.
                }
                hit_idxs[output_basic_idx + rid]  = node_idx;
                min_depth[output_basic_idx + rid] = voxel_depths.x;
                max_depth[output_basic_idx + rid] = voxel_depths.y;
                
                // atomicAddFloat(&(ray_total_valid_len[queue_basic_idx]), voxel_depths.y - voxel_depths.x);
                ray_total_valid_len[queue_basic_idx] += voxel_depths.y - voxel_depths.x;
                rid += 1; // update the queue idx.
                // printf("min, max depth: %f, %f \n", voxel_depths.x, voxel_depths.y);
            } else {
                hit_idxs[output_basic_idx + rid]  = node_idx;
                min_depth[output_basic_idx + rid] = voxel_depths.x;
                max_depth[output_basic_idx + rid] = voxel_depths.y;
                
                // atomicAddFloat(&(ray_total_valid_len[queue_basic_idx]), voxel_depths.y - voxel_depths.x);
                ray_total_valid_len[queue_basic_idx] += voxel_depths.y - voxel_depths.x;
                rid += 1; // update the queue idx.
            }

            index_queue[queue_basic_idx] = rid;
            
            return rid - 1;
        }
    }
    return -1;
}

__global__ void ray_voxel_traversal( // reference to A Fast Voxel Traversal Algorithm
    const int * coarse_volume_ptr, // e.g., [B,8,8,8]
    OcTree * octree_nodes, // all octree nodes.
    int coarse_level_start_idx, // the start index of coarse nodes.
    const float * ray_start, // [B, N, 3], (N=1024)
    const float * ray_dir, // [B, N, 3], (N=1024)
    // the output tensors.
    int * hit_idxs, // the hitted voxels' index, [B, N, n_max_hits.]
    float * min_depth, // the inserted min z, [B, N, n_max_hits.]
    float * max_depth, // the inserted max z, [B, N, n_max_hits.]
    int * index_queue, // the controled index, [B, N, 1]
    float * ray_total_valid_len, // [B, N, 1]
    // the parameters.
    const float * volume_bbox, // [B, 3, 2]
    const float * voxel_size, // [B, 1]
    const int n_ray,
    const int n_max_hits,
    const int num_nodes_total, // N1+N2+N3+...+NB.
    const int volume_dim_bottom, // e.g., 256;
    const int volume_dim_top, // e.g., 8
    const int batch_size,
    int * num_max_intersected_voxels,
    const int th_early_stop,
    const int opt_num_intersected_voxels,
    bool only_voxel_traversal,
    bool is_training
) {
    #pragma unroll
    CUDA_KERNEL_LOOP(k, batch_size * n_ray) { // in parallel (all rays.)
        int batch_idx = (int) k / n_ray;
        int ray_idx = k - batch_idx * n_ray;
        int ray_basic_idx = batch_idx * n_ray * 3 + ray_idx * 3;
        int volume_basic_idx = batch_idx * volume_dim_top * volume_dim_top * volume_dim_top;
        int queue_basic_idx  = batch_idx * n_ray + ray_idx;
        int output_basic_idx = batch_idx * n_ray * n_max_hits + ray_idx * n_max_hits;
        // step 1. the ray intersect with the largest voxel.
        // get the center of the volume bounding box : (min + max) / 2.0f;
        // float bbox_size = volume_bbox[batch_idx*6+0*2+1] - volume_bbox[batch_idx*6+0*2+0]; // just get the bbox size .
        float x_center = (volume_bbox[batch_idx*6+0*2+0] + volume_bbox[batch_idx*6+0*2+1]) / 2.0f;
        float y_center = (volume_bbox[batch_idx*6+1*2+0] + volume_bbox[batch_idx*6+1*2+1]) / 2.0f;
        float z_center = (volume_bbox[batch_idx*6+2*2+0] + volume_bbox[batch_idx*6+2*2+1]) / 2.0f;

        float voxel_size_batch = voxel_size[batch_idx];
        float voxel_size_top   = voxel_size_batch * (volume_dim_bottom / volume_dim_top); // size * rate(e.g., 4)
        // printf("voxel_size, %f, %f\n", voxel_size_batch, voxel_size_top);
        // the start bbox, subed by half of voxel size : x_min - v_size / 2.0f;
        float3 bbox_st    = make_float3(volume_bbox[batch_idx*6+0*2+0] - voxel_size_batch / 2.0f,
                                        volume_bbox[batch_idx*6+1*2+0] - voxel_size_batch / 2.0f,
                                        volume_bbox[batch_idx*6+2*2+0] - voxel_size_batch / 2.0f);
        
        float3 ray_start_ = make_float3(ray_start[ray_basic_idx+0], ray_start[ray_basic_idx+1], ray_start[ray_basic_idx+2]);
        float3 ray_dir_   = make_float3(ray_dir[ray_basic_idx+0], ray_dir[ray_basic_idx+1], ray_dir[ray_basic_idx+2]);

        float3 pcd_center = make_float3(x_center, y_center, z_center);
        // intersect with the largest volume, the bbox's size is voxel_size * volume_dim;
        float2 depths = RayVoxelIntersection(
            ray_start_, ray_dir_,
            pcd_center, voxel_size_top * volume_dim_top
        );

        if (depths.x < 0.0f || depths.y < 0.0f) continue; // not valid voxels.
        /*----------------------------------------------------*/
        
        float3 start_pos = make_float3(ray_start_.x + depths.x * ray_dir_.x, 
                                       ray_start_.y + depths.x * ray_dir_.y, 
                                       ray_start_.z + depths.x * ray_dir_.z);

        // float3 end_pos = make_float3(start_pos.x, start_pos.y, start_pos.z);
        float3 st_offset = make_float3(start_pos.x - bbox_st.x, start_pos.y - bbox_st.y, start_pos.z - bbox_st.z);
        // get start & end idx of the voxels (note : we clip the idx to [0, dim_top - 1]).
        int st_x = (int)(st_offset.x / voxel_size_top), st_y = (int)(st_offset.y / voxel_size_top), st_z = (int)(st_offset.z / voxel_size_top);
        // st_x = clamp(st_x, 0, volume_dim_top - 1); st_y = clamp(st_y, 0, volume_dim_top - 1); st_z = clamp(st_z, 0, volume_dim_top - 1);
        // the start position clamp in [0, volume_dim_top - 1]
        st_x = ((st_x < 0) ? 0 : st_x); st_y = ((st_y < 0) ? 0 : st_y); st_z = ((st_z < 0) ? 0 : st_z);
        int3 st_pos = make_int3((st_x < volume_dim_top) ? st_x : volume_dim_top - 1,
                                (st_y < volume_dim_top) ? st_y : volume_dim_top - 1,
                                (st_z < volume_dim_top) ? st_z : volume_dim_top - 1);

        // int3 st_pos = make_int3(st_x, st_y, st_z);
        // the intersection kernel.
        float step_x = (ray_dir_.x >= 0) ? 1 : -1, step_y = (ray_dir_.y >= 0) ? 1 : -1, step_z = (ray_dir_.z >= 0) ? 1 : -1;
        // get the first intersected point in boundaries.
        float nxt_v_bd_x = (step_x > 0) ? (st_pos.x + step_x) * voxel_size_top : st_pos.x * voxel_size_top,
              nxt_v_bd_y = (step_y > 0) ? (st_pos.y + step_y) * voxel_size_top : st_pos.y * voxel_size_top,
              nxt_v_bd_z = (step_z > 0) ? (st_pos.z + step_z) * voxel_size_top : st_pos.z * voxel_size_top;

        float t_max_x = (ray_dir_.x != 0) ? (nxt_v_bd_x - st_offset.x)/ray_dir_.x : DBL_MAX, // for example, MAX here.
              t_max_y = (ray_dir_.y != 0) ? (nxt_v_bd_y - st_offset.y)/ray_dir_.y : DBL_MAX,
              t_max_z = (ray_dir_.z != 0) ? (nxt_v_bd_z - st_offset.z)/ray_dir_.z : DBL_MAX;
        
        float t_delta_x = (ray_dir_.x != 0) ? voxel_size_top/ray_dir_.x * step_x : DBL_MAX,
              t_delta_y = (ray_dir_.y != 0) ? voxel_size_top/ray_dir_.y * step_y : DBL_MAX,
              t_delta_z = (ray_dir_.z != 0) ? voxel_size_top/ray_dir_.z * step_z : DBL_MAX;
        
        float tx = t_max_x, ty = t_max_y, tz = t_max_z;
        int j=0, J=volume_dim_top+volume_dim_top+volume_dim_top+volume_dim_top; // max iteration=512, e.g.;
        // int j=0, J=volume_dim_top*volume_dim_top*volume_dim_top; // max intersected volume : 32 ** 3
        // printf("volume dims, %d, %d, %d\n", st_pos.x, st_pos.y, st_pos.z);
        // printf("bbox center, %f, %f, %f\n", x_center, y_center, z_center);

        bool pre_empty = false, has_valid_node = false; 
        int num_succ_empty_node = 0, num_intersected_voxels = 0; // for early stop.
        while ( j < J ) { // parallel in each ray.
            int rid;
            // if intersected with the current voxel, then append to node list.
            if (only_voxel_traversal) { // directly voxel traversal, inter
                rid = RayVoxelIntersectionNodeListNew(octree_nodes, volume_basic_idx, st_pos, 
                                                      volume_dim_top, coarse_volume_ptr,
                                                      queue_basic_idx, hit_idxs, min_depth, 
                                                      max_depth, index_queue, output_basic_idx, 
                                                      ray_basic_idx, ray_start_, ray_dir_, ray_total_valid_len,
                                                      is_training);
            } else {
                rid = RayVoxelIntersectionNodeList(octree_nodes, volume_basic_idx, st_pos, 
                                                    volume_dim_top, coarse_volume_ptr,
                                                    queue_basic_idx, hit_idxs, min_depth, 
                                                    max_depth, index_queue, output_basic_idx, 
                                                    ray_basic_idx, ray_start_, ray_dir_);
            }
            atomicMax(&(num_max_intersected_voxels[0]), rid+1);

            if (th_early_stop > -1) { // support early stop.
                // early stop.
                if (rid == -1) { // when the return rid = -1, is the empty-node.
                    if (pre_empty && has_valid_node) {num_succ_empty_node++;}
                    pre_empty = true;
                } else {
                    num_succ_empty_node = 0; // clear the num of empty nodes.
                    pre_empty = false; has_valid_node = true;
                }
                if ((num_succ_empty_node + 1) >= th_early_stop) break;
            }
            // TODO. set the intersected voxels' num, reduce the sampling num of the points;
            if (opt_num_intersected_voxels > -1) { // support trunc-intersected voxels;
                if ( rid+1 > 0 ) num_intersected_voxels++; // itersected with more than 1 voxel
                // when intersected with more than the max_interseced voxels;
                if ( num_intersected_voxels >= opt_num_intersected_voxels ) break;
            }
            
            if (tx < ty) {
                if (tx < tz) { st_pos.x += step_x; tx += t_delta_x;}
                else {st_pos.z += step_z; tz += t_delta_z;}
            }
            else {
                if (ty < tz) {st_pos.y += step_y; ty += t_delta_y;}
                else {st_pos.z += step_z; tz += t_delta_z;}
            }
            
            if (st_pos.x < 0 || st_pos.x >= volume_dim_top || st_pos.y < 0 || st_pos.y >= volume_dim_top || st_pos.z < 0 || st_pos.z >= volume_dim_top) break;
            j += 1;
        }
        
    }
}

__global__ void ray_octree_traversal(
    OcTree * octree_nodes,
    const float * ray_start, // [B, N, 3], (N=1024)
    const float * ray_dir, // [B, N, 3]
    // the output tensors.
    int * hit_idxs_coarse, // the hitted voxels' index, [B, N, n_max_hits.]
    float * min_depth_coarse, // the inserted min z, [B, N, n_max_hits.]
    float * max_depth_coarse, // the inserted max z, [B, N, n_max_hits.]
    int * hit_idxs, // the hitted voxels' index, [B, N, n_max_hits.]
    float * min_depth, // the inserted max z, [B, N, n_max_hits.]
    float * max_depth, // the inserted max z, [B, N, n_max_hits.]
    // int * index_queue_coarse, // the controled index, [B, N, 1]
    int * index_queue, // the controled index, [B, N, 1]
    float * ray_total_valid_len, // [B,N,1]
    // the parameters.
    const int n_ray, // in parallel (n_rays * n_nodes).
    const int n_max_hits_coarse, // the num of intersected voxels(maxs).
    const int n_max_hits, // the num of intersected voxels(maxs).
    int* num_max_intersected_voxels_fine, // N1+N2+N3..+NB
    int num_max_int_coarse_voxels,
    const int batch_size, // num of batch.
    const int rate // the children's volume size.
) {
    __shared__ OcTree *shared_octree; // put in shard memory.
    #pragma unroll
    CUDA_KERNEL_LOOP(k, batch_size * n_ray * num_max_int_coarse_voxels) { // in parallel [B, N_ray, MAX_INSERTED]
        shared_octree = octree_nodes;
        __syncthreads();
        // get node idx, ray idx.
        int batch_idx = (int) (k / (n_ray * num_max_int_coarse_voxels));
        int ray_idx = (int)(k - batch_idx * n_ray * num_max_int_coarse_voxels) / num_max_int_coarse_voxels;
        int voxel_local_id = k - batch_idx*n_ray*num_max_int_coarse_voxels - ray_idx*num_max_int_coarse_voxels;
        // get the ouput's idx.
        int coarse_basic_idx = batch_idx*n_ray*n_max_hits_coarse + ray_idx*n_max_hits_coarse;
        int output_basic_idx = batch_idx*n_ray*n_max_hits + ray_idx*n_max_hits;
        
        int voxel_idx = hit_idxs_coarse[coarse_basic_idx+voxel_local_id];
        if (voxel_idx == -1) continue; // not valid node.
        
        OcTree curr_node = shared_octree[voxel_idx];
        float start_voxel_size = curr_node.voxel_size;
        // assert(curr_node.batch_id == batch_idx && curr_node.valid); // must be valid node.
        int ray_basic_idx = batch_idx*n_ray*3 + ray_idx*3;
        int queue_basic_idx = batch_idx * n_ray + ray_idx;
        // get the ray information (start, end)
        float3 ray_start_ = make_float3(ray_start[ray_basic_idx+0], ray_start[ray_basic_idx+1], ray_start[ray_basic_idx+2]);
        float3 ray_dir_   = make_float3(ray_dir[ray_basic_idx+0], ray_dir[ray_basic_idx+1], ray_dir[ray_basic_idx+2]);
        
        // recurrent traversal.
        int idxs_stack[N_STACK_MAX_NODES] = {-1}; // DFS (in register.)
        int idx_ptr = 0, cur_node_idx = -1, start_idx = curr_node.gindex, valid_idx = 1;
        // float2 depths;
        idxs_stack[idx_ptr] = curr_node.gindex;
        // Intersected nodes buffer; To record the valid min, max depths and idxs;
        // note: for the same min,max depths, only record the min_resolution's idx.
        int valid_idxs[N_STACK_MAX_NODES] = {-1};
        float valid_min_depths[N_STACK_MAX_NODES] = {-1.0f};
        float valid_max_depths[N_STACK_MAX_NODES] = {-1.0f};
        float valid_scale[N_STACK_MAX_NODES] = {-1.0f};
        // init the first node.
        valid_idxs[0] = start_idx;
        valid_min_depths[0] = min_depth_coarse[coarse_basic_idx+voxel_local_id];
        valid_max_depths[0] = max_depth_coarse[coarse_basic_idx+voxel_local_id];
        valid_scale[0] = 1.0;
        
        while (idx_ptr >= 0) { // push in stack.
            cur_node_idx = idxs_stack[idx_ptr];
            idxs_stack[idx_ptr] = -1; idx_ptr--;

            // append the min,max depth to the output bbox.
            OcTree node = shared_octree[cur_node_idx]; // get the curr node.
            // float3 v_center   = make_float3(node.xyz_center_.x, node.xyz_center_.y, node.xyz_center_.z);
            // float v_size = node.voxel_size;
            // assert(depths.x > -1 && depths.y > -1);
            // if (cur_node_idx == start_idx) {
            //     depths = make_float2(min_depth[output_basic_idx+voxel_local_id], max_depth[output_basic_idx+voxel_local_id]);
            // } else {
            //     depths = RayVoxelIntersection( // get the intersected info.
            //         ray_start_, ray_dir_, v_center, v_size
            //     );
            //     int rid = atomicAdd(&(index_queue[queue_basic_idx]), 1);
            //     hit_idxs[output_basic_idx + rid]  = cur_node_idx;
            //     min_depth[output_basic_idx + rid] = depths.x;
            //     max_depth[output_basic_idx + rid] = depths.y;
            // }

            if (cur_node_idx != start_idx) { // when the current idx is not the start_idx, when the idx is valid.
                float3 v_center = make_float3(node.xyz_center_.x, node.xyz_center_.y, node.xyz_center_.z);
                float v_size    = node.voxel_size;
                float2 depths   = RayVoxelIntersection( // get the intersected info.
                    ray_start_, ray_dir_, v_center, v_size
                );

                if(depths.x == -1 || depths.y == -1) continue; // when meet the not valid node.
                // Adjust whether the min or max depths have appeared in intersected voxels (important.).

                // int rid = atomicAdd(&(index_queue[queue_basic_idx]), 1); // syn adding.
                // hit_idxs[output_basic_idx + rid]  = cur_node_idx;
                // min_depth[output_basic_idx + rid] = depths.x;
                // max_depth[output_basic_idx + rid] = depths.y;
                // ray_total_valid_len[queue_basic_idx] += (depths.y - depths.x);
                
                // assert(depths.y > depths.x);
                
                // judge whether the intersected (min,max depths) have appeared in the intersected buffer.
                // needed to be accumlated !
                bool valid_flag = true;
                for (int j=0; j < valid_idx; j++) { // adjust whether there is an intersection (min_d, max_d are existed.)
                    if (valid_min_depths[j] == depths.x && valid_max_depths[j] == depths.y) {
                        int pre_node_idx = valid_idxs[j]; // get the previous node's idx;
                        OcTree pre_node = shared_octree[pre_node_idx];
                        if (pre_node.voxel_size > v_size) { // if current node's resolution is smaller, we will replace the idx.
                            valid_idxs[j] = cur_node_idx; // update the curr node's idx;
                            valid_scale[j] = (float) start_voxel_size / v_size;
                        }
                        valid_flag = false; // assign the existed node.
                        break;
                    }
                }
                if (valid_flag) { // update the intersected nodes' buffer.
                    valid_idxs[valid_idx]       = cur_node_idx;
                    valid_min_depths[valid_idx] = depths.x;
                    valid_max_depths[valid_idx] = depths.y;
                    valid_scale[valid_idx]      = (float) start_voxel_size / v_size;
                    valid_idx ++; // add the valid idx;
                }
                
                // atomicAddFloat(&(ray_total_valid_len[queue_basic_idx]), depths.y - depths.x);
                // atomicMax(&(num_max_intersected_voxels[0]), valid_idx); // update the num of max intersected voxels.

                if (node.valid_num_leaves == 0) continue; // directly jump out the children nodes.
            }

            // TODO. perform voxel traversal in cur node (slower here when rate==2).
            // float child_v_size = v_size / rate;
            // float3 start_pos = make_float3(ray_start_.x + depths.x * ray_dir_.x, ray_start_.y + depths.x * ray_dir_.y, ray_start_.z + depths.x * ray_dir_.z);
            // float3 st_offset = make_float3(start_pos.x-v_center.x+v_size/2.0f, start_pos.y-v_center.y+v_size/2.0f, start_pos.z-v_center.z+v_size/2.0f);
            
            // int st_x = (int)(st_offset.x / child_v_size), st_y = (int)(st_offset.y / child_v_size), st_z = (int)(st_offset.z / child_v_size);
            // int3 st_pos = make_int3((st_x < rate) ? st_x : rate - 1,
            //                         (st_y < rate) ? st_y : rate - 1,
            //                         (st_z < rate) ? st_z : rate - 1);
            // // the intersection kernel.
            // float step_x = (ray_dir_.x >= 0) ? 1 : -1, step_y = (ray_dir_.y >= 0) ? 1 : -1, step_z = (ray_dir_.z >= 0) ? 1 : -1;
            // // get the first intersected point in boundaries.
            // float nxt_v_bd_x = (step_x > 0) ? (st_pos.x + step_x) * child_v_size : st_pos.x * child_v_size,
            //       nxt_v_bd_y = (step_y > 0) ? (st_pos.y + step_y) * child_v_size : st_pos.y * child_v_size,
            //       nxt_v_bd_z = (step_z > 0) ? (st_pos.z + step_z) * child_v_size : st_pos.z * child_v_size;

            // float t_max_x = (ray_dir_.x != 0) ? (nxt_v_bd_x - st_offset.x)/ray_dir_.x : DBL_MAX, // for example, MAX here.
            //       t_max_y = (ray_dir_.y != 0) ? (nxt_v_bd_y - st_offset.y)/ray_dir_.y : DBL_MAX,
            //       t_max_z = (ray_dir_.z != 0) ? (nxt_v_bd_z - st_offset.z)/ray_dir_.z : DBL_MAX;
            
            // float t_delta_x = (ray_dir_.x != 0) ? child_v_size/ray_dir_.x * step_x : DBL_MAX,
            //       t_delta_y = (ray_dir_.y != 0) ? child_v_size/ray_dir_.y * step_y : DBL_MAX,
            //       t_delta_z = (ray_dir_.z != 0) ? child_v_size/ray_dir_.z * step_z : DBL_MAX;
            
            // float tx = t_max_x, ty = t_max_y, tz = t_max_z;
            // int j=0, J=rate*rate*rate; // max iteration=8, e.g.;
            
            // while ( j < J ) {
            //     int i = st_pos.x*rate*rate + st_pos.y*rate + st_pos.z;
            //     int child_idx = node.children_gindex[i];
            //     if (child_idx != -1) { // the child is ok.
            //         // printf("idx, %d, %d\n", child_idx, j);
            //         idx_ptr++; idxs_stack[idx_ptr] = child_idx;
            //     }
                
            //     if (tx < ty) {
            //         if (tx < tz) {st_pos.x += step_x; tx += t_delta_x;}
            //         else {st_pos.z += step_z; tz += t_delta_z;}
            //     }
            //     else {
            //         if (ty < tz) {st_pos.y += step_y; ty += t_delta_y;}
            //         else {st_pos.z += step_z; tz += t_delta_z;}
            //     }
            //     if (st_pos.x < 0 || st_pos.x >= rate || st_pos.y < 0 || st_pos.y >= rate || st_pos.z < 0 || st_pos.z >= rate) break;
            //     j++;
            // }
            
            // get the children nodes.
            for (int i=0;i<MAX_LEAVES;i++) { // Need to judge the intersected children nodes.
                int child_idx = node.children_gindex[i];
                if (child_idx != -1) { // the child is ok.
                    idx_ptr++; idxs_stack[idx_ptr] = child_idx;
                } else break; // after -1, all children nodes' idxs are -1.
            }
        }
        
        // final update the global buffer.
        int rid = 0;
        for (int j=0; j < valid_idx; j++) {
            rid = atomicAdd(&(index_queue[queue_basic_idx]), 1); // syn adding, return the previous idx.
            hit_idxs[output_basic_idx  + rid] = valid_idxs[j];
            min_depth[output_basic_idx + rid] = valid_min_depths[j];
            max_depth[output_basic_idx + rid] = valid_max_depths[j];
            atomicAddFloat(&(ray_total_valid_len[queue_basic_idx]), (valid_max_depths[j] - valid_min_depths[j]) * valid_scale[j]);
            // printf("rate : %f\n", valid_scale[j]);
        }
        atomicMax(&(num_max_intersected_voxels_fine[0]), rid+1); // update the num of max intersected voxels.
        //
    }
}


// svo intersection.
__global__ void ray_octree_intersection(
    OcTree * octree_nodes, // all octree nodes.
    int coarse_level_start_idx, // the start index of the coarse tree.
    int * queue_control_list, // the num of valid octree nodes
    const float * ray_start, // [B, N, 3], (N=1024)
    const float * ray_dir, // [B, N, 3]
    // the output tensors.
    int * hit_idxs, // the hitted voxels' index, [B, N, n_max_hits.]
    float * min_depth, // the inserted min z, [B, N, n_max_hits.]
    float * max_depth, // the inserted max z, [B, N, n_max_hits.]
    int * index_queue, // the controled index, [B, N, 1]
    // the parameters.
    const float * voxel_size, // [B,1] the size of voxel.
    const int n_ray, // in parallel (n_rays * n_nodes).
    const int n_max_hits, // the num of intersected voxels(maxs).
    const int *num_nodes_batches, // [N1, N2, N3...NB] totally batch is B.
    const int num_nodes_total, // N1+N2+N3..+NB
    const int batch_size // num of batch.
) {
    // reference to NSVF & An efficient parametric algorithm for octree traversal
    #pragma unroll
    CUDA_KERNEL_LOOP(k, n_ray * num_nodes_total) { // N1 * n_ray + N2 * n_ray + ... + NB * n_ray.
        OcTree * o_coarse_nodes = octree_nodes + coarse_level_start_idx; // all coarse nodes.

        // int batch_idx = 0, tmp = 0, tmp_k = k;
        // for (; batch_idx < batch_size; batch_idx++) {
        //     int num_nodes = num_nodes_batches[batch_idx] * n_ray;
        //     if (k >= tmp && k < tmp + num_nodes) break;
        //     tmp += num_nodes;
        // }
        // assert(batch_idx < batch_size);
        // // printf("batch id: %d \n ", batch_idx);
        // for (int i=1;i<=batch_idx;i++) {
        //     tmp_k -= n_ray * num_nodes_batches[i-1];
        // }
        
        // get basic indexs.
        int coarse_node_idx = (int) k / n_ray;
        int ray_idx = k - coarse_node_idx * n_ray;
        OcTree cur_node = o_coarse_nodes[coarse_node_idx]; // get the current octree node.
        int batch_idx = cur_node.batch_id;
        float curr_voxel_size = voxel_size[batch_idx];

        // int ray_idx = (int) k / num_nodes_total;
        // int coarse_node_idx = k - ray_idx * num_nodes_total;
        size_t ray_basic_idx    = batch_idx * n_ray * 3 + ray_idx * 3;
        size_t queue_basic_idx  = batch_idx * n_ray + ray_idx;
        size_t output_basic_idx = batch_idx * n_ray * n_max_hits + ray_idx * n_max_hits;
        // printf("ray, node: %d, %d, %d \n ", coarse_node_idx, ray_idx, num_nodes_total);
        
        if (cur_node.valid == 0) continue;

        int idxs_stack[N_STACK_MAX_NODES] = {-1}; // DFS.
        int idx_ptr = 0, cur_node_idx = -1;
        idxs_stack[idx_ptr] = cur_node.gindex; 
        while (idx_ptr >= 0) {
            cur_node_idx = idxs_stack[idx_ptr];
            OcTree node = octree_nodes[cur_node_idx]; // get the node.

            float2 depths = RayVoxelIntersection( // get the intersected info.
                make_float3(ray_start[ray_basic_idx+0], ray_start[ray_basic_idx+1], ray_start[ray_basic_idx+2]),
                make_float3(ray_dir[ray_basic_idx+0], ray_dir[ray_basic_idx+1], ray_dir[ray_basic_idx+2]),
                make_float3(node.xyz_center_.x, node.xyz_center_.y, node.xyz_center_.z),
                node.voxel_size
            );

            idxs_stack[idx_ptr] = -1; idx_ptr--;
            if (depths.x > -1) { // this voxel is intersected.
                // append the min,max depth to the output bbox.
                int rid = atomicAdd(&(index_queue[queue_basic_idx]), 1);
                hit_idxs[output_basic_idx + rid]  = cur_node_idx;
                min_depth[output_basic_idx + rid] = depths.x;
                max_depth[output_basic_idx + rid] = depths.y;
                
                // get the children nodes.
                for (int i=0;i<MAX_LEAVES;i++) { // Need to judge the intersected children nodes.
                    // TODO.
                    int child_idx = node.children_gindex[i];
                    if (child_idx != -1) { // the child is ok.
                        idx_ptr++; idxs_stack[idx_ptr] = child_idx;
                    }
                }
            }
        }
        
    }
    
}


// intersection of rays and voxels.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
MultiLayerOctree::octree_traversal(
    torch::Tensor batch_rays_ori, // [B, N, 3] (N=1024)
    torch::Tensor batch_rays_dir, // [B, N, 3]
    // torch::Tensor idxs, torch::Tensor min_depths, torch::Tensor max_depths, torch::Tensor index_q_control,
    // torch::Tensor idxs_c, torch::Tensor min_depths_c, torch::Tensor max_depths_c, torch::Tensor index_q_control_c, 
    // torch::Tensor ray_total_valid_len,
    const int n_max_hits_coarse, const int n_max_hits,
    const int start_level, const int th_early_stop, const int opt_num_intersected_voxels
) {
    cudaSetDevice(this->device); // on GPU device.
    CUDA_CHECK_ERRORS();

    int num_rays    = batch_rays_ori.size(1); // default as 1024.
    int num_batches = batch_rays_ori.size(0); // batch size.
    // assert(num_batches == batch_size);

    // the coarse level ptr.
    int * num_voxels_coarse = this->num_nodes_levels_batches[this->num_levels - start_level - 1];
    int num_v_coarse_total  = this->num_octree_nodes_levels[this->num_levels - start_level - 1];
    int coarse_start_idx    = this->octree_nodes_index[this->num_levels - start_level - 1];
    float * ray_start_ptr   = batch_rays_ori.contiguous().data_ptr<float>();
    float * ray_dir_ptr     = batch_rays_dir.contiguous().data_ptr<float>();
    int * volume_top_ptr    = this->dense_volume_ptr[this->num_levels - start_level - 1];
    int vol_dim_bot         = this->volume_dims[0];
    int vol_dim_top         = this->volume_dims[num_levels - start_level - 1];
    num_max_intersected_voxels[0] = 0; // update the pool.
    num_max_intersected_voxels_fine[0] = 0;

    /* init all tensors */
    torch::Tensor ray_total_valid_len = torch::full({batch_size, num_rays, 1}, 0, batch_rays_ori.options());
    torch::Tensor index_q_control_c = torch::full({batch_size, num_rays, 1}, 0, batch_rays_ori.options()).to(torch::kInt);
    torch::Tensor idxs_c = torch::full({batch_size, num_rays, n_max_hits_coarse}, -1, batch_rays_ori.options()).to(torch::kInt);
    torch::Tensor min_depths_c = torch::full({batch_size, num_rays, n_max_hits_coarse}, -1, batch_rays_ori.options());
    torch::Tensor max_depths_c = torch::full({batch_size, num_rays, n_max_hits_coarse}, -1, batch_rays_ori.options());

    torch::Tensor index_q_control = torch::full({batch_size, num_rays, 1}, 0, batch_rays_ori.options()).to(torch::kInt);
    torch::Tensor idxs = torch::full({batch_size, num_rays, n_max_hits}, -1, batch_rays_ori.options()).to(torch::kInt);
    torch::Tensor min_depths = torch::full({batch_size, num_rays, n_max_hits}, -1, batch_rays_ori.options());
    torch::Tensor max_depths = torch::full({batch_size, num_rays, n_max_hits}, -1, batch_rays_ori.options());
    
    // block size, grid size;
    // printf("num of nodes: %d \n", num_v_coarse_total);
    
    /* ------ original version -------*/
    // cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, ray_octree_intersection, 0, 0);
    // grid_size = (int) (num_v_coarse_total * num_rays + block_size - 1) / block_size;

    // ray_octree_intersection <<< grid_size, block_size >>> (
    //     RAW_PTR(this->octree_nodes), coarse_start_idx, RAW_PTR(queue_control), ray_start_ptr, ray_dir_ptr,
    //     idxs.contiguous().data_ptr<int>(), min_depths.contiguous().data_ptr<float>(), max_depths.contiguous().data_ptr<float>(), index_q_control.contiguous().data_ptr<int>(),
    //     this->voxel_size, num_rays, n_max_hits, num_voxels_coarse, num_v_coarse_total, batch_size
    // );
    /* -----------------------------*/
    
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, ray_voxel_traversal, 0, 0);
    grid_size = (int) (batch_size * num_rays + block_size - 1) / block_size;
    // get all coarse voxels (of max resolutions) by voxel travelsal.
    ray_voxel_traversal <<< grid_size, block_size >>> (
        volume_top_ptr, RAW_PTR(this->octree_nodes), coarse_start_idx, ray_start_ptr, ray_dir_ptr,
        idxs_c.contiguous().data_ptr<int>(), min_depths_c.contiguous().data_ptr<float>(), max_depths_c.contiguous().data_ptr<float>(), index_q_control_c.contiguous().data_ptr<int>(),
        ray_total_valid_len.contiguous().data_ptr<float>(),
        this->volume_bbox, this->voxel_size, num_rays, n_max_hits_coarse, num_v_coarse_total, vol_dim_bot, vol_dim_top, batch_size, RAW_PTR(num_max_intersected_voxels), 
        th_early_stop, opt_num_intersected_voxels, false, false
    );
    
    if (num_max_intersected_voxels[0] == 0) 
        // return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>{idxs, min_depths, max_depths, ray_total_valid_len};
        return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
           {idxs, min_depths, max_depths, ray_total_valid_len, idxs_c, min_depths_c, max_depths_c};
           
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, ray_octree_traversal, 0, 0);
    grid_size = (int) (batch_size * num_rays * num_max_intersected_voxels[0] + block_size - 1) / block_size;
    
    ray_octree_traversal <<< grid_size, block_size >>> (
        RAW_PTR(this->octree_nodes), ray_start_ptr, ray_dir_ptr,
        idxs_c.contiguous().data_ptr<int>(), min_depths_c.contiguous().data_ptr<float>(), max_depths_c.contiguous().data_ptr<float>(), 
        idxs.contiguous().data_ptr<int>(), min_depths.contiguous().data_ptr<float>(), max_depths.contiguous().data_ptr<float>(), index_q_control.contiguous().data_ptr<int>(),
        ray_total_valid_len.contiguous().data_ptr<float>(),
        num_rays, n_max_hits_coarse, n_max_hits, 
        RAW_PTR(num_max_intersected_voxels_fine), num_max_intersected_voxels[0], batch_size, this->rate
    );
    // thrust::host_vector<int> num_max_intersected_voxels_host = num_max_intersected_voxels_fine;
    // printf("max_voxels : %d\n", num_max_intersected_voxels_host[0]);
    // coarse level intersection, no need for getting total rays' len.
    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
           {idxs, min_depths, max_depths, ray_total_valid_len, idxs_c, min_depths_c, max_depths_c};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
MultiLayerOctree::voxel_traversal(
    torch::Tensor batch_rays_ori, // [B, N, 3] (N=1024)
    torch::Tensor batch_rays_dir, // [B, N, 3]
    const int n_max_hits_coarse, 
    const int start_level, const int th_early_stop, const bool is_training, 
    const int opt_num_intersected_voxels
) {
    cudaSetDevice(this->device); // on GPU device.
    CUDA_CHECK_ERRORS();

    int num_rays    = batch_rays_ori.size(1); // default as 1024.
    int num_batches = batch_rays_ori.size(0); // batch size.
    // assert(num_batches == batch_size);

    // the coarse level ptr.
    int * num_voxels_coarse = this->num_nodes_levels_batches[this->num_levels - start_level - 1];
    int num_v_coarse_total  = this->num_octree_nodes_levels[this->num_levels - start_level - 1];
    int coarse_start_idx    = this->octree_nodes_index[this->num_levels - start_level - 1];
    float * ray_start_ptr   = batch_rays_ori.contiguous().data_ptr<float>();
    float * ray_dir_ptr     = batch_rays_dir.contiguous().data_ptr<float>();
    int * volume_top_ptr    = this->dense_volume_ptr[this->num_levels - start_level - 1];
    int vol_dim_bot         = this->volume_dims[0];
    int vol_dim_top         = this->volume_dims[num_levels - start_level - 1];
    num_max_intersected_voxels[0] = 0; // update the pool.

    /* init all tensors */
    torch::Tensor ray_total_valid_len = torch::full({batch_size, num_rays, 1}, 0, batch_rays_ori.options());
    torch::Tensor index_q_control_c = torch::full({batch_size, num_rays, 1}, 0, batch_rays_ori.options()).toType(torch::kInt);
    torch::Tensor idxs_c = torch::full({batch_size, num_rays, n_max_hits_coarse}, -1, batch_rays_ori.options()).toType(torch::kInt);
    // check-max depth values float type here.
    torch::Tensor min_depths_c = torch::full({batch_size, num_rays, n_max_hits_coarse}, -1, batch_rays_ori.options());
    torch::Tensor max_depths_c = torch::full({batch_size, num_rays, n_max_hits_coarse}, -1, batch_rays_ori.options());

    // torch::Tensor index_q_control = torch::full({batch_size, num_rays, 1}, 0, batch_rays_ori.options()).to(torch::kInt);
    // torch::Tensor idxs = torch::full({batch_size, num_rays, n_max_hits}, -1, batch_rays_ori.options()).to(torch::kInt);
    // torch::Tensor min_depths = torch::full({batch_size, num_rays, n_max_hits}, -1, batch_rays_ori.options());
    // torch::Tensor max_depths = torch::full({batch_size, num_rays, n_max_hits}, -1, batch_rays_ori.options());
    
    // block size, grid size;
    // printf("num of nodes: %d \n", num_v_coarse_total);
    
    /* ------ original version -------*/
    // cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, ray_octree_intersection, 0, 0);
    // grid_size = (int) (num_v_coarse_total * num_rays + block_size - 1) / block_size;

    // ray_octree_intersection <<< grid_size, block_size >>> (
    //     RAW_PTR(this->octree_nodes), coarse_start_idx, RAW_PTR(queue_control), ray_start_ptr, ray_dir_ptr,
    //     idxs.contiguous().data_ptr<int>(), min_depths.contiguous().data_ptr<float>(), max_depths.contiguous().data_ptr<float>(), index_q_control.contiguous().data_ptr<int>(),
    //     this->voxel_size, num_rays, n_max_hits, num_voxels_coarse, num_v_coarse_total, batch_size
    // );
    /* -----------------------------*/
    CUDA_CHECK_ERRORS();
    
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, ray_voxel_traversal, 0, 0);
    grid_size = (int) (batch_size * num_rays + block_size - 1) / block_size;
    // get all coarse voxels (of max resolutions) by voxel travelsal.
    ray_voxel_traversal <<< grid_size, block_size >>> (
        volume_top_ptr, RAW_PTR(this->octree_nodes), coarse_start_idx, ray_start_ptr, ray_dir_ptr,
        idxs_c.contiguous().data_ptr<int>(), min_depths_c.contiguous().data_ptr<float>(), max_depths_c.contiguous().data_ptr<float>(), index_q_control_c.contiguous().data_ptr<int>(),
        ray_total_valid_len.contiguous().data_ptr<float>(),
        this->volume_bbox, this->voxel_size, num_rays, n_max_hits_coarse, num_v_coarse_total, vol_dim_bot, vol_dim_top, batch_size, RAW_PTR(num_max_intersected_voxels), 
        th_early_stop, opt_num_intersected_voxels, true, is_training
    );

    CUDA_CHECK_ERRORS();

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>{idxs_c, min_depths_c, max_depths_c, ray_total_valid_len};
}