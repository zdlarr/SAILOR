#include <math.h>
#include <vector>
#include <stdint.h>
#include <assert.h>

// #include <omp.h>
#include "../include/cuda_helper.h"
#include "../include/octree.h"
// #include "./intersection.cpp"
#include <string>
#include <stdio.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>

// given the octree nodes, building the father&children in nxt level.
__global__ void gen_higher_octree_nodes (
    int * occupied_volume_nxt, // [B, N,N,N]
    OcTree * sparse_nodes_iden_curr, // [M1+M2+...+M_b]
    OcTree * sparse_nodes_iden_next, // [M1+M2+...+Mb]
    const int next_st_idx,
    // const int * num_o_voxels_curr, // [B]
    // const int * num_o_voxels_next, // [B]
    const int total_num_o_voxels_curr, // M1+M2+...+Mb;
    // const int total_num_o_voxels_next, // M1+M2+...+Mb;
    // const float * voxel_size, // [B]
    const float * vol_bbox, // [B,3,2]
    int * queue_control_nxt, // global idx buf.
    // const int voxel_dim_curr,
    const int rate
) {
    #pragma unrool
    CUDA_KERNEL_LOOP(i, total_num_o_voxels_curr) { // the parallel' num is the num_voxels.
        // for each item, get the children.
        OcTree & curr_node = sparse_nodes_iden_curr[i];
        if (!curr_node.valid) continue; // only valid voxels.
        int3 xyz_index     = curr_node.xyz_center;
        int vol_dim_curr   = curr_node.vol_dim; // the volume dim;
        int vol_dim_nxt    = vol_dim_curr * rate;
        int vol_size_cur   = vol_dim_curr * vol_dim_curr * vol_dim_curr;
        int vol_size_nxt   = vol_dim_nxt * vol_dim_nxt * vol_dim_nxt;
        int cur_depth      = curr_node.depth;
        int voxel_res      = curr_node.size;

        // int batch_id = 0, tmp_i = 0;
        // for (int k=0; k < batch_size; k++) {
        //     tmp_i += num_o_voxels_curr[k];
        //     if (i < tmp_i) break;
        //     else batch_id++;
        // } 
        int batch_id = curr_node.batch_id;
        float voxel_size_cur = curr_node.voxel_size;
        float voxel_size_nxt = (float) voxel_size_cur / rate; // e.g., current voxel size * 2;
        float3 vol_bbox_st = make_float3(vol_bbox[batch_id*6+0*2+0], 
                                         vol_bbox[batch_id*6+1*2+0], 
                                         vol_bbox[batch_id*6+2*2+0]);

        // get sub voxels indexs.
        size_t nxt_x_st = xyz_index.x * rate;
        size_t nxt_x_ed = (xyz_index.x + 1) * rate;
        size_t nxt_y_st = xyz_index.y * rate;
        size_t nxt_y_ed = (xyz_index.y + 1) * rate;
        size_t nxt_z_st = xyz_index.z * rate;
        size_t nxt_z_ed = (xyz_index.z + 1) * rate;
        
        // pass through each node to update the voxels' info. 2*2*2
        int tmp_id = 0;
        FOR_STEP(k,nxt_x_st, nxt_x_ed) {
            FOR_STEP(m,nxt_y_st,nxt_y_ed) {
                FOR_STEP(n,nxt_z_st,nxt_z_ed) {

                    size_t nxt_v_id = k*vol_dim_nxt*vol_dim_nxt+m*vol_dim_nxt+n;
                    int occ_val_nxt = (int) (occupied_volume_nxt[batch_id*vol_size_nxt+nxt_v_id] > -1);
                    if (occ_val_nxt) {
                        int nid_nxt = atomicAdd(&(queue_control_nxt[0]), 1); // increasing the control_list;
                        OcTree & nxt_node = sparse_nodes_iden_next[nid_nxt];
                        nxt_node.init(make_int3(k,m,n), voxel_size_nxt, vol_dim_nxt,
                                      cur_depth + 1, nxt_v_id, next_st_idx + nid_nxt,
                                      (int) (voxel_res / rate), // smaller size;
                                      batch_id);
                        // for each children nodes, re-calculate the bbox start & corners.
                        nxt_node.calculate_bbox_start(vol_bbox_st.x, vol_bbox_st.y, vol_bbox_st.z);
                        nxt_node.calculate_xyz_corners();
                        // update node's idx.
                        occupied_volume_nxt[batch_id*vol_size_nxt+nxt_v_id] = (int) (next_st_idx + nid_nxt);
                        // set father & children nodes.
                        nxt_node.set_father(&curr_node, curr_node.gindex);
                        curr_node.insert_child(&nxt_node, tmp_id++, nxt_node.gindex);
                    }
                    // tmp_id++; // add the index to the global idx, incase that idx is not matched.
                }
            }
        }
        //
        curr_node.update_state();
    }

}

inline __device__ bool if_single_occ_in_neighbors(
    const int * occupied_volume_top,
    const int batch_id,
    const int x, const int y, const int z,
    const int volume_dim, const int volume_size
) {
    // 3*3 = 27 step.
    FOR_STEP(k, x-1,x+2) {
        FOR_STEP(m, y-1,y+2) {
            FOR_STEP(n, z-1,z+2) {

                if ((k >= 0 && k < volume_dim) && (m >= 0 && m < volume_dim) && (n >= 0 && n < volume_dim) && !(k==x && m==y && n==z)) {
                    size_t idx = batch_id*volume_size + k*volume_dim*volume_dim + m*volume_dim + n;
                    if (occupied_volume_top[idx] > -1) return false;
                    // else continue;
                }

            }
        }
    }
    return true;
}


// up-to-down generate octree ptr in dense ptr matrix.
__global__ void gen_root_octree_nodes (
    int * occupied_volume_top, // [B, 8,8,8]
    int * occupied_volume_sec, // [B, 16,16,16]
    OcTree * sparse_nodes_iden_curr, // [M1+M2+...+M_b]
    OcTree * sparse_nodes_iden_next, // [M1+M2+...+M_b]
    // OcTreeCorner * sparse_corners_iden, // [OcTreeCorner indentity]
    const int curr_st_idx,
    const int next_st_idx,
    // two control buffer;
    int * queue_control, // init as 0;
    // int * queue_control_corners, // init as 0;
    // int queue_control_next, // default as 0; 
    const float * voxel_size, // [B]
    const float * vol_bbox, // [B, 3, 2];
    const int vol_dim_top, // such as 8.
    const int vol_dim_bottom, // such as 512;
    const int rate, // 2.
    const int batch_size
) {
    int curr_depth = (int) log2f((float) vol_dim_top); // 3
    size_t vol_size_curr = vol_dim_top * vol_dim_top * vol_dim_top; // e.g., 8.
    size_t vol_dim_nxt = vol_dim_top * rate; // e.g., 16
    size_t vol_size_next = vol_dim_nxt * vol_dim_nxt * vol_dim_nxt;

    #pragma unrool
    CUDA_KERNEL_LOOP(i, batch_size * vol_size_curr) { // [B,N,N,N]
        int occ_val = (int) (occupied_volume_top[i] > -1); // valid volume.
        
        if (occ_val) {
            int batch_id = static_cast<int>(i / vol_size_curr);
            int curr_v_id = i - batch_id * vol_size_curr;
            float voxel_size_bottom = voxel_size[batch_id]; // such as 512;
            float voxel_size_top    = (float) voxel_size_bottom * (vol_dim_bottom / vol_dim_top); // such as (512/8) * 0.003;

            // get x,y,z index.
            size_t curr_x = static_cast<size_t>( curr_v_id / (vol_dim_top*vol_dim_top) );
            size_t curr_y = static_cast<size_t>( (curr_v_id - curr_x*vol_dim_top*vol_dim_top) / vol_dim_top );
            size_t curr_z = static_cast<size_t>( curr_v_id - curr_x*vol_dim_top*vol_dim_top - curr_y*vol_dim_top );
            // filter out the single voxels (no neighbor nodes.)
            if (if_single_occ_in_neighbors(occupied_volume_top, batch_id, curr_x, curr_y, curr_z, vol_dim_top, vol_size_curr)) continue;

            float3 vol_bbox_st = make_float3(vol_bbox[batch_id*6+0*2+0], 
                                             vol_bbox[batch_id*6+1*2+0], 
                                             vol_bbox[batch_id*6+2*2+0]);

            int nid_curr = atomicAdd(&(queue_control[0]), 1); // get ptr id;
            // printf("id0: %d \n", nid_curr);
            // make attribution. (xyz, corner)
            int3 xyz_center = make_int3(curr_x, curr_y, curr_z);
            // init the octree-node.
            OcTree & curr_node = sparse_nodes_iden_curr[nid_curr];
            curr_node.init(xyz_center, voxel_size_top, vol_dim_top,
                            curr_depth, curr_v_id, curr_st_idx + nid_curr,
                            (int) vol_dim_bottom / vol_dim_top, 
                            batch_id);
            // update xyz start & corners. xyz_idx;
            curr_node.calculate_bbox_start(vol_bbox_st.x, vol_bbox_st.y, vol_bbox_st.z);
            // printf("vol_bbox_st, %f, %f \n", sparse_nodes_iden_curr[nid_curr].bbox_start.x, vol_bbox_st.x);
            curr_node.calculate_xyz_corners(); // xyz_p * 8;
            // printf("xyz corners: %f, %f, %f \n", curr_node.xyz_corners[0].x, curr_node.xyz_corners[0].y, curr_node.xyz_corners[0].z);
            // update the node's correspondent idx.
            occupied_volume_top[i] = (int) (curr_st_idx + nid_curr);

            // get nxt xyz ranges (min-max);
            size_t nxt_x_st = xyz_center.x * rate;
            size_t nxt_x_ed = (xyz_center.x + 1) * rate;
            size_t nxt_y_st = xyz_center.y * rate;
            size_t nxt_y_ed = (xyz_center.y + 1) * rate;
            size_t nxt_z_st = xyz_center.z * rate;
            size_t nxt_z_ed = (xyz_center.z + 1) * rate;

            // // e.g., 2*2*2 for.
            int tmp_id = 0;
            FOR_STEP(k,nxt_x_st, nxt_x_ed) {
                FOR_STEP(m,nxt_y_st,nxt_y_ed) {
                    FOR_STEP(n,nxt_z_st,nxt_z_ed) {
                        // get the voxel id in nxt volume (batch).
                        size_t nxt_v_id = k*vol_dim_nxt*vol_dim_nxt+m*vol_dim_nxt+n;
                        int occ_val_nxt = (int) (occupied_volume_sec[batch_id*vol_size_next+nxt_v_id] > -1);
                        if (occ_val_nxt) {
                            int nid_nxt = atomicAdd(&(queue_control[1]), 1);
                            // printf("id1: %d \n", nid_nxt);
                            float voxel_size_nxt = (float) voxel_size_bottom * (vol_dim_bottom / vol_dim_nxt);
                            
                            OcTree &nxt_node = sparse_nodes_iden_next[nid_nxt];
                            nxt_node.init(make_int3(k,m,n), voxel_size_nxt, vol_dim_nxt,
                                          curr_depth + 1, nxt_v_id, next_st_idx + nid_nxt,
                                          (int) (vol_dim_bottom / vol_dim_nxt),
                                          batch_id);
                            // for each children nodes, re-calculate the bbox start & corners.
                            nxt_node.calculate_bbox_start(vol_bbox_st.x, vol_bbox_st.y, vol_bbox_st.z);
                            nxt_node.calculate_xyz_corners();
                            // update node's correspondent idx.
                            occupied_volume_sec[batch_id*vol_size_next+nxt_v_id] = (int) (next_st_idx + nid_nxt);
                            // set father & children nodes.
                            nxt_node.set_father(&curr_node, curr_node.gindex);
                            curr_node.insert_child(&nxt_node, tmp_id++, nxt_node.gindex);
                            // printf("nxt node info: %f \n", curr_node.children[0].voxel_size);
                            // printf("curr node info: %f, %f \n", nxt_node.father->voxel_size, curr_node.voxel_size);
                        }

                    }
                }
            }
            // update information.
            curr_node.update_state();
            // printf("curr node children num: %d \n", curr_node.valid_num_leaves);
            // if (curr_node.valid_num_leaves > 0) {
                // printf("nxt node info: %d, %f \n", curr_node.children_gindex[0].father_gindex, );
            // }
        }
    }
    
}

// merge non-empty-nodes (not very efficient, but is not enough.).
__global__ void merge_non_empty_nodes(
    int * occupied_volume_curr, // [B, N,N,N, 1]
    // float * occupied_idx, // [B, ]
    int * num_occupied_voxels, // [B,1]
    // OcTree ** octree_nodes_curr, // [B,N*N*N]
    // OcTree ** octree_nodes_next, // [B,2N*2N*2N]
    // OcTree * octree_nodes_curr_buf, // [B,N*N*N]
    // OcTree * octree_nodes_next_buf, // [B,2N*2N*2N]
    const int * occupied_volume_next, // [B, 2N, 2N, 2N, 1];
    const int vol_dim_curr,
    const int vol_dim_next,
    const int batch_size
) {
    size_t vol_size_curr = vol_dim_curr * vol_dim_curr * vol_dim_curr;
    size_t vol_size_next = vol_dim_next * vol_dim_next * vol_dim_next;
    int rate_dim = (int) vol_dim_next / vol_dim_curr; // the tree's nodes : rate_dim^3;

    #pragma unrool
    CUDA_KERNEL_LOOP(i, batch_size * vol_size_curr) { // not in parallel of 
        int batch_id = static_cast<int>(i / vol_size_curr);
        int curr_voxel_id = i - batch_id * vol_size_curr;

        size_t curr_voxel_x = static_cast<size_t>( curr_voxel_id / (vol_dim_curr*vol_dim_curr) );
        size_t curr_voxel_y = static_cast<size_t>( (curr_voxel_id - curr_voxel_x*vol_dim_curr*vol_dim_curr) / vol_dim_curr );
        size_t curr_voxel_z = static_cast<size_t>( curr_voxel_id - curr_voxel_x*vol_dim_curr*vol_dim_curr - curr_voxel_y*vol_dim_curr );
        
        size_t next_voxel_x_start = curr_voxel_x * rate_dim;
        size_t next_voxel_x_end   = (curr_voxel_x + 1) * rate_dim;
        size_t next_voxel_y_start = curr_voxel_y * rate_dim;
        size_t next_voxel_y_end   = (curr_voxel_y + 1) * rate_dim;
        size_t next_voxel_z_start = curr_voxel_z * rate_dim;
        size_t next_voxel_z_end   = (curr_voxel_z + 1) * rate_dim;
        
        // e.g., 2*2*2's for;
        int is_occupied = 0;
        for (int k=next_voxel_x_start; k < next_voxel_x_end; k++) {
            for (int m=next_voxel_y_start; m < next_voxel_y_end; m++) {
                for (int n=next_voxel_z_start; n < next_voxel_z_end; n++) {
                    size_t next_voxel_id = batch_id * vol_size_next
                                         + k * vol_dim_next * vol_dim_next 
                                         + m * vol_dim_next + n;
                    int occupied_val = (int) (occupied_volume_next[next_voxel_id] > -1);
                    is_occupied += occupied_val;
                    // if (occupied_val && octree_nodes_next[next_voxel_id] == nullptr) {
                    //     // when the next_voxels are occupied here, and no ptr exists. Make a new ptr:
                    //     octree_nodes_next[next_voxel_id] = &octree_nodes_next_buf[next_voxel_id];
                        
                    // }
                }
            }

        }

        if (is_occupied) {
            // when meet the last level's octree. expand the.
            occupied_volume_curr[i] = curr_voxel_id;
            atomicAdd(&(num_occupied_voxels[batch_id]), 1);
        }
        else {occupied_volume_curr[i] = (int)-1;}

    }
}

// __global__ void filter_out_single_voxels() {
    // num_occupied_voxels
// }

__global__ void write_all_nodes(
    OcTree * octree_nodes_vec,
    float * nodes_all_positions, // [N, 5], each nodes has 8 float5(xyz+) points.
    const int num_occupied_voxels, // N
    const int num_item_voxels // default as 5.
    // const int batch_size // batch_id.
) {
    #pragma unrool
    CUDA_KERNEL_LOOP(i, num_occupied_voxels) {
        OcTree & node = octree_nodes_vec[i]; // get the octree node.
        int v_idx = i * num_item_voxels; // the start index of the num occupied voxels.
        nodes_all_positions[v_idx+0] = node.xyz_center_.x; // x
        nodes_all_positions[v_idx+1] = node.xyz_center_.y; // y
        nodes_all_positions[v_idx+2] = node.xyz_center_.z; // z
        nodes_all_positions[v_idx+3] = node.voxel_size; // voxel_size.
        nodes_all_positions[v_idx+4] = (float) node.valid;   // the valid item.
        nodes_all_positions[v_idx+5] = (float) node.batch_id;   // the batch_id
        // printf("infos: %f, %f \n", node.voxel_size, (float) node.batch_id);
    }
}


__global__ void write_all_nodes_corners(
    OcTree *octree_nodes_vec,
    float * nodes_all_corners, // [N, 8, 3]
    const int num_voxels
) {
    CUDA_KERNEL_LOOP(i, num_voxels*MAX_LEAVES*MAX_XYZ_DIM) {
        uint32_t voxel_idx = (uint32_t) (i / MAX_LEAVES*MAX_XYZ_DIM);
        uint32_t cidx      = (uint32_t) (i - voxel_idx*MAX_LEAVES*MAX_XYZ_DIM) / MAX_XYZ_DIM;
        uint32_t xyz_dim   = (uint32_t) (i - voxel_idx*MAX_LEAVES*MAX_XYZ_DIM - cidx*MAX_XYZ_DIM);

        OcTree & curr_node = octree_nodes_vec[voxel_idx];
        const float3 *cx = &curr_node.xyz_corners[0];
        
        if (xyz_dim == 0) {
            nodes_all_corners[i] = cx[cidx].x;
        } else if(xyz_dim == 1) {
            nodes_all_corners[i] = cx[cidx].y;
        } else {
            nodes_all_corners[i] = cx[cidx].z;
        }
        
        // #pragma unrool 8
        // for (int k=0; k < 8; k++) {
        //     uint32_t basic_cidx = i*MAX_LEAVES*MAX_XYZ_DIM + k*MAX_XYZ_DIM;
        //     const float c[3] = {cx[k].x, cx[k].y, cx[k].z}; // record the xyz corners.

        //     #pragma unrool 3
        //     for (int m=0; m < 3; m++) {
        //         nodes_all_corners[basic_cidx+m] = c[m];
        //     }
        // }

    }
}


__host__ void build_multi_res_occupied_volumes(
    std::vector<torch::Tensor> dense_volumes, // [B, N,N,N]*L,
    std::vector<torch::Tensor> num_occupied_voxels_torch, // [B,1]*L
    const int level, // total num of resolutions.
    const int rate,
    const int device
) {
    // std::vector<torch::Tensor> dense_volumes;
    // auto device_torch = dense_volumes[0].device();
    int batch_size = (int) dense_volumes[0].size(0);
    int vol_dim = (int) dense_volumes[0].size(1);
    // float * dense_o_volume_ptr = dense_occupied_volume.contiguous().data_ptr<float>();
    
    // dense_volumes.push_back(dense_occupied_volume);

    cudaSetDevice(device);
    CUDA_CHECK_ERRORS();
    int block_size, grid_size, min_grid_size;

    for (int i=0; i < level-1; i++) {
        int vol_dim_lower = (int) vol_dim / pow(rate, (i+1));
        // build new Tensor.
        // torch::Tensor dense_o_volume_lower = torch::zeros({batch_size, 
        //                                                    vol_dim_lower, 
        //                                                    vol_dim_lower, 
        //                                                    vol_dim_lower}).to(torch::kFloat).to(device_torch);
        
        // to ptr
        int * dense_o_volume_lower_ptr = dense_volumes[i+1].contiguous().data_ptr<int>();
        int * dense_o_voluem_next_ptr  = dense_volumes[i].contiguous().data_ptr<int>();
        
        // get num_voxels, update the total num of features.
        int * num_occupied_voxels = num_occupied_voxels_torch[i+1].contiguous().data_ptr<int>();

        size_t voxel_sizes = batch_size * vol_dim_lower * vol_dim_lower * vol_dim_lower;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, merge_non_empty_nodes, 0, 0);
        grid_size = (int) (voxel_sizes + block_size - 1) / block_size;
        // merge non empty nodes in parallel.
        merge_non_empty_nodes <<< grid_size, block_size >>> (
            dense_o_volume_lower_ptr, num_occupied_voxels, dense_o_voluem_next_ptr,
            vol_dim_lower, vol_dim_lower * rate,
            batch_size
        );
    }
}

__host__ thrust::tuple<thrust::device_vector<OcTree>, thrust::device_vector<int>> // return the queried nodes & queue_vec.
    test_gen_octree_nodes(
    std::vector<torch::Tensor> dense_volumes, // [B, N,N,N]*L,
    torch::Tensor voxel_size,
    torch::Tensor vol_bbox,
    std::vector<int> total_num_occupied_voxels, // L levels' nodes.
    std::vector<int> octree_nodes_index, // L levels' index.
    // const int num_occupied_voxels_curr,
    // const int num_occupied_voxels_next,
    const int rate,
    const int device
) {
    cudaSetDevice(device);
    CUDA_CHECK_ERRORS();
    
    int num_levels = dense_volumes.size();
    int batch_size = dense_volumes[0].size(0);
    float * voxel_size_ptr = voxel_size.contiguous().data_ptr<float>();
    float * vol_bbox_ptr   = vol_bbox.contiguous().data_ptr<float>();
    
    // get top2 layers' volume.
    torch::Tensor volume_top = dense_volumes[num_levels - 1];
    torch::Tensor volume_sec = dense_volumes[num_levels - 2];
    int * volume_top_ptr = volume_top.contiguous().data_ptr<int>();
    int * volume_sec_ptr = volume_sec.contiguous().data_ptr<int>();

    // saving these variables.
    int vol_dim_top = volume_top.size(1);
    int vol_dim_bot = dense_volumes[0].size(1);

    // init queue_control vectors (for numbers).
    thrust::device_vector<int> queue_control;
    queue_control.resize(num_levels);
    thrust::fill(queue_control.begin(), queue_control.end(), 0);
    
    // init the nodes vector for counting node numbers.
    thrust::device_vector<OcTree> octree_nodes_vec; // containing all nodes.
    OcTree tmp_node; tmp_node.valid = 0; // default valid = 0;
    int total_occupied_voxels = std::accumulate(total_num_occupied_voxels.begin(), total_num_occupied_voxels.end(), (int)0);
    octree_nodes_vec.resize(total_occupied_voxels);
    thrust::fill(octree_nodes_vec.begin(), octree_nodes_vec.end(), tmp_node);
    // printf("size: %d \n", octree_nodes_vec.size());

    int block_size, grid_size, min_grid_size;
    int num_voxels_total = batch_size * vol_dim_top * vol_dim_top * vol_dim_top;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, gen_root_octree_nodes, 0, 0);
    grid_size = (int) (num_voxels_total + block_size - 1) / block_size;
    
    // for the root nodes initialization.
    gen_root_octree_nodes <<< grid_size, block_size >>> (
        volume_top_ptr, volume_sec_ptr,
        RAW_PTR(octree_nodes_vec) + octree_nodes_index[num_levels - 1],
        RAW_PTR(octree_nodes_vec) + octree_nodes_index[num_levels - 2],
        octree_nodes_index[num_levels - 1], octree_nodes_index[num_levels - 2],
        RAW_PTR(queue_control), voxel_size_ptr, vol_bbox_ptr,
        vol_dim_top, vol_dim_bot, rate, batch_size
    );

    // generate higher nodes.
    for (int k=num_levels - 2; k > 0; k--) { // for last l-2 layers.
        int * volume_refine_ptr = dense_volumes[k-1].contiguous().data_ptr<int>();
        
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, gen_higher_octree_nodes, 0, 0);
        grid_size = (int) (total_num_occupied_voxels[k] + block_size - 1) / block_size;

        gen_higher_octree_nodes <<< grid_size, block_size >>> (
            volume_refine_ptr,
            RAW_PTR(octree_nodes_vec) + octree_nodes_index[k],
            RAW_PTR(octree_nodes_vec) + octree_nodes_index[k-1],
            octree_nodes_index[k-1],
            total_num_occupied_voxels[k],
            vol_bbox_ptr, RAW_PTR(queue_control) + (num_levels - k), rate
        );
    }

    // device to host vector, equal to std::vector, copy to host (waste time.)
    // thrust::host_vector<OcTree> host_octree_nodes = octree_nodes_vec;
    // thrust::host_vector<int>   host_queue_control = queue_control;
    return thrust::make_tuple(octree_nodes_vec, queue_control); // keep on GPU vectors.
}

__host__ void get_all_nodes_torch(
    std::vector<torch::Tensor> dense_volumes, // [B, N,N,N]*L,
    torch::Tensor voxel_size, // [B]
    torch::Tensor vol_bbox, // [B, 3, 2]
    std::vector<int> total_num_occupied_voxels, // L levels' nodes.
    std::vector<int> octree_nodes_index, // L levels' index.
    torch::Tensor all_voxels, // [N_voxels, 8, 3], saving the 8 corners.
    // const int num_occupied_voxels_curr,
    // const int num_occupied_voxels_next,
    const int rate,
    const int device
) {
    cudaSetDevice(device);
    CUDA_CHECK_ERRORS();
    // step1. get all nodes. vector<OcTree>
    auto octree_infos = test_gen_octree_nodes(dense_volumes, voxel_size, vol_bbox,
                                              total_num_occupied_voxels, octree_nodes_index,
                                              rate, device);
    int num_levels = dense_volumes.size();
    thrust::device_vector<OcTree> octree_nodes = thrust::get<0>(octree_infos);
    thrust::device_vector<int> queue_control   = thrust::get<1>(octree_infos);
    // input to ptr;
    float *all_voxels_ptr   = all_voxels.contiguous().data_ptr<float>();
    int num_occupied_voxels = all_voxels.size(0);
    int num_item_voxels     = all_voxels.size(1); // default as 5;

    // printf("id max : %d, %d, %d, %d \n", host_queue_control[0], host_queue_control[1], host_queue_control[2], host_queue_control[num_levels - 2]);
    // for (int k = 0; k < host_octree_nodes.size(); k++) {
    //     // printf("infos : %d, %d, %f\n", octree_nodes[k].valid_num_leaves, octree_nodes[k].gindex, octree_nodes[k].voxel_size);
    //     int child_idx  = host_octree_nodes[k].children_gindex[7];
    //     if (child_idx != -1) {
    //         int father_idx = host_octree_nodes[child_idx].father_gindex;
    //         printf("Infos : %d, %d, %f, %f \n", host_octree_nodes[father_idx].gindex, host_octree_nodes[k].gindex, host_octree_nodes[father_idx].bbox_start.x, host_octree_nodes[k].bbox_start.x);
    //     }
    // }

    int block_size, grid_size, min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, write_all_nodes, 0, 0);
    grid_size = (int) (num_occupied_voxels + block_size - 1) / block_size;
    
    write_all_nodes <<< grid_size, block_size >>> (
        RAW_PTR(octree_nodes), all_voxels_ptr, num_occupied_voxels, num_item_voxels
    );

}



// ***********************************************
// the interface functions for MultiLayerOctree.

MultiLayerOctree::~MultiLayerOctree() {
    this->clear_buffer();
}

void MultiLayerOctree::clear_buffer(void) {
    // clear the buffer.
    this->tmp_node.valid = (u_short) 0;
    // this->tmp_corner.valid = (u_short) 0;
    thrust::fill(this->octree_nodes.begin(), this->octree_nodes.end(), this->tmp_node);
    // thrust::fill(this->octree_corners.begin(), this->octree_corners.end(), this->tmp_corner);
    thrust::fill(this->queue_control.begin(), this->queue_control.end(), 0);
    // thrust::fill(this->queue_control_corners.begin(), this->queue_control_corners.end(), 0);
    thrust::fill(this->volume_dims.begin(), this->volume_dims.end(), 0);
    thrust::fill(this->num_octree_nodes_levels.begin(), this->num_octree_nodes_levels.end(), 0);
    thrust::fill(this->octree_nodes_index.begin(), this->octree_nodes_index.end(), 0);
    // the corner infos.
    // clear the ptr.
    thrust::fill(this->dense_volume_ptr.begin(), this->dense_volume_ptr.end(), nullptr);
    thrust::fill(this->num_nodes_levels_batches.begin(), this->num_nodes_levels_batches.end(), nullptr);
    thrust::fill(this->num_max_intersected_voxels.begin(), this->num_max_intersected_voxels.end(), 0);
    this->voxel_size  = nullptr;
    this->volume_bbox = nullptr;
}

void MultiLayerOctree::init_octree(
    std::vector<torch::Tensor> dense_volumes_torch,  // [B,N,N,N] * L
    std::vector<int> total_num_occupied_voxels,
    std::vector<torch::Tensor> num_occ_voxels_batches,
    std::vector<int> octree_nodes_index,
    torch::Tensor vol_size, // the voxel's size. [B]
    torch::Tensor vol_bbox
) {
    this->clear_buffer(); // first clear all buffers.
    this->num_levels = dense_volumes_torch.size();
    this->batch_size = dense_volumes_torch[0].size(0);
    this->num_octree_nodes = std::accumulate(total_num_occupied_voxels.begin(), total_num_occupied_voxels.end(), (int)0);
    this->tmp_node.valid = 0; // init tmp node (invalid).
    // this->tmp_corner.valid = 0; // init tmp corner (invalid)
    // init the variables (volume ptr, queue control vector.)
    // this->init_octree_nodes();
    this->build_init_variables(dense_volumes_torch, total_num_occupied_voxels, num_occ_voxels_batches,
                               octree_nodes_index);
    // init the octree nodes & corners points.
    this->init_octree_nodes();
    // this->init_octree_corners();
    
    this->init_voxels_info(vol_size, vol_bbox); // using voxel size,bbox to initizalize.
}

void MultiLayerOctree::init_octree_nodes(void) {
    this->octree_nodes.resize(this->num_octree_nodes); // the occupied size all octree_nodes,
    thrust::fill(this->octree_nodes.begin(), this->octree_nodes.end(), this->tmp_node);
}

void MultiLayerOctree::init_octree_corners(void) {
    // int dense_corners_num = (int) pow(this->volume_dims[0] + 1, 3);
    // this->octree_corners.resize(this->num_octree_nodes * MAX_LEAVES); // max existed cornerns num is num_voxels * 8 hopefully;
    // printf("num of corners : %d", this->num_octree_nodes * MAX_LEAVES);
    // initialize the each corners infos.
    // thrust::fill(this->octree_corners.begin(), this->octree_corners.end(), this->tmp_corner);
}

void MultiLayerOctree::build_init_variables(
    std::vector<torch::Tensor> dense_volumes,
    std::vector<int> total_num_occupied_voxels,
    std::vector<torch::Tensor> num_occ_voxels_batches,
    std::vector<int> octree_nodes_index                                        
) {
    // 1. the queue(index) controlable variables.
    this->queue_control.resize(this->num_levels);
    thrust::fill(this->queue_control.begin(), this->queue_control.end(), 0); // fill with 0 first.
    // this->queue_control_corners.resize(1);
    // thrust::fill(this->queue_control_corners.begin(), this->queue_control_corners.end(), 0); // fill with 0.

    // 2. initialize the volume ptr.
    this->dense_volume_ptr.resize(0);
    this->volume_dims.resize(0);
    for (torch::Tensor &volume : dense_volumes) {
        int * volume_ptr = volume.contiguous().data_ptr<int>(); // to int ptr;
        this->dense_volume_ptr.push_back(volume_ptr);
        this->volume_dims.push_back(volume.size(1)); // e.g., 512, 256, 128, ... 64;
    }
    // 3. initialize the num of voxels & indexs.
    this->num_octree_nodes_levels = thrust::host_vector<int>(total_num_occupied_voxels.begin(), total_num_occupied_voxels.end());
    this->octree_nodes_index      = thrust::host_vector<int>(octree_nodes_index.begin(), octree_nodes_index.end());
    // printf("nums: %d, %d \n", this->num_octree_nodes_levels[3], this->octree_nodes_index[3]);
    this->num_nodes_levels_batches.resize(0);
    for (torch::Tensor &num_voxels : num_occ_voxels_batches) {
        int * num_voxels_ptr = num_voxels.contiguous().data_ptr<int>();
        this->num_nodes_levels_batches.push_back(num_voxels_ptr);
    }
    this->num_max_intersected_voxels.push_back(0);
    this->num_max_intersected_voxels_fine.push_back(0);
}

void MultiLayerOctree::init_voxels_info(torch::Tensor vol_size, torch::Tensor vol_bbox) {
    // init the ptr in device.
    this->voxel_size  = vol_size.contiguous().data_ptr<float>();
    this->volume_bbox = vol_bbox.contiguous().data_ptr<float>();
}

// important function !.
void MultiLayerOctree::update_octree_by_o_volume(torch::Tensor dense_volume_bottom) {
    int * volume_ptr = dense_volume_bottom.contiguous().data_ptr<int>();
    // update the octree nodes (insert & delete).
}

void MultiLayerOctree::search_octree_nodes(void) {
    
}

void MultiLayerOctree::build_octree(void) {
    cudaSetDevice(this->device); // on GPU device.
    CUDA_CHECK_ERRORS();
    
    int vol_dim_top = this->volume_dims[this->num_levels - 1];
    int vol_dim_bot = this->volume_dims[0];

    int* volume_top_ptr = this->dense_volume_ptr[this->num_levels - 1];
    int* volume_sec_ptr = this->dense_volume_ptr[this->num_levels - 2];
    
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, gen_root_octree_nodes, 0, 0);
    grid_size = (int) (num_octree_nodes_levels[this->num_levels - 1] + block_size - 1) / block_size;
    
    // init the root nodes.
    gen_root_octree_nodes <<< grid_size, block_size >>> (
        volume_top_ptr, volume_sec_ptr,
        RAW_PTR(this->octree_nodes) + this->octree_nodes_index[num_levels - 1],
        RAW_PTR(this->octree_nodes) + this->octree_nodes_index[num_levels - 2],
        // RAW_PTR(this->octree_corners), // to obtain all corner idxs;
        this->octree_nodes_index[num_levels - 1], this->octree_nodes_index[num_levels - 2],
        RAW_PTR(queue_control),
        this->voxel_size, this->volume_bbox,
        vol_dim_top, vol_dim_bot, rate, batch_size
    );

    // generate higher nodes, if containing only two layers, not forwarding.
    for (int k=num_levels - 2; k > 0; k--) { // for last l-2 layers.
        int * volume_refine_ptr = this->dense_volume_ptr[k-1]; // the volume ptr.
        
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, gen_higher_octree_nodes, 0, 0);
        grid_size = (int) (num_octree_nodes_levels[k] + block_size - 1) / block_size;

        gen_higher_octree_nodes <<< grid_size, block_size >>> (
            volume_refine_ptr,
            RAW_PTR(this->octree_nodes) + octree_nodes_index[k],
            RAW_PTR(this->octree_nodes) + octree_nodes_index[k-1],
            octree_nodes_index[k-1],
            num_octree_nodes_levels[k],
            this->volume_bbox, RAW_PTR(queue_control) + (num_levels - k), rate
        );
    }
}

torch::Tensor MultiLayerOctree::output_all_nodes_tensor(
    // torch::Device device
) {
    cudaSetDevice(this->device);
    CUDA_CHECK_ERRORS();
    torch::Tensor all_nodes_tensor = torch::empty({this->num_octree_nodes, 6}, torch::ScalarType::Float).to(torch::kCUDA);

    float *all_voxels_ptr   = all_nodes_tensor.contiguous().data_ptr<float>();
    int num_occupied_voxels = all_nodes_tensor.size(0);
    int num_item_voxels     = all_nodes_tensor.size(1); // default as 5;
    
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, write_all_nodes, 0, 0);
    grid_size = (int) (this->num_octree_nodes + block_size - 1) / block_size;
    
    write_all_nodes <<< grid_size, block_size >>> (
        RAW_PTR(this->octree_nodes), all_voxels_ptr, this->num_octree_nodes, num_item_voxels
    );
    return all_nodes_tensor;
}

torch::Tensor MultiLayerOctree::generate_corner_points(void) {
    // get all the corner points of the voxels list and avoid the repeat corners.
    cudaSetDevice(this->device);
    CUDA_CHECK_ERRORS();
    
    torch::Tensor all_corners_tensor = torch::empty({this->num_octree_nodes, MAX_LEAVES, MAX_XYZ_DIM}, torch::ScalarType::Float).to(torch::kCUDA);
    torch::Tensor all_corners_xy     = torch::empty({this->num_octree_nodes, MAX_LEAVES, 2}, torch::ScalarType::Float).to(torch::kCUDA);
    torch::Tensor all_corners_z      = torch::empty({this->num_octree_nodes, MAX_LEAVES, 1}, torch::ScalarType::Float).to(torch::kCUDA);
    
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, write_all_nodes_corners, 0, 0);
    grid_size = (int) (this->num_octree_nodes*MAX_LEAVES*MAX_XYZ_DIM + block_size - 1) / block_size;

    write_all_nodes_corners <<< grid_size, block_size >>> (
        RAW_PTR(this->octree_nodes), 
        all_corners_tensor.contiguous().data_ptr<float>(),
        this->num_octree_nodes
    );
    
    return all_corners_tensor;
}