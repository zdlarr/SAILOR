#include "../include/octree.h"
#include <omp.h>
#include <chrono>
#include <utility>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>

void get_all_nodes_torch(
    std::vector<torch::Tensor> dense_volumes,
    torch::Tensor voxel_size,
    torch::Tensor vol_bbox,
    std::vector<int> total_num_occupied_voxels,
    std::vector<int> octree_nodes_index,
    torch::Tensor all_voxels,
    const int rate,
    const int device
);

void build_multi_res_occupied_volumes(
    std::vector<torch::Tensor> o_volumes,
    std::vector<torch::Tensor> num_occupied_voxels, // [B,1]*L
    const int level,
    const int rate,
    const int device
);

void interface_build_m_res_volumes(
    std::vector<torch::Tensor> o_volumes, // [B,N**3] * L
    std::vector<torch::Tensor> num_occupied_voxels, // [B,1]*L
    const int level,
    const int rate,
    const int device
) {
    for (torch::Tensor &volume : o_volumes) { CHECK_INPUT(volume); }
    for (torch::Tensor &num_voxels : num_occupied_voxels) {CHECK_INPUT(num_voxels);}
    build_multi_res_occupied_volumes(o_volumes, num_occupied_voxels, level, rate, device);
}

void interface_get_all_nodes_torch(
    std::vector<torch::Tensor> dense_volumes,
    torch::Tensor voxel_size,
    torch::Tensor vol_bbox,
    std::vector<int> total_num_occupied_voxels,
    std::vector<int> octree_nodes_index,
    torch::Tensor all_voxels,
    const int rate,
    const int device
) {
    for (torch::Tensor &volume : dense_volumes) { CHECK_INPUT(volume); }
    CHECK_INPUT(voxel_size); CHECK_INPUT(vol_bbox); CHECK_INPUT(all_voxels);
    get_all_nodes_torch(dense_volumes, voxel_size, vol_bbox,
                        total_num_occupied_voxels, octree_nodes_index,
                        all_voxels, rate, device);
}


void build_octree(
    std::vector<torch::Tensor> o_volumes,
    std::vector<torch::Tensor> num_occupied_voxels,
    const int level,
    const int rate,
    const int device
) {
    cudaSetDevice(device);
    CUDA_CHECK_ERRORS();

    int batch_size = (int) o_volumes[0].size(0);
    int vol_dim = (int) o_volumes[0].size(1); // e.g., 512
    // size_t num_threads = omp_get_num_procs();
    
    // init all octree nodes.
    // thrust::device_vector<OcTree*> octree_nodes; // totally: [B(N1) + .. B(N_l)]
    // octree_nodes.resize(level * )
    
    // build the octree-nodes for the coarse level.
    int l = level - 1;
    int vol_dim_curr = (int) vol_dim / pow(rate, l);
    size_t num_voxels = batch_size * vol_dim_curr * vol_dim_curr * vol_dim_curr;
    
    CHECK_INPUT(o_volumes[l]);
    int32_t * coa_ovol_ptr = o_volumes[l].contiguous().data_ptr<int32_t>();
    thrust::device_vector<int32_t> coa_ovol_vec(coa_ovol_ptr, coa_ovol_ptr + num_voxels);
    printf("val: %d \n", coa_ovol_vec[10]);
    
    // #pragma omp parallel for num_threads(2 * num_threads - 1)
    // for (int k=0; k < num_voxels; k ++) {
    //     printf("idx: %d \n", coarest_o_volume_ptr[k]);
    //     if (coarest_o_volume_ptr[k] != -1) {
    //         // printf("idx: %d \n", coarest_o_volume_ptr[k]);
    //         // OcTree *node = new OcTree;
            
    //     }
    // }
        
    // up-to-down building octree.
    // for (l=level-2; l >= 0; l--) { // l=level-1; the coarsest volume;
    //     vol_dim_curr = (int) vol_dim / pow(rate, l);
    //     num_voxels = vol_dim_curr * vol_dim_curr * vol_dim_curr;
    //     for (int i=0; i< batch_size; i++) {
    //         const int *dense_o_volume_ptr = o_volumes[l][i].contiguous().data_ptr<int>();
            
    //         //in parallel.
    //         // for 
    //     }
    // }
    
    // for (torch::Tensor &volume : o_volumes) { 
    //     CHECK_INPUT(volume); // [B, N, N, N]
        
    //     for (int i=0; i< batch_size; i++) {
            
    //     }
    // }
}