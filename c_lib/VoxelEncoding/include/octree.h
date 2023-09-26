#pragma once
#include "cuda_helper.h"
#include <utility>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Our Octree defined here : save 8 node leaves & whether it need to be updated.
// the octree is defined on a Volume, is resolution is the vol_len.
#define NODE_OCTREES_DEVICE thrust::device_vector<OcTree> // multi-res octree nodes.
#define NODE_CORNERS_DEVICE thrust::device_vector<OcTreeCorner> // multi-res
#define NODE_OCTREES_HOST thrust::host_vector<OcTree> // multi-res octree nodes.
#define MAX_XYZ_DIM 3
#define MAX_LEAVES 4*4*4
// the basic octree structure:

// to record the 8 corner points (along with the index) of the octree.
typedef struct OcTreeCorner
{
    u_short valid; // label for corner nodes, default as 1.0;

    u_short batch_id; // the current bid (person id.); 0～65535
    uint32_t gindex; // the global corner idx in the global vector.
    u_short xyz_idx[3]; // the xyz_indexs
    float xyz_pos[3]; // the global position in world coordinate.
    
} OcTreeCorner;


// to record the octree nodes of the octree.
typedef struct OcTree
{
    u_short valid; // label for validation, default as 1.0;
    
    int depth; // current level
    int gindex; // the index in global vector.
    int vindex; // the item index in volume.
    int valid_num_leaves; // 0, indicate the non-leaf node.
    int size; // the voxel's size; e.g., 512/256 = 2;
    int vol_dim; // the resolution of the volume.
    int batch_id; // the current bid;
    float voxel_size; // the physic unit size of a voxel.

    int3 xyz_center; // [idx_X, idx_Y, idx_Z];
    float3 xyz_center_; // [px, py, pz];
    float3 bbox_start; //  [px,py,pz]
    float3 xyz_corners[MAX_LEAVES]; // the corners xyz: [MAX_LEAVES, 3]
    // float * center; // [idx_X, idx_Y, idx_Z];
    // float * corners; //

    // struct OcTree *father; // recored the father node.
    // struct OcTree *children[MAX_LEAVES]; // eight sub octrees(ptr);
    int father_gindex; // the global index of father in nodes list.
    int children_gindex[MAX_LEAVES]; // save the global gindex of the child voxels.
    // uint32_t octree_corner_gindex[MAX_LEAVES]; // the global gindex of all the corner points;
    
    // the init function of OcTree;
    inline __host__ __device__
        void init(int3 xyz_center, float voxel_size, int vol_dim,
              int depth, int vindex, int gindex, int size, int batch_id) {
        this->valid      = (u_short) 1;
        this->xyz_center = xyz_center;
        this->voxel_size = voxel_size;
        this->depth      = depth;
        this->vindex     = vindex;
        this->gindex     = gindex;
        this->size       = size;
        this->vol_dim    = vol_dim;
        this->batch_id   = batch_id;
        this->valid_num_leaves = 0; // default as 0 leaves;
        // init the father & children nodes (invalid index;)
        this->father_gindex = -1;
        for (int i=0; i < MAX_LEAVES; i++) this->children_gindex[i] = -1;
    }

    inline __host__ __device__
        void set_father(OcTree * node, int father_index) { // the node ptr.
        assert(node != nullptr && node->valid == 1);
        this->father_gindex = father_index;
    }

    inline __host__ __device__
        void insert_child(OcTree * node, int index, int child_gindex) {
        assert(node != nullptr && node->valid == 1);
        assert(index < MAX_LEAVES && index >= 0); // update the children indexs.
        this->children_gindex[index] = child_gindex; // replace the ptr of index k with node;
    }
 
    inline __host__ __device__
        void update_state(void) {
        for (int i=0; i < MAX_LEAVES; i++) {
            if (this->children_gindex[i] != -1) this->valid_num_leaves++; // when id is not -1, valid.
        }
        // printf("valid max leaves: %d, %d \n", this->valid_num_leaves, this->father_gindex);
    }

    inline __host__ __device__
        OcTree &get_child_node(OcTree * all_nodes, int lindex) { // return a quote of the ptr;
        assert(lindex < MAX_LEAVES && lindex >= 0);
        int gindex = this->children_gindex[lindex]; // the global index in total list.
        return all_nodes[gindex];
    }

    inline __host__ __device__ 
        void judge_null_child_node(bool &is_null, int index) {
        assert(index < MAX_LEAVES && index >= 0);
        is_null = (this->children_gindex[index] == -1);
    }

    inline __host__ __device__
        void calculate_xyz_corners(void) {
        // must be calculated.
        assert(!(this->bbox_start.x == 0 && this->bbox_start.y == 0 && this->bbox_start.z == 0));
        // center position;
        float px_ = this->bbox_start.x + this->voxel_size * this->xyz_center.x;
        float py_ = this->bbox_start.y + this->voxel_size * this->xyz_center.y;
        float pz_ = this->bbox_start.z + this->voxel_size * this->xyz_center.z;
        // update the center xyz position.
        this->xyz_center_ = make_float3(px_, py_, pz_);
        
        int tmp_id = 0;
        for (int k=-1;k<=1;k+=2) {
            for (int m=-1;m<=1;m+=2) {
                for (int n=-1;n<=1;n+=2) {
                    this->xyz_corners[tmp_id++] = make_float3(px_ + k*this->voxel_size/2.0f, 
                                                              py_ + m*this->voxel_size/2.0f, 
                                                              pz_ + n*this->voxel_size/2.0f);
                }
            }
        }

    }

    inline __host__ __device__ 
        void calculate_bbox_start(float px, float py, float pz) {
        // px,py,pz are original volume's start index.
        float offset_voxel = 0.0f;
        float basic_voxel_size = this->voxel_size / this->size;
        int rate_vol = (int) log2f((float) this->size);
        // get offset.
        for (int i=0;i<rate_vol;i++) {
            offset_voxel += pow(2.0f, i) * basic_voxel_size / 2.0f;
        }
        // printf("%d, %f, %f \n", rate_vol, offset_voxel, basic_voxel_size);
        // move to new start, 
        // float px_ = px + offset_voxel + this->voxel_size * this->xyz_center.x;
        // float py_ = py + offset_voxel + this->voxel_size * this->xyz_center.y;
        // float pz_ = pz + offset_voxel + this->voxel_size * this->xyz_center.z;
        this->bbox_start = make_float3(px + offset_voxel,
                                       py + offset_voxel,
                                       pz + offset_voxel);
    }

} OcTree;


class MultiLayerOctree {

public:
    // initialization function & destoried function.
    ~MultiLayerOctree();
    void clear_buffer();
    explicit MultiLayerOctree(const int & num_levels, const int & batch_size,
                              const int & rate, const int & device) { // [L] storing the num of voxels.
        // dense_volumes_torch -> int ptr; total_num_occupied_voxels: int vector.
        this->device = device; this->rate = rate;
        this->num_levels = num_levels; this->batch_size = batch_size;
    }

    // init functions.
    void init_octree(std::vector<torch::Tensor>, std::vector<int>, std::vector<torch::Tensor>,
                     std::vector<int>, torch::Tensor, torch::Tensor);
    void init_octree_nodes(void); // init all the octree nodes.
    void init_octree_corners(void); // init all the octree corner points.

    // generate corners points (x,y,z), for grid sample the geometry features.
    torch::Tensor generate_corner_points(void);

    void build_init_variables(std::vector<torch::Tensor>, std::vector<int>, std::vector<torch::Tensor>, std::vector<int>);
    void init_voxels_info(torch::Tensor, torch::Tensor);
    void build_octree(void); // building the octree.

    // basic functions.
    void update_octree_by_o_volume(torch::Tensor dense_volume_bottom);
    void search_octree_nodes(void);

    // voxel intersections when passing rays.
    // given batch rays: [B,N_views, N_pixels,3(d)], return the near, far.
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
    octree_traversal(
        torch::Tensor, torch::Tensor, 
        const int, const int, const int, const int, const int
    );

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    voxel_traversal(
        torch::Tensor, torch::Tensor, const int, const int, const int, const bool, const int
    );

    // points sampling.
    std::tuple<torch::Tensor, torch::Tensor> ray_voxels_points_sampling( // hit_idx, min_depth, max_depth, sampled_idx, sampled_depth.
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
        torch::Tensor, const int, const int
    );

    std::tuple<torch::Tensor, torch::Tensor> ray_voxels_points_sampling_coarse(
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        const int, const int
    );

    // for torch tensor sorting.
    std::tuple<torch::Tensor, torch::Tensor> sort_samplings(torch::Tensor); 

    // Sampling features : given all the sampled points.
    // void grid_sample_features(
    //     std::vector<torch::Tensor>,  // input. [feature: (B*N_v, C, H, W), ...];
    //     std::vector<torch::Tensor>,  // output.[feature: (B*N_v, C, N_ray*N_points)];
    //     torch::Tensor, torch::Tensor, // ray_ori and ray_dir.
    //     torch::Tensor, torch::Tensor, // sampled_z & voxel_idx.
    //     torch::Tensor Ks, torch::Tensor RTs, // K and RT matrixs.
    //     torch::Tensor proj_xyz,
    //     const int ori_w, const int ori_h
    // );

    std::tuple<torch::Tensor, torch::Tensor> project_sampled_xyz(
        torch::Tensor ray_ori, torch::Tensor ray_dir,
        torch::Tensor sampled_z, torch::Tensor sampled_idx, 
        torch::Tensor Ks, torch::Tensor RTs, // [B, N_view, 3, 3] and [B, N_view, 3, 4]
        // torch::Tensor proj_xy, torch::Tensor proj_z, // output: [B, N_view, N_ray, N_sampled, 3] (x,y, z_local)
        const int ori_w, const int ori_h, const int num_views,
        const bool record_corners=true
    );

    torch::Tensor trilinear_aggregate_features(
        torch::Tensor ray_ori, torch::Tensor ray_dir,
        torch::Tensor sampled_z, torch::Tensor sampled_idx,
        torch::Tensor input_feats
    );

    torch::Tensor trilinear_aggregate_features_backward(
        torch::Tensor ray_ori, torch::Tensor ray_dir,
        torch::Tensor sampled_z, torch::Tensor sampled_idx,
        torch::Tensor input_feats, torch::Tensor grad_output
    );

    // undistoarion, not used now.
    void UndistortImage(std::string xml_file, std::string path, std::vector<std::string> list, std::string save_path);

    // utils
    torch::Tensor output_all_nodes_tensor(void);
    std::vector<OcTree> get_octree_nodes(void) {
        NODE_OCTREES_HOST octree_nodes_host = this->octree_nodes;
        std::vector<OcTree> nodes(octree_nodes_host.begin(), octree_nodes_host.end());
        return nodes;
    }
    int get_num_octree_nodes(void) {return this->num_octree_nodes;}
    int get_num_levels(void) {return this->num_levels;}
    int get_batch_size(void) {return this->batch_size;}
    int * get_volume_ptr(int index) { // in device.
        assert(index >= 0 && index < this->num_levels); // index aranges.
        return this->dense_volume_ptr[index];
    }
    std::vector<int> get_volume_dim(void) {
        std::vector<int> volume_dims_std(this->volume_dims.begin(), this->volume_dims.end());
        return volume_dims_std;
    }
    std::vector<int> get_num_valid_nodes() {
        // return the octree nodes vector.
        thrust::host_vector<int> queue_control_host = this->queue_control; // to host.
        std::vector<int> queue_control_std(queue_control_host.begin(), queue_control_host.end());
        return queue_control_std;
    }


private:
    // thrust::host_vector<NODE_OCTREES_DEVICE> octrees_nodes_2;
    NODE_OCTREES_DEVICE octree_nodes; // all batch octree nodes are integrated in one device vec.
    // NODE_CORNERS_DEVICE octree_corners; // all batch octree corners are integrated in one device vec.
    thrust::device_vector<int> queue_control; // to control the index of valid nodes.
    // thrust::device_vector<int> queue_control_corners; // to control the index of all the corners.
    thrust::host_vector<int*>  dense_volume_ptr; // saving the ptr for each pointer.
    thrust::host_vector<int>   volume_dims; // the dimension of volumes
    thrust::host_vector<int>   num_octree_nodes_levels; // the num of octree nodes in each level.
    thrust::host_vector<int*>  num_nodes_levels_batches; // the num of octree nodes for each batch in each level.
    thrust::host_vector<int>   octree_nodes_index; // the nodes start index in list.
    thrust::device_vector<int> num_max_intersected_voxels;
    thrust::device_vector<int> num_max_intersected_voxels_fine;
    __constant__ float * voxel_size; // [B] in device.
    __constant__ float * volume_bbox; // [B,3,2] in device.

    __constant__ int batch_size; // batch nums.
    __constant__ int num_levels; // num of levels.
    __constant__ int num_octree_nodes; // the total num of the octree nodes.
    __constant__ int device; // the device for GPU.
    __constant__ int rate; // the rate of the two-layers.
    OcTree tmp_node;
    // OcTreeCorner tmp_corner;

    // the variables for CUDA kernel funtions.
    int block_size;
    int grid_size;
    int min_grid_size;
    int num_in_parallel_total;
};


// building octree.
void build_octree(
    std::vector<torch::Tensor> o_volumes,
    std::vector<torch::Tensor> num_occupied_voxels,
    const int level, 
    const int rate,
    const int device
);

std::vector<torch::Tensor> undistort_images(
	// torch::Tensor bgs, // [2, 3, h, w]
	torch::Tensor input_rgbs, // [2, 3, h, w]
	torch::Tensor input_depths, // [2, 1, h', w']
	// parameters of colors and depths.
	torch::Tensor K_colors, // [2, 3, 3]
	torch::Tensor K_depths, // [2, 3, 3]
	torch::Tensor DIS_C, // [2, 5]
	torch::Tensor DIS_D,  // [2, 5]
	const int device
);