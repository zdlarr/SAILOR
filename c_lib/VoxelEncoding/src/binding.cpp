
#include "../include/freq_encoding.h"
#include "../include/sh_encoding.h"
#include "../include/sampling.h"
#include "../include/depth_normalizer.h"
#include "../include/volume_rendering.h"
#include "../include/fusion.cuh"
#include "../include/torch_where.h"
#include "../include/depth2color.h"
// #include "fusion.cpp"
#include "octree.cpp"


#include <pybind11/pybind11.h>
namespace py=pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("integrate", &fusion_cuda_integrate);
    m.def("integrate2", &fusion_cuda_integrate_refined);
    m.def("get_origin_bbox", &get_origin_bbox);
    m.def("get_center_xyz", &get_center_xyz);
    m.def("build_octree", &build_octree);
    
    m.def("build_multi_res_volumes", &interface_build_m_res_volumes);
    // m.def("test_gen_octree_nodes", &interface_test_gen_octree_nodes);
    m.def("get_all_nodes_torch", &interface_get_all_nodes_torch);
    // volume rendering functions.
    m.def("volume_rendering_training_forward", &volume_rendering_training_forward);
    m.def("volume_rendering_occ_forward", &volume_rendering_occ_forward);
    m.def("volume_rendering_training_backward", &volume_rendering_training_backward);
    // frequency encoding.
    m.def("freq_encode_forward", &freq_encode_forward);
    m.def("freq_encode_backward", &freq_encode_backward);
    // sh encoding.
    m.def("sh_encode_forward", &sh_encode_forward);
    m.def("sh_encode_backward", &sh_encode_backward);
    // depth normalizer.
    m.def("depth_normalize", &depth_normalize);
    // ray sampling, rgbdv sampling.
    m.def("rays_calculating", &rays_calculating);
    m.def("rays_calculating_parallel", &rays_calculating_parallel);
    m.def("rays_selecting", &rays_selecting);
    m.def("rgbdv_selecting", &rgbdv_selecting);
    // udf_calculating
    m.def("udf_calculating", &udf_calculating);
    m.def("udf_calculating_v2", &udf_calculating_v2);
    // torch utils.
    m.def("tensor_selection", &tensor_selection);
    m.def("bbox_mask", &bbox_mask);
    // distortion.
    m.def("undistort_images", &undistort_images);
    // depth2color warpping.
    m.def("depth2color", &depth2color);
    
    // the octree functions.
    py::class_<MultiLayerOctree>(m, "MultiLayerOctree")
       .def(py::init<const int &, const int &, const int &, const int &>())
       .def("init_octree", &MultiLayerOctree::init_octree)
       .def("get_nodes", &MultiLayerOctree::get_octree_nodes)
       .def("get_num_nodes", &MultiLayerOctree::get_num_octree_nodes)
       .def("get_num_nodes_valid", &MultiLayerOctree::get_num_valid_nodes)
       .def("build_octree", &MultiLayerOctree::build_octree)
       .def("get_nodes_tensor", &MultiLayerOctree::output_all_nodes_tensor)
       .def("octree_traversal", &MultiLayerOctree::octree_traversal)
       .def("voxel_traversal", &MultiLayerOctree::voxel_traversal)
       .def("ray_voxels_points_sampling", &MultiLayerOctree::ray_voxels_points_sampling)
       .def("ray_voxels_points_sampling_coarse", &MultiLayerOctree::ray_voxels_points_sampling_coarse)
       .def("sort_samplings", &MultiLayerOctree::sort_samplings)
       .def("undistort_image", &MultiLayerOctree::UndistortImage)
    //    .def("sample_features", &MultiLayerOctree::grid_sample_features)
       .def("project_sampled_xyz", &MultiLayerOctree::project_sampled_xyz)
       .def("trilinear_aggregate_features", &MultiLayerOctree::trilinear_aggregate_features)
       .def("generate_corner_points", &MultiLayerOctree::generate_corner_points)
       .def("trilinear_aggregate_features_backward", &MultiLayerOctree::trilinear_aggregate_features_backward);
}