/*
    The depth to color script.
*/

#include <omp.h>
#include <iostream>
#include <random>
#include <cmath>
#include <ctime>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

namespace py = pybind11;
using namespace std;
using namespace Eigen;
using namespace cv;


// void points2depth(
//     py::array_t<float> points_clouds, // the points cloud 
//     py::array_t<float> &depth2color, // the depth map aligned with color map.
//     py::array_t<float> K_c, py::array_t<float> RT_c, // the camera parameters.
//     const size_t height_c,
//     const size_t width_c,
//     const size_t n_points
// ) {
//     py::buffer_info points_buf = points_clouds.request();
//     py::buffer_info depth2color_buf = depth2color.request();
//     py::buffer_info K_c_buf = K_c.request();
//     py::buffer_info RT_c_buf = RT_c.request();

//     float *points_ptr = (float *)points_buf.ptr;
//     float *depth2color_ptr = (float *)depth2color_buf.ptr;
//     float *K_c_ptr   = (float *)K_c_buf.ptr;
//     float *RT_c_ptr = (float *)RT_c_buf.ptr;

//     Eigen::MatrixXf K_c_mat, RT_c_mat, depth2color_mat, points_mat;
//     depth2color_mat.resize(height_c, width_c);
//     depth2color_mat.fill(0.f);

//     // ! note that all matrix all transposed.
//     Map<MatrixXf> K_c_map(K_c_ptr, 3, 3);
//     Map<MatrixXf> RT_c_map(RT_c_ptr, 4, 3);  // assign the K & RT matrix.
//     Map<MatrixXf> points_map(points_ptr, 3, n_points);
//     K_c_mat = K_c_map; RT_c_mat = RT_c_map; points_mat = points_map; // [3, N_p]
//     K_c_mat.transposeInPlace(); RT_c_mat.transposeInPlace();

//     Eigen::MatrixXf R_c_mat = RT_c_mat.block(0,0,3,3), // matrix of [3,3];
//                     T_c_mat = RT_c_mat.col(3); // matrix of [3, 1];

//     size_t num_threads = omp_get_num_procs();
//     #pragma omp parallel for num_threads(2 * num_threads - 1)
//     for (int n=0; n < n_points; n++) {
//         Eigen::MatrixXf ver = points_mat.col(n); // [3,1]
//         Eigen::MatrixXf tt = R_c_mat * ver + T_c_mat; // to color camera's coordinate [3,1];
//         float d_c = tt(2,0);
//         tt /= d_c; // divided by depths.
//         Eigen::MatrixXf pc = K_c_mat * tt; // to color camera's pixel coordinate.
//     }
// }

void depth2color(
    py::array_t<float> depths, // depth is of float value origin (cm), for unsigned int (0, 2**16)
    py::array_t<float> &depth2color, // the depth map aligned with color map.
    py::array_t<float> K_d, py::array_t<float> K_c, py::array_t<float> RT_d2c,
    const size_t height_c,
    const size_t width_c
) {
    py::buffer_info depths_buf = depths.request();
    py::buffer_info depth2color_buf = depth2color.request();
    py::buffer_info K_d_buf = K_d.request();
    py::buffer_info K_c_buf = K_c.request();
    py::buffer_info RT_d2c_buf = RT_d2c.request();

    float *depths_ptr = (float *)depths_buf.ptr;
    float *depth2color_ptr = (float *)depth2color_buf.ptr;
    float *K_d_ptr   = (float *)K_d_buf.ptr;
    float *K_c_ptr = (float *)K_c_buf.ptr;
    float *RT_d2c_ptr = (float*)RT_d2c_buf.ptr;

    size_t height_d = depths_buf.shape[0],
           width_d  = depths_buf.shape[1];

    // redefine the depth2color map;
    // std::vector<ssize_t> output_shape = {(ssize_t) height_c, (ssize_t) width_c};
    // py::array_t<float> depth2color(output_shape);
    // py::buffer_info depth2color_buf = depth2color.request();
    // float *depth2color_buf_ptr = (float *) depth2color_buf.ptr;

    Eigen::MatrixXf RT_d2c_mat, K_d_mat, K_c_mat, depths_mat, depth2color_mat;
    depth2color_mat.resize(height_c, width_c);
    depth2color_mat.fill(0.f);
    
    // ! note that all matrix all transposed.
    Map<MatrixXf> depths_map(depths_ptr, width_d, height_d);
    Map<MatrixXf> RT_d2c_map(RT_d2c_ptr, 4, 3);
    Map<MatrixXf> K_d_map(K_d_ptr, 3, 3);
    Map<MatrixXf> K_c_map(K_c_ptr, 3, 3);
    depths_mat = depths_map;
    RT_d2c_mat = RT_d2c_map; // [3, 4];
    K_d_mat = K_d_map; K_c_mat = K_c_map;
    K_d_mat.transposeInPlace(); K_c_mat.transposeInPlace(); RT_d2c_mat.transposeInPlace();
    depths_mat.transposeInPlace();

    Eigen::MatrixXf R_d2c_mat = RT_d2c_mat.block(0,0,3,3), // matrix of [3,3];
                    T_d2c_mat = RT_d2c_mat.col(3); // matrix of [3, 1];

    #pragma omp parallel for collapse(2)
    for (int h = 0; h < height_d; h++) {
        for (int w = 0; w < width_d; w++) {
            float d = depths_mat(h, w) / 1000.f;
            if (d < 0.2 || d > 2.2f) continue; // when d > 2.5m, we don't consider here;
            
            Eigen::MatrixXf ver = Eigen::MatrixXf::Zero(3, 1);
            ver << float(w), float(h), 1.f;
            // get the camera coordinate's position;
            ver = (K_d_mat.inverse() * ver) * d; // P_C = (K^{-1} * ver) * d;

            Eigen::MatrixXf tt = R_d2c_mat * ver + T_d2c_mat; // to color camera's coordinate [3,1];
            float d_c = tt(2,0);
            tt /= d_c; // divided by depths.
            Eigen::MatrixXf pc = K_c_mat * tt; // to color camera's pixel coordinate.

            int u = std::round(pc(0,0)), v = std::round(pc(1,0));
        
            // assign the depth2color mat.
            if (u >= 0 && u < width_c && v >= 0 && v < height_c) {
                depth2color_mat(v, u) = float(d_c * 1000.f); // the updated depth2color mat 
            }
        }
    }

    Eigen::MatrixXf depth2color_tmp = depth2color_mat;
    int size_w = 2;

    #pragma omp parallel for collapse(2)
    for (int h = size_w; h < height_c - size_w; h++) {
        for (int w = size_w; w < width_c - size_w; w++) {
            float d_tmp = depth2color_tmp(h, w);
            if (d_tmp < 1e-5) { // given a small numel for depth2color mat;
                float sum = 0.f;
                float num_h = 0.f;
                for (int dh = -size_w; dh <= size_w; dh++) {
					for (int dw = -size_w; dw <= size_w; dw++) {
						float d_h = depth2color_tmp(h+dh, w+dw);
                        if (d_h > 0.f) {
                            sum += d_h;
                            num_h += 1.f;
                        }
                    }
                }
                if (num_h > 0) {
                    depth2color_mat(h, w) = float(sum / num_h);
                }
            }
            float d_final = depth2color_mat(h, w);
            // filter the invalid value of the depth2color mat.
            if (std::isinf(d_final) || std::isnan(d_final)) depth2color_mat(h, w) = 0.f;
        }
    }
    
    // ofstream fout("./test.txt");
    // fout << depth2color_mat;
    // fout.close();

    // erode opeation;
	cv::Mat tmp_h = cv::Mat::ones(3, 3, CV_32F);
    cv::Mat depth2color_cvmat_erode, depth2color_cvmat;
    cv::eigen2cv(depth2color_mat, depth2color_cvmat); // depth-cvmat [h,w]
    cv::erode(depth2color_cvmat, depth2color_cvmat_erode, tmp_h);
    cv::cv2eigen(depth2color_cvmat_erode, depth2color_mat);
    
    Eigen::MatrixXf depth2color_mat_T = depth2color_mat.transpose();

    std::memcpy(depth2color_ptr, depth2color_mat_T.data(), sizeof(float) * (height_c * width_c));
}

PYBIND11_MODULE(DepthToColor, m) {
    m.doc() = "Utils for transform depth to color map";
    m.def("depth2color", &depth2color, "depth to color");
    // m.def("points2depth", &points2depth, "points to depth");
}