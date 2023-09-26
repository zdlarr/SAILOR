/*
    The depth augmentation
*/

#include <omp.h>
#include <iostream>
#include <random>
#include <cmath>
#include <ctime>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

namespace py = pybind11;
using namespace std;
using namespace Eigen;
using namespace cv;

template <typename dtype>
dtype generate_random(dtype mean = 0, dtype std = 0.5)
{
    static std::default_random_engine e(time(NULL));
    static std::normal_distribution<dtype> n(mean, std);
    return n(e);
}

template <typename dtype>
void generate_gaussian_map(Eigen::Matrix<dtype, -1, -1> &mat, const size_t height, const size_t width)
{
    mat.resize(height, width);
    mat.fill(static_cast<dtype>(0));
    mat = mat.unaryExpr([](dtype dummy) {
        return generate_random<dtype>();
    });
}

template <typename dtype>
void meshgrid(const Eigen::Matrix<dtype, -1, 1> &x,
              const Eigen::Matrix<dtype, -1, 1> &y,
              Eigen::Matrix<dtype, -1, -1> &X,
              Eigen::Matrix<dtype, -1, -1> &Y)
{
    size_t num_threads = omp_get_num_procs();
    
    const size_t nx = x.size(), ny = y.size();
    X.resize(nx, ny);
    Y.resize(nx, ny);

    // build the meshgrid matrix.
    #pragma omp parallel for num_threads(2 * num_threads - 1)
    for (size_t i = 0; i < ny; i++) {
        X.col(i) = x;
    }
    #pragma omp parallel for num_threads(2 * num_threads - 1)
    for (size_t i = 0; i < nx; i++) {
        Y.row(i) = y.transpose();
    }
}

template <typename DerivedV, typename DerivedB>
auto Clip(const Eigen::ArrayBase<DerivedV> &v,
          const Eigen::ArrayBase<DerivedB> &bound)
    -> decltype(v.min(bound).max(-bound))
{
    return v.min(bound).max(-bound);
}

void add_gaussian_shift(Eigen::MatrixXf depth_map,
                        Eigen::MatrixXf &depth_shift,
                        const size_t height,
                        const size_t width,
                        const float std=0.5f)
{
    // randomly generate the shift image.
    Eigen::MatrixXf gaussian_shift_x, gaussian_shift_y, xp, yp;
    generate_gaussian_map<float>(gaussian_shift_x, height, width); // gaussian_shift matrix, of shape: [H, W];
    generate_gaussian_map<float>(gaussian_shift_y, height, width);
    
    // generate x,y meshgrid.
    auto x = Eigen::VectorXf::LinSpaced(height, 0, height-1).transpose(),
         y = Eigen::VectorXf::LinSpaced(width,  0, width-1).transpose();

    meshgrid<float>(x,y,xp,yp);
    // clip the xp, yp to [0, width-1], [0, height-1];
    xp = (xp + gaussian_shift_x).cwiseMin(height-1).cwiseMax(0); // xp and yp are of shape : [H, W];
    yp = (yp + gaussian_shift_y).cwiseMin(width-1).cwiseMax(0);

    // transform eigen matrix to opencv mat.
    {
        cv::Mat depth_cvmat, xp_cvmat, yp_cvmat;
        cv::eigen2cv(depth_map, depth_cvmat); // depth-cvmat [h,w]
        cv::eigen2cv(xp, xp_cvmat); // xp, yp cvmat: [h,w]
        cv::eigen2cv(yp, yp_cvmat);
        cv::Mat depth_shift_cvmat = depth_cvmat.clone();

        depth_shift.resize(height, width);
        cv::remap(depth_cvmat, depth_shift_cvmat, yp_cvmat, xp_cvmat, cv::INTER_LINEAR);
        cv::cv2eigen(depth_shift_cvmat, depth_shift); // re-transform cvmat to eigen matrix.
    }
    
}

void filter_disp(Eigen::MatrixXf disp, 
                 Eigen::MatrixXf &output_disp,
                 Eigen::MatrixXf kinect_pattern,
                 const size_t h_d,
                 const size_t w_d,
                 const size_t h_kp,
                 const size_t w_kp,
                 const float invalid_disp)
{
    const size_t num_threads = omp_get_num_procs();
    const size_t size_filter = 9;
    size_t center = size_t(size_filter / 2.f);
    Eigen::MatrixXf xf, yf;
    auto x = Eigen::VectorXf::LinSpaced(size_filter, 0, size_filter-1).transpose(),
         y = Eigen::VectorXf::LinSpaced(size_filter, 0, size_filter-1).transpose();
    
    meshgrid<float>(x,y,xf,yf);

    xf.array() -= center;
    yf.array() -= center;
    
    Eigen::MatrixXf sqr_radius = (xf.array().square() + yf.array().square()); // (x**2 + y **2)
    Eigen::MatrixXf vals = sqr_radius * std::pow(1.2, 2); // sqr * 1.2 ** 2;
    float * vals_ptr = vals.data();
    
    #pragma omp parallel for num_threads(2 * num_threads - 1)
    for (size_t i=0; i < vals.size(); i++) { // vals[vals == 0] = 1;
        if (vals_ptr[i] == 0) vals_ptr[i] = 1;
    }
    
    Eigen::MatrixXf weights = vals.array().cwiseInverse(); // weigths = 1 / vals;
    Eigen::MatrixXf fill_weights = (sqr_radius.array() + float(1)).array().cwiseInverse(); // weights = 1 / (1 + sqr_radius);
    float *fill_weights_ptr = fill_weights.data();
    
    #pragma omp parallel for num_threads(2 * num_threads - 1)
    for (size_t i=0; i < fill_weights.size(); i++) { // vals[vals == 0] = 1;
        if (fill_weights_ptr[i] > size_filter) fill_weights_ptr[i] = -1;
    }

    size_t lim_rows = h_d < h_kp ? h_d - size_filter : h_kp - size_filter;
    size_t lim_cols = w_d < w_kp ? w_d - size_filter : w_kp - size_filter;

    float window_inlier_distance_ = 0.1f;

    output_disp.resize(h_d, w_d); output_disp.fill(invalid_disp);
    Eigen::MatrixXf interpolation_map = Eigen::MatrixXf::Zero(h_d, w_d);

    #pragma omp parallel for collapse(2)
    for (size_t r = 0; r < lim_rows; r++) {
        for (size_t c = 0; c < lim_cols; c++) {
            if ( kinect_pattern(r + center, c + center) > 0 ) {
                Eigen::MatrixXf window = disp.block(r, c, size_filter, size_filter);
                Eigen::MatrixXf dot_window = kinect_pattern.block(r, c, size_filter, size_filter);
                float * window_ptr = window.data();
                float * dot_window_ptr = dot_window.data();

                std::vector<float> valid_dots, valid_dots_;
                assert(window.size() == dot_window.size());
                for (size_t i = 0; i < window.size(); i++) {
                    if (window_ptr[i] < invalid_disp) {
                        valid_dots.emplace_back(dot_window_ptr[i]);
                        valid_dots_.emplace_back(window_ptr[i]);
                    }
                }

                Eigen::VectorXf valid_dots_v = Eigen::Map<VectorXf, Eigen::Unaligned>(valid_dots.data(), valid_dots.size());
                Eigen::VectorXf valid_dots_v_ = Eigen::Map<VectorXf, Eigen::Unaligned>(valid_dots_.data(), valid_dots_.size());
                float n_valids = valid_dots_v.sum() / 255.f;
                float n_thresh = dot_window.sum() / 255.f;

                if (n_valids > n_thresh / 1.2f)
                {
                    float mean = valid_dots_v_.mean();
                    Eigen::MatrixXf diffs = (window.array() - mean).cwiseAbs();
                    Eigen::MatrixXf diffs_ = diffs.cwiseProduct(weights);
                    float *diffs_ptr = diffs_.data();
                    
                    Eigen::MatrixXf tmp_mat0, tmp_mat1;
                    tmp_mat0.resize(size_filter, size_filter);
                    tmp_mat1.resize(size_filter, size_filter);
                    float *tmp_mat0_ptr = tmp_mat0.data();
                    float *tmp_mat1_ptr = tmp_mat1.data();

                    for (size_t i = 0; i < window.size(); i++) {
                        if (window_ptr[i] < invalid_disp)
                            tmp_mat0_ptr[i] = dot_window_ptr[i];
                        else
                            tmp_mat0_ptr[i] = 0;

                        if (diffs_ptr[i] < window_inlier_distance_)
                            tmp_mat1_ptr[i] = 1;
                        else
                            tmp_mat1_ptr[i] = 0;
                    }

                    Eigen::MatrixXf cur_valid_dots = tmp_mat0.cwiseProduct(tmp_mat1);
                    float n_valids = cur_valid_dots.sum() / 255.f;

                    if (n_valids > n_thresh / 1.2f) {
                        float accu = window(center, center); 
                        assert(accu <= invalid_disp);
                        output_disp(r + center, c + center) = round((accu)*8.0) / 8.0;

                        float *interpolation_window_ptr = interpolation_map.data();
                        float *output_disp_tep_ptr = output_disp.data();

                        for (size_t i = 0; i < size_filter; i++) {
                            for (size_t j = 0; j < size_filter; j++) {
                                size_t idx = (c + i) * h_d + r + j;
                                size_t idx_ = i * size_filter + j;
                                if (interpolation_window_ptr[idx] < fill_weights_ptr[idx_]) {
                                    interpolation_window_ptr[idx] = fill_weights_ptr[idx_];
                                    output_disp_tep_ptr[idx] = output_disp(r + center, c + center);
                                }
                            }
                        }

                    }
                }
            }
        }
    }
    
}


void add_holes(Eigen::MatrixXf &depth_aug,
              Eigen::MatrixXf mask_mat,
              const size_t height,
              const size_t width,
              const float holes_rate=0.005f
) {
    // randomly add holes to the depth map.
    // step 1. find the values of one in the mask.
    const size_t num_threads = omp_get_num_procs();
    const float *mask_ptr = mask_mat.data();
    std::srand((unsigned)time(NULL));

    #pragma omp parallel for num_threads(2 * num_threads - 1)
    for (int i = 0; i < mask_mat.size(); i++) {
        if (mask_ptr[i] != 0) { // the valid region.
            float rd = std::rand() / float(RAND_MAX); // range in (0,1)
            if (rd < holes_rate) { // the holes.
                int idx_j = int(i / width),
                    idx_i = int(i % width);
                int hole_rad = (std::rand() % (3-0+1)) + 0;
                for (int k = idx_i - hole_rad; k < idx_i + hole_rad; k++) {
                    for (int j = idx_j - hole_rad; j < idx_j + hole_rad; j++) {
                        // clip the idx, incase that the idx is out of (height, width);
                        int k_clip = k >= 0 ? k : 0, j_clip = j >= 0 ? j : 0; 
                        k_clip = k_clip < height ? k_clip : height - 1;
                        j_clip = j_clip < width ? j_clip : width - 1;
                        depth_aug(k_clip, j_clip) = float(1e-5); // give a very little value.
                    }
                }
            }
        }
    }
}

void simulate_TOF(Eigen::MatrixXf &aug_depth, Eigen::MatrixXf depth_origin, 
                   const size_t height, const size_t width, const float z_size, const float sigma_d=-1.0f) {
    // add kinect noise, which considering the depth value.
    const float * depth_ori_ptr = depth_origin.data();
    float * depth_aug_ptr = aug_depth.data();
    const size_t num_threads = omp_get_num_procs();
    // the scale is validated in ~ 1.5;
    float scale = z_size / 1.5f;

    #pragma omp parallel for num_threads(2 * num_threads - 1)
    for (size_t i = 0; i < depth_origin.size(); i++) {
        float d = depth_ori_ptr[i] / scale;
        // choose a function that meet the property: the axis is ~0.5m, the min sigma is ~1 / 20f : (-b / 2a = 0.5, -0.25a + c = 1)
        // the returned value (mm);
        float sigma = (1.5 * std::pow(d, 2.f) - 1.5 * d + 1.375f) / 2000.f; // when using cm as degree, divided by 100 here;
        if (sigma_d > 0) sigma = sigma_d; // use the given sigma to train the model;
        
        static std::default_random_engine e(time(NULL));
        static std::normal_distribution<float> n(0, sigma);
        depth_aug_ptr[i] += n(e);
        if (depth_aug_ptr[i] < 0) depth_aug_ptr[i] = 0;
    }
}


void depth_blur(
    py::array_t<float> depths_map, // depth is of float value (cm)
    py::array_t<float> &blurred_depths_map, // the output blurred depth maps.
    const int blur_radius = 3,
    const float depth_thres = 0.03
) {
    py::buffer_info depths_map_buf = depths_map.request();
    py::buffer_info blurred_depths_map_buf = blurred_depths_map.request();

    float *depths_map_ptr = (float *)depths_map_buf.ptr;
    float *blurred_depths_map_ptr = (float *)blurred_depths_map_buf.ptr;

    size_t h_d = depths_map_buf.shape[0],
           w_d = depths_map_buf.shape[1];

    int k_width = (int) blur_radius / 2;
    // kernel avg filtering.
    // filter the valid depth value.
    const size_t num_threads = omp_get_num_procs();
    #pragma omp parallel for num_threads(2 * num_threads - 1)
    for (size_t i=0; i < h_d*w_d; i++) {
        size_t h_idx = (size_t) i / w_d;
        size_t w_idx = (size_t) i - h_idx * w_d;
        float depth_value = depths_map_ptr[i];
        
        int num_valid_pixel = 0;
        float depth_total = 0.0;
        #pragma omp parallel for collapse(2)
        for (int k = -k_width; k <= k_width; k++) {
            for (int l = -k_width; l <= k_width; l++) {
                size_t h_idx_ = h_idx + k;
                size_t w_idx_ = w_idx + l;
                // valid pixel here;
                if (h_idx_ >= 0 && h_idx_ < h_d && w_idx_ >= 0 && w_idx_ < w_d) {
                    size_t i_ = h_idx_ * w_d + w_idx_;
                    float depth_value_ = depths_map_ptr[i_];
                    float diff_depth = depth_value_ - depth_value;
                    if (diff_depth < depth_thres && diff_depth > - depth_thres) {
                        depth_total += depth_value_;
                        num_valid_pixel += 1;
                    }
                }
            }
        }
        // std::cout << num_valid_pixel << std::endl;
        blurred_depths_map_ptr[i] = depth_total / num_valid_pixel;
    }
}


void aug_depth(
    py::array_t<float> depths_map, // depth is of float value (cm)
    py::array_t<float> mask,
    py::array_t<float> kinect_pattern, //kinect pattern matrix.
    py::array_t<float> K, // intrinsic matrix.
    py::array_t<float> &aug_depths_map, // the output augmented depth map;
    const float scale_factor = 341.f, // the scale between depth(buf) and depth(cm); to 1.5m approximately.
    const float baseline_m = 0.02f,
    const float invalid_disp = 99999999.9f, // invalid disparity.
    const float z_size = 512.f, // the camera's position to the person center's length.
    const float holes_rate = 0.005f, // the default holes rate.
    const float sigma_d = -1.f
    )
{
    py::buffer_info depths_map_buf = depths_map.request();
    py::buffer_info aug_depths_map_buf = aug_depths_map.request();
    py::buffer_info mask_buf = mask.request();
    py::buffer_info kinect_pattern_buf = kinect_pattern.request();
    py::buffer_info K_buf = K.request();

    float *depths_map_ptr = (float *)depths_map_buf.ptr;
    float *aug_depths_map_ptr = (float *)aug_depths_map_buf.ptr;
    float *mask_ptr   = (float *)mask_buf.ptr;
    float *kinect_pattern_ptr = (float *)kinect_pattern_buf.ptr;
    float *K_ptr = (float*)K_buf.ptr;
    
    // get the shape of the eigen buffer;
    assert(depths_map_buf.shape == mask_buf.shape);
    assert(aug_depths_map_buf.shape == depths_map_buf.shape);
    size_t h_d = depths_map_buf.shape[0],
           w_d  = depths_map_buf.shape[1];
    size_t h_kp = kinect_pattern_buf.shape[0],
           w_kp = kinect_pattern_buf.shape[1];

    // transform the ptr to eigen matrix. ! notes that the matrix are all loaded in [W,H];
    float focal_x = K_ptr[0];
    Eigen::MatrixXf depths_map_mat, kinect_pattern_mat, mask_mat, depth_interp, output_disp;
    Eigen::Map<Eigen::MatrixXf> depths_map_map(depths_map_ptr, w_d, h_d);
    Eigen::Map<Eigen::MatrixXf> kinect_pattern_map(kinect_pattern_ptr, w_kp, h_kp);
    Eigen::Map<Eigen::MatrixXf> mask_map(mask_ptr, w_d, h_d);
    depths_map_mat = depths_map_map;
    kinect_pattern_mat = kinect_pattern_map;
    mask_mat = mask_map;
    // reshape to [H, W];
    depths_map_mat.transposeInPlace(); kinect_pattern_mat.transposeInPlace(); mask_mat.transposeInPlace();
    
    // depth to cm meters.
    depths_map_mat /= scale_factor;
    // randomly shift (gaussian)
    add_gaussian_shift(depths_map_mat, depth_interp, h_d, w_d, 0.5f);
    // calculate disparity.
    // disp = baseline * focal / (depth + 1e-6)
    Eigen::MatrixXf disp = (depth_interp.array() + float(1e-8)).cwiseAbs().cwiseInverse() * (focal_x * baseline_m);
    Eigen::MatrixXf depth_f = (disp * 8.0f).array().round() / 8.0f;  // round(disp * 8) / 8

    // add kinect noise on disparity
    filter_disp(depth_f, output_disp, kinect_pattern_mat, h_d, w_d, h_kp, w_kp, invalid_disp);
    // transform to depth
    Eigen::MatrixXf depth_aug = (output_disp.array() + float(1e-8)).cwiseInverse() * (focal_x * baseline_m);

    // filter the valid depth value.
    const size_t num_threads = omp_get_num_procs();
    #pragma omp parallel for num_threads(2 * num_threads - 1)
    for (size_t i = 0; i < h_d * w_d; i++)
    {
        if (output_disp.data()[i] == invalid_disp)
            depth_aug.data()[i] = 0;
    }
    depth_aug *= scale_factor;
    
    // simulate TOF noise finally.
    simulate_TOF(depth_aug, depths_map_mat * scale_factor, h_d, w_d, z_size, sigma_d);

    // randomly add some holes.
    add_holes(depth_aug, mask_mat, h_d, w_d, holes_rate);
    // transpose to [H, W].
    Eigen::MatrixXf depth_aug_t = depth_aug.transpose();

    std::memcpy(aug_depths_map_ptr, depth_aug_t.data(), sizeof(float) * (h_d * w_d));

}

PYBIND11_MODULE(DepthAug, m) {
    m.doc() = "Utils for depth augmentation";
    m.def("aug_depth", &aug_depth, "A fuction for depth augmentation.");
    m.def("depth_blur", &depth_blur, "A function for bluring depth maps.");
}