#include <math.h>
#include <vector>
#include <stdint.h>
#include <assert.h>

// #include <omp.h>
#include "../include/cuda_helper.h"
#include "../include/fusion.cuh"
#include "../include/octree.h"

using namespace std;
#define MAX_VALUE  9999999;
#define MIN_VALUE -9999999;

__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

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

__global__ void obtain_resized_bbox(
    const float* vol_origin, // [B, 3, 2(min,max)]
    const int batch_size,
    const int vol_dim, // the vol_dim is fixed.
    float *vol_bbox, // [B, 3, 2(min,max)]
    float *voxel_size // [B, 1] each bbox for batch has a voxel_sizes.
) {
    #pragma unroll
    CUDA_KERNEL_LOOP(i, batch_size) {
        float x_min = vol_origin[i*6+0*2+0],
              x_max = vol_origin[i*6+0*2+1],
              y_min = vol_origin[i*6+1*2+0],
              y_max = vol_origin[i*6+1*2+1],
              z_min = vol_origin[i*6+2*2+0],
              z_max = vol_origin[i*6+2*2+1];

        float x_len = x_max - x_min,
              y_len = y_max - y_min,
              z_len = z_max - z_min;

        float max_len = ::fmaxf(x_len, ::fmaxf(y_len, z_len)); // get the max len in one axis.
        
        float x_center = (x_min + x_max) / 2.0,
              y_center = (y_min + y_max) / 2.0,
              z_center = (z_min + z_max) / 2.0;

        voxel_size[i] = (float) max_len / (vol_dim - 1); // voxel size = total_len / (vol_dim - 1);
        if (max_len == x_len) { // update y & z 's minmax values.
            y_min = y_center - max_len / 2.0f;
            y_max = y_center + max_len / 2.0f;
            z_min = z_center - max_len / 2.0f;
            z_max = z_center + max_len / 2.0f;
        } else if (max_len == y_len) { // update x & z 's minmax values.
            x_min = x_center - max_len / 2.0f;
            x_max = x_center + max_len / 2.0f;
            z_min = z_center - max_len / 2.0f;
            z_max = z_center + max_len / 2.0f;
        } else { // update x & y 's minmax values.
            x_min = x_center - max_len / 2.0f;
            x_max = x_center + max_len / 2.0f;
            y_min = y_center - max_len / 2.0f;
            y_max = y_center + max_len / 2.0f;
        }

        // update the vol_bbox, with the min & max xyz;
        vol_bbox[i*6+0*2+0] = x_min;
        vol_bbox[i*6+0*2+1] = x_max;
        vol_bbox[i*6+1*2+0] = y_min;
        vol_bbox[i*6+1*2+1] = y_max;
        vol_bbox[i*6+2*2+0] = z_min;
        vol_bbox[i*6+2*2+1] = z_max;
    }
}

__global__ void inverse_mat3x3(
    const float* Ms, // [B, N, 3, 3]
    const int batch_num, 
    const int num_views,
    float* inv_Ms
) {
    CUDA_KERNEL_LOOP(i, batch_num*num_views) {
        size_t batch_id = static_cast<size_t>(i / num_views);
        size_t view_id  = i - batch_id * num_views;
        size_t basic_id = batch_id * num_views * 9 + view_id * 9;
        float a1 = Ms[basic_id+0],
              b1 = Ms[basic_id+1],
              c1 = Ms[basic_id+2],
              a2 = Ms[basic_id+3],
              b2 = Ms[basic_id+4],
              c2 = Ms[basic_id+5],
              a3 = Ms[basic_id+6],
              b3 = Ms[basic_id+7],
              c3 = Ms[basic_id+8];
        float det = a1*(b2*c3-c2*b3)-a2*(b1*c3-c1*b3)+a3*(b1*c2-c1*b2);
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

__global__ void obtain_origin_bbox(
    const float* depths_ims, // [B, N, H, W]
    const float* cam_intr_inv, // [B, N, 3, 3]
    const float* cam_R_inv, // [B, N, 3, 3]
    const float* cam_T, // [B, N, 3]
    size_t im_h,
    size_t im_w,
    int batch_num,
    int num_views,
    float * vol_origin, // [B, 3, 2(min,max)]
    float thres
) {
    // the function that return the bbox_infos in vol_origin: [B, 6(min,max)]
    // step1. reproject the depth map to 3D points.
    // float thres = 0.02f;
    #pragma unroll
    CUDA_KERNEL_LOOP(i, im_w*im_h*batch_num*num_views) {
        float d = depths_ims[i];
        if (d < 0.1f || d > 5.0f) continue; // first filter out the point (when d is not in the range.)

        size_t batch_id = static_cast<size_t>( i / (num_views*im_h*im_w) );
        size_t view_id  = static_cast<size_t>( (i - batch_id*num_views*im_h*im_w) / (im_h*im_w) );
        size_t pixel_y  = static_cast<size_t>( (i - batch_id*num_views*im_h*im_w - view_id*im_h*im_w) / im_w);
        size_t pixel_x  = static_cast<size_t>( i - batch_id*num_views*im_h*im_w - view_id*im_h*im_w - pixel_y * im_w );
        size_t idx_K    = batch_id*num_views*9 + view_id*9;
        size_t idx_T    = batch_id*num_views*3 + view_id*3;
        // printf("x,y: %d, %d \n", pixel_x, pixel_y);
        // get the 3D position of the point.
        // int_K @ [y,x,1] * d.
        float cam_pt_x = (cam_intr_inv[0+idx_K]*pixel_x + cam_intr_inv[1+idx_K]*pixel_y + cam_intr_inv[2+idx_K]) * d;
        float cam_pt_y = (cam_intr_inv[3+idx_K]*pixel_x + cam_intr_inv[4+idx_K]*pixel_y + cam_intr_inv[5+idx_K]) * d;
        float cam_pt_z = (cam_intr_inv[6+idx_K]*pixel_x + cam_intr_inv[7+idx_K]*pixel_y + cam_intr_inv[8+idx_K]) * d;
        // printf("cam x,y,z: %f, %f, %f \n", cam_pt_x, cam_pt_y, cam_pt_z);
        // to global position; int_R @ (p - T);
        cam_pt_x -= cam_T[0+idx_T]; cam_pt_y -= cam_T[1+idx_T]; cam_pt_z -= cam_T[2+idx_T];
        float pt_x = cam_R_inv[0+idx_K]*cam_pt_x + cam_R_inv[1+idx_K]*cam_pt_y + cam_R_inv[2+idx_K]*cam_pt_z;
        float pt_y = cam_R_inv[3+idx_K]*cam_pt_x + cam_R_inv[4+idx_K]*cam_pt_y + cam_R_inv[5+idx_K]*cam_pt_z;
        float pt_z = cam_R_inv[6+idx_K]*cam_pt_x + cam_R_inv[7+idx_K]*cam_pt_y + cam_R_inv[8+idx_K]*cam_pt_z;
        // printf("world x,y,z: %f, %f, %f \n", pt_x, pt_y, pt_z);
        // assign the min_max_value to the array.
        atomicMin(vol_origin + batch_id*6+0*2+0, pt_x - thres);
        atomicMax(vol_origin + batch_id*6+0*2+1, pt_x + thres);
        atomicMin(vol_origin + batch_id*6+1*2+0, pt_y - thres);
        atomicMax(vol_origin + batch_id*6+1*2+1, pt_y + thres);
        atomicMin(vol_origin + batch_id*6+2*2+0, pt_z - thres);
        atomicMax(vol_origin + batch_id*6+2*2+1, pt_z + thres);
    }
}

__global__ void obtain_center_points(
    const float* depths_ims, // [B, N, H, W]
    const float* cam_intr_inv, // [B, N, 3, 3]
    const float* cam_R_inv, // [B, N, 3, 3]
    const float* cam_T, // [B, N, 3]
    size_t im_h,
    size_t im_w,
    int batch_num,
    int num_views,
    float * vol_center, // [B, 3]
    float * vol_num_pts // [B, 1]
) {
    // the function that return the center of points in vol_origin: [B, 6(min,max)]
    // step1. reproject the depth map to 3D points.
    // float thres = 0.02f;
    #pragma unroll
    CUDA_KERNEL_LOOP(i, im_w*im_h*batch_num*num_views) {
        float d = depths_ims[i];
        if (d < 0.1f || d > 4.0f) continue; // first filter out the point (when d is not in the range.)

        size_t batch_id = static_cast<size_t>( i / (num_views*im_h*im_w) );
        size_t view_id  = static_cast<size_t>( (i - batch_id*num_views*im_h*im_w) / (im_h*im_w) );
        size_t pixel_y  = static_cast<size_t>( (i - batch_id*num_views*im_h*im_w - view_id*im_h*im_w) / im_w);
        size_t pixel_x  = static_cast<size_t>( i - batch_id*num_views*im_h*im_w - view_id*im_h*im_w - pixel_y * im_w );
        size_t idx_K    = batch_id*num_views*9 + view_id*9;
        size_t idx_T    = batch_id*num_views*3 + view_id*3;
        // printf("x,y: %d, %d \n", pixel_x, pixel_y);
        // get the 3D position of the point.
        // int_K @ [y,x,1] * d.
        float cam_pt_x = (cam_intr_inv[0+idx_K]*pixel_x + cam_intr_inv[1+idx_K]*pixel_y + cam_intr_inv[2+idx_K]) * d;
        float cam_pt_y = (cam_intr_inv[3+idx_K]*pixel_x + cam_intr_inv[4+idx_K]*pixel_y + cam_intr_inv[5+idx_K]) * d;
        float cam_pt_z = (cam_intr_inv[6+idx_K]*pixel_x + cam_intr_inv[7+idx_K]*pixel_y + cam_intr_inv[8+idx_K]) * d;
        // printf("cam x,y,z: %f, %f, %f \n", cam_pt_x, cam_pt_y, cam_pt_z);
        // to global position; int_R @ (p - T);
        cam_pt_x -= cam_T[0+idx_T]; cam_pt_y -= cam_T[1+idx_T]; cam_pt_z -= cam_T[2+idx_T];
        float pt_x = cam_R_inv[0+idx_K]*cam_pt_x + cam_R_inv[1+idx_K]*cam_pt_y + cam_R_inv[2+idx_K]*cam_pt_z;
        float pt_y = cam_R_inv[3+idx_K]*cam_pt_x + cam_R_inv[4+idx_K]*cam_pt_y + cam_R_inv[5+idx_K]*cam_pt_z;
        float pt_z = cam_R_inv[6+idx_K]*cam_pt_x + cam_R_inv[7+idx_K]*cam_pt_y + cam_R_inv[8+idx_K]*cam_pt_z;
        // printf("world x,y,z: %f, %f, %f \n", pt_x, pt_y, pt_z);
        // assign the min_max_value to the array.
        int basic_idx_center  = batch_id * 3;
        int basic_idx_num_pts = batch_id * 1;
        atomicAddFloat(vol_center + basic_idx_center+0, pt_x);
        atomicAddFloat(vol_center + basic_idx_center+1, pt_y);
        atomicAddFloat(vol_center + basic_idx_center+2, pt_z);
        atomicAddFloat(vol_num_pts + basic_idx_num_pts+0, 1.0);
    }
}


__global__ void integrate_cuda(float * tsdf_vol,
                               float * weight_vol,
                               float * color_vol,
                               int * occupied_vol, // [B, N**3];
                               int * num_occupied_voxels, // [B], each batch(person) has N occupied nodes.
                            //    int * num_corner_points, // [B] to record the each batch(person) has N corners points.
                               float * vol_bbox, // [B, 3, 2(min,max)]
                               float * cam_intr,
                               float * cam_pose,
                               float * color_im, // [B,N,H,W], the rgb is zipped in an int value.
                               float * depth_im, // [B,N,H,W]
                               int   * mask_im, //  [B,N,H,W]
                               float * voxel_size,
                               const float obs_weight,
                               const int im_h, const int im_w,
                               const int batch_num, const int num_views, const int vol_size,
                               const float trunc_weight,
                               const float tsdf_th_low, const float tsdf_th_high,
                               const int vol_dim_x, const int vol_dim_y, const int vol_dim_z) {
    // size_t num_threads = omp_get_num_procs();
    
    #pragma unroll
    CUDA_KERNEL_LOOP(i, batch_num * vol_size) { // loop 1.
        size_t voxel_id = static_cast<size_t>(i);
        size_t batch_id = static_cast<size_t>(i / vol_size);
        // trunc margin mark the regions of depth insided the image.
        float trunc_margin = voxel_size[batch_id] * trunc_weight; // default as voxel size * 5 as unit. 256 -> 64, 5 / 4 = 1.25;
        
        // Get voxel grid coordinates (note: be careful when casting)
        int v_id_batch = voxel_id - batch_id * vol_size;
        size_t voxel_x = static_cast<size_t>( v_id_batch / (vol_dim_y*vol_dim_z) );
        size_t voxel_y = static_cast<size_t>( (v_id_batch - voxel_x*vol_dim_y*vol_dim_z) / vol_dim_z );
        size_t voxel_z = static_cast<size_t>( v_id_batch - voxel_x*vol_dim_y*vol_dim_z - voxel_y*vol_dim_z );
        // Voxel grid coordinates to world coordinates (original position);
        
        float pt_x = vol_bbox[batch_id*6+0*2+0]+voxel_x*voxel_size[batch_id]; // add min_x
        float pt_y = vol_bbox[batch_id*6+1*2+0]+voxel_y*voxel_size[batch_id]; // add min_y
        float pt_z = vol_bbox[batch_id*6+2*2+0]+voxel_z*voxel_size[batch_id]; // add min_z

        // world coordinates to camera coordinates, (in batch)
        // #pragma omp parallel for num_threads(2 * num_threads - 1)
        // int inside_mask = 0; // judge whether the point is projected in the binary mask.
        #pragma unroll
        for (size_t k=0; k < num_views; k++) {
            size_t idx_RT = batch_id*num_views*12 + k*12;
            size_t idx_K  = batch_id*num_views*9  + k*9;
            // obtain the position in camera' coodinate.
            float cam_pt_x = cam_pose[0+idx_RT]*pt_x+cam_pose[1+idx_RT]*pt_y+cam_pose[2+idx_RT] *pt_z +cam_pose[3+idx_RT];
            float cam_pt_y = cam_pose[4+idx_RT]*pt_x+cam_pose[5+idx_RT]*pt_y+cam_pose[6+idx_RT] *pt_z +cam_pose[7+idx_RT];
            float cam_pt_z = cam_pose[8+idx_RT]*pt_x+cam_pose[9+idx_RT]*pt_y+cam_pose[10+idx_RT]*pt_z +cam_pose[11+idx_RT];
            
            // obtain the pixel coordinate in image space.
            float pixel_x_, pixel_y_, pixel_z_;
            pixel_x_ = cam_intr[0+idx_K] * cam_pt_x + cam_intr[1+idx_K] * cam_pt_y + cam_intr[2+idx_K] * cam_pt_z;
            pixel_y_ = cam_intr[3+idx_K] * cam_pt_x + cam_intr[4+idx_K] * cam_pt_y + cam_intr[5+idx_K] * cam_pt_z;
            pixel_z_ = cam_intr[6+idx_K] * cam_pt_x + cam_intr[7+idx_K] * cam_pt_y + cam_intr[8+idx_K] * cam_pt_z;
            pixel_x_ = pixel_x_ / pixel_z_;
            pixel_y_ = pixel_y_ / pixel_z_;
            int pixel_x = (int) roundf(pixel_x_); int pixel_y = (int) roundf(pixel_y_);

            // Skip if outside view frustum.
            if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h || cam_pt_z < 0)
                continue;

            // obtain the pixel_idx [B, N, H, W];
            size_t pixel_idx = batch_id*num_views*im_w*im_h + k*im_w*im_h + pixel_y*im_w + pixel_x;
            // skip the invalid depth;
            float depth_value = depth_im[pixel_idx];
            if (depth_value < 0.1f || depth_value > 4.0f) continue; // filter out the invalid depth value.
            
            float depth_diff = depth_value - cam_pt_z; // if depth_diff < 0, inside the body.
            if (depth_diff < -trunc_margin) continue;

            // Integrate tsdf volume & weight volume.
            float dist = fmin(1.0f, depth_diff / trunc_margin);
            float w_old = weight_vol[voxel_id];
            float w_new = w_old + obs_weight;
            weight_vol[voxel_id] = w_new;
            tsdf_vol[voxel_id] = (tsdf_vol[voxel_id]*w_old+obs_weight*dist)/w_new;

            // when filtering, if the pixels are inside the masks, inside ++
            // if (mask_im[pixel_idx] == 1) inside_mask += 1;

            // Integrate color's volume;
            float old_color = color_vol[voxel_id];
            float old_b = floorf(old_color/(256*256));
            float old_g = floorf((old_color-old_b*256*256)/256);
            float old_r = old_color-old_b*256*256-old_g*256;
            float new_color = color_im[pixel_idx];
            float new_b = floorf(new_color/(256*256));
            float new_g = floorf((new_color-new_b*256*256)/256);
            float new_r = new_color-new_b*256*256-new_g*256;
            new_b = fmin(roundf((old_b*w_old+obs_weight*new_b)/w_new),255.0f);
            new_g = fmin(roundf((old_g*w_old+obs_weight*new_g)/w_new),255.0f);
            new_r = fmin(roundf((old_r*w_old+obs_weight*new_r)/w_new),255.0f);
            color_vol[voxel_id] = new_b*256*256+new_g*256+new_r;
        }
        // fill the occupied voxels, 0 is the surface planes.
        if (tsdf_vol[voxel_id] < trunc_margin * tsdf_th_high && tsdf_vol[voxel_id] > -trunc_margin * tsdf_th_low) {
            occupied_vol[voxel_id] = v_id_batch;
            atomicAdd(&(num_occupied_voxels[batch_id]), 1);
        }
        else {occupied_vol[voxel_id] = (int)-1;}
    }
}


__global__ void integrate(int * occupied_vol, // [B, N**3];
                          int * num_occupied_voxels, // [B], each batch(person) has N occupied nodes.
                    //    int * num_corner_points, // [B] to record the each batch(person) has N corners points.
                          float * vol_bbox, // [B, 3, 2(min,max)]
                          float * cam_intr_inv,
                          float * cam_R_inv,
                          float * cam_T,
                        //   float * color_im, // [B,N,H,W], the rgb is zipped in an int value.
                          float * depth_im, // [B,N,H,W]
                        //   int   * mask_im, //  [B,N,H,W]
                          float * voxel_size, // [B]
                          const float obs_weight,
                          const int im_h, const int im_w,
                          const int batch_num, const int num_views, const int vol_size,
                          const float tsdf_th_low, const float tsdf_th_high,
                          const int vol_dim_x, const int vol_dim_y, const int vol_dim_z) {
    // the integrate func get the voxel's position for each pixel in the depth maps.
    #pragma unroll
    CUDA_KERNEL_LOOP(i, batch_num * num_views * im_h * im_w) { // loop 1.
        // get the depth values first from the depth maps.
        float d = depth_im[i];
        if (d < 0.1f || d > 4.0f) continue;

        size_t batch_id = static_cast<size_t>( i / (num_views*im_h*im_w) );
        size_t view_id  = static_cast<size_t>( (i - batch_id*num_views*im_h*im_w) / (im_h*im_w) );
        size_t pixel_y  = static_cast<size_t>( (i - batch_id*num_views*im_h*im_w - view_id*im_h*im_w) / im_w);
        size_t pixel_x  = static_cast<size_t>( i - batch_id*num_views*im_h*im_w - view_id*im_h*im_w - pixel_y * im_w );
        size_t idx_K    = batch_id*num_views*9 + view_id*9;
        size_t idx_T    = batch_id*num_views*3 + view_id*3;

        // get the 3D position of the point.
        // int_K @ [y,x,1] * d;
        float cam_pt_x = (cam_intr_inv[0+idx_K]*pixel_x + cam_intr_inv[1+idx_K]*pixel_y + cam_intr_inv[2+idx_K]) * d;
        float cam_pt_y = (cam_intr_inv[3+idx_K]*pixel_x + cam_intr_inv[4+idx_K]*pixel_y + cam_intr_inv[5+idx_K]) * d;
        float cam_pt_z = (cam_intr_inv[6+idx_K]*pixel_x + cam_intr_inv[7+idx_K]*pixel_y + cam_intr_inv[8+idx_K]) * d;
        // printf("cam x,y,z: %f, %f, %f \n", cam_pt_x, cam_pt_y, cam_pt_z);
        // to global position; int_R @ (p - T);
        cam_pt_x -= cam_T[0+idx_T]; cam_pt_y -= cam_T[1+idx_T]; cam_pt_z -= cam_T[2+idx_T];
        float pt_x = cam_R_inv[0+idx_K]*cam_pt_x + cam_R_inv[1+idx_K]*cam_pt_y + cam_R_inv[2+idx_K]*cam_pt_z;
        float pt_y = cam_R_inv[3+idx_K]*cam_pt_x + cam_R_inv[4+idx_K]*cam_pt_y + cam_R_inv[5+idx_K]*cam_pt_z;
        float pt_z = cam_R_inv[6+idx_K]*cam_pt_x + cam_R_inv[7+idx_K]*cam_pt_y + cam_R_inv[8+idx_K]*cam_pt_z;
        // transform the 3d position to xyz index.
        float v_size = voxel_size[batch_id];
        float3 bbox_start_xyz = make_float3(vol_bbox[batch_id*6+0*2+0], 
                                            vol_bbox[batch_id*6+1*2+0], 
                                            vol_bbox[batch_id*6+2*2+0]);
        // printf("x,y,z: %f, %f, %f \n", bbox_start_xyz.x, bbox_start_xyz.y, bbox_start_xyz.z);
        // printf("x_,y_,z_: %f, %f, %f \n", pt_x, pt_y, pt_z);
        // printf("voxel_size: %f \n", v_size);
        // The ix, iy, iz will repeated.
        int ix = (int) (pt_x - bbox_start_xyz.x) / v_size;
        int iy = (int) (pt_y - bbox_start_xyz.y) / v_size;
        int iz = (int) (pt_z - bbox_start_xyz.z) / v_size;
        // printf("ix,iy,iz: %d, %d, %d \n", ix, iy, iz);
        if (ix < 0 || ix >= vol_dim_x || iy < 0 || iy >= vol_dim_y || iz < 0 || iz >= vol_dim_z) continue;
        // fill the occupied voxels.
        size_t voxel_id = static_cast<size_t>(batch_id*vol_size+ix*vol_dim_y*vol_dim_z+iy*vol_dim_z+iz);
        occupied_vol[voxel_id] = ix*vol_dim_y*vol_dim_z+iy*vol_dim_z+iz;
    }
}


__global__ void cal_non_empty_voxels(
    int * occupied_vol, // [B, N**3];
    int * num_occupied_voxels, // [B] for each voxel.
    const int batch_size, const int vol_size
) {
    #pragma unroll
    CUDA_KERNEL_LOOP(i, batch_size * vol_size) {
        int batch_id = (int) i / vol_size;
        int occ_val = occupied_vol[i];

        if (occ_val < 0) continue;
        atomicAdd(&(num_occupied_voxels[batch_id]), 1);
    }
}


__global__ void build_multi_res_voxels(
    const float * tsdf_vol, // the TSDF Volume save the SDF values.
    const float * color_vol, // the color Volume save the color values.
    const int vol_dim, // the vol_dim of the original TSDF volume.
    // const float init_voxel_res, // default as 
    // const int curr_level, // 0,1,2,3,
    const int start_level, // default as 1 (at least 8 voxels).
    const int end_level, // default as 7.
    const int batch_size, 
    const size_t total_voxels_num, // default as N= \sum^7_{i=1} (2**i)**3
    // const float * vol_bbox, // save the bounding boxes.
    // const float * voxel_size, // [B, 1]
    const int output_ft_dim, // e.g, 1, saving the weight; default as 3 
    float * multi_res_voxels // [B, N, 5(v_x, v_y, v_z, scale.)]
) {
    // the multi_res_voxels save the voxel's center position(voxel_x, v_y, v_z) & resolution(len) (3 + 1).
    #pragma unroll
    CUDA_KERNEL_LOOP(k, total_voxels_num * batch_size) {
        size_t batch_id = static_cast<size_t>(k / total_voxels_num);
        size_t voxel_id = k - batch_id * total_voxels_num;
        size_t voxel_id_curr_level = voxel_id;

        // get current level.
        int curr_level = 0, tmp_id = 0;
        for (int l=start_level; l <= end_level; l++) {
            size_t level_voxels_num = pow(2,l) * pow(2,l) * pow(2,l);
            int tmp_voxels_num = tmp_id + level_voxels_num;
            // e.g., idx=75, 75 > 73, 75 < 585, l=3;
            if (voxel_id >= tmp_id && voxel_id < tmp_voxels_num) {
                curr_level = l;
                break;
            }
            tmp_id = tmp_voxels_num;
            voxel_id_curr_level -= level_voxels_num;
        }
        
        // get the current voxel's center position
        size_t curr_vol_dim = pow(2, curr_level); // 2**l, e.g, 1.
        size_t curr_voxel_len = static_cast<size_t>(vol_dim / curr_vol_dim); // e.g., 256 / 2=128;
        size_t curr_voxel_x = static_cast<size_t>( voxel_id_curr_level / (curr_vol_dim*curr_vol_dim) );
        size_t curr_voxel_y = static_cast<size_t>( (voxel_id_curr_level - curr_voxel_x*curr_vol_dim*curr_vol_dim) / curr_vol_dim );
        size_t curr_voxel_z = static_cast<size_t>( voxel_id_curr_level - curr_voxel_x*curr_vol_dim*curr_vol_dim - curr_voxel_y*curr_vol_dim );
        // get the index_range in original tsdf_volume.
        size_t ori_voxel_x_min = curr_voxel_x * curr_voxel_len;
        size_t ori_voxel_y_min = curr_voxel_y * curr_voxel_len;
        size_t ori_voxel_z_min = curr_voxel_z * curr_voxel_len;

        size_t multi_res_basic_id = batch_id * total_voxels_num * output_ft_dim + voxel_id * output_ft_dim;
        // multi_res_voxels[multi_res_basic_id+0] = 
    }
}

__host__ torch::Tensor get_origin_bbox(
    torch::Tensor depth_ims_torch,
    torch::Tensor cam_intr_torch, // [B, N, 3, 3]
    // torch::Tensor cam_intr_inv_torch, // [B, N, 3, 3]
    // torch::Tensor cam_pose_torch, // [B, N, 3, 4]
    torch::Tensor cam_R_torch, // [B, N, 3, 3]
    torch::Tensor cam_T_torch, // [B, N, 3]
    const float th_bbox, //default as 0.01m
    const int device
) {
    cudaSetDevice(device);
    CUDA_CHECK_ERRORS();

    int batch_num = (int) depth_ims_torch.size(0);
    int num_views = (int) depth_ims_torch.size(1);
    int im_h      = (int) depth_ims_torch.size(2);
    int im_w      = (int) depth_ims_torch.size(3);

    float* cam_intr     = cam_intr_torch.contiguous().data_ptr<float>();
    // float* cam_int_inv  = cam_intr_inv_torch.contiguous().data_ptr<float>();
    float* cam_r        = cam_R_torch.contiguous().data_ptr<float>();
    // float* cam_r_inv    = cam_R_inv_torch.contiguous().data_ptr<float>();
    float* cam_t        = cam_T_torch.contiguous().data_ptr<float>();

    torch::Tensor cam_intr_inv_torch = torch::full({batch_num, num_views, 3, 3}, 0, cam_intr_torch.options());
    float* cam_int_inv = cam_intr_inv_torch.contiguous().data_ptr<float>();

    torch::Tensor cam_R_inv_torch = torch::full({batch_num, num_views, 3, 3}, 0, cam_R_torch.options());
    float* cam_r_inv = cam_R_inv_torch.contiguous().data_ptr<float>();

    float* depth_ims    = depth_ims_torch.contiguous().data_ptr<float>();
    
    torch::Tensor vol_origin_torch_min = torch::full({batch_num, 3, 1}, 99999,  cam_intr_torch.options());
    torch::Tensor vol_origin_torch_max = torch::full({batch_num, 3, 1}, -99999, cam_intr_torch.options());
    // assign the max, min;
    torch::Tensor vol_origin_torch = torch::cat({vol_origin_torch_min, vol_origin_torch_max}, 2);
    float* vol_origin = vol_origin_torch.contiguous().data_ptr<float>(); // [B, 3]

    int block_size, grid_size, min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, inverse_mat3x3, 0, 0);
    grid_size = (int) (batch_num * num_views + block_size - 1) / block_size;
    // step0. get inv matrix;
    inverse_mat3x3 <<< grid_size, block_size >>> (
        cam_intr, batch_num, num_views, cam_int_inv
    );

    inverse_mat3x3 <<< grid_size, block_size >>> (
        cam_r, batch_num, num_views, cam_r_inv
    );

    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, obtain_origin_bbox, 0, 0);
    grid_size = (int) (batch_num * num_views * im_h * im_w + block_size - 1) / block_size;
    // step1. obtain the original bbox (min,max);
    obtain_origin_bbox <<< grid_size, block_size >>> (
        depth_ims, cam_int_inv, cam_r_inv, cam_t, 
        (size_t) im_h, (size_t) im_w, 
        batch_num, num_views, vol_origin, th_bbox
    );
    CUDA_CHECK_ERRORS();

    return vol_origin_torch;
}

std::vector<torch::Tensor> get_center_xyz(
    torch::Tensor depth_ims_torch,
    torch::Tensor cam_intr_torch, // [B, N, 3, 3]
    // torch::Tensor cam_intr_inv_torch, // [B, N, 3, 3]
    // torch::Tensor cam_pose_torch, // [B, N, 3, 4]
    torch::Tensor cam_R_torch, // [B, N, 3, 3]
    torch::Tensor cam_T_torch, // [B, N, 3]
    const float th_bbox, //default as 0.01m
    const int device
) {
    cudaSetDevice(device);
    CUDA_CHECK_ERRORS();

    int batch_num = (int) depth_ims_torch.size(0);
    int num_views = (int) depth_ims_torch.size(1);
    int im_h      = (int) depth_ims_torch.size(2);
    int im_w      = (int) depth_ims_torch.size(3);

    float* cam_intr     = cam_intr_torch.contiguous().data_ptr<float>();
    // float* cam_int_inv  = cam_intr_inv_torch.contiguous().data_ptr<float>();
    float* cam_r        = cam_R_torch.contiguous().data_ptr<float>();
    // float* cam_r_inv    = cam_R_inv_torch.contiguous().data_ptr<float>();
    float* cam_t        = cam_T_torch.contiguous().data_ptr<float>();

    torch::Tensor cam_intr_inv_torch = torch::full({batch_num, num_views, 3, 3}, 0, cam_intr_torch.options());
    float* cam_int_inv = cam_intr_inv_torch.contiguous().data_ptr<float>();

    torch::Tensor cam_R_inv_torch = torch::full({batch_num, num_views, 3, 3}, 0, cam_R_torch.options());
    float* cam_r_inv = cam_R_inv_torch.contiguous().data_ptr<float>();

    float* depth_ims    = depth_ims_torch.contiguous().data_ptr<float>();
    
    // record the mean xyz values.
    torch::Tensor vol_num_pts_torch = torch::full({batch_num, 1}, 0.0,  cam_intr_torch.options());
    torch::Tensor vol_center_torch  = torch::full({batch_num, 3}, 0.0,  cam_intr_torch.options());
    float* vol_center  = vol_center_torch.contiguous().data_ptr<float>(); // [B, 3]
    float* vol_num_pts = vol_num_pts_torch.contiguous().data_ptr<float>(); // [B, 1]

    int block_size, grid_size, min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, inverse_mat3x3, 0, 0);
    grid_size = (int) (batch_num * num_views + block_size - 1) / block_size;
    // step0. get inv matrix;
    inverse_mat3x3 <<< grid_size, block_size >>> (
        cam_intr, batch_num, num_views, cam_int_inv
    );

    inverse_mat3x3 <<< grid_size, block_size >>> (
        cam_r, batch_num, num_views, cam_r_inv
    );

    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, obtain_center_points, 0, 0);
    grid_size = (int) (batch_num * num_views * im_h * im_w + block_size - 1) / block_size;
    // step1. obtain the original bbox (min,max);
    obtain_center_points <<< grid_size, block_size >>> (
        depth_ims, cam_int_inv, cam_r_inv, cam_t, 
        (size_t) im_h, (size_t) im_w, 
        batch_num, num_views, vol_center, vol_num_pts
    );
    
    CUDA_CHECK_ERRORS();

    return {vol_center_torch, vol_num_pts_torch};
}

__host__ 
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
fusion_cuda_integrate(
    // torch::Tensor tsdf_vol_torch,  // [B, H, W, K]
    // torch::Tensor weight_vol_torch,  // [B, H, W, K]
    // torch::Tensor color_vol_torch, // [B, H, W, K]
    torch::Tensor occupied_vol_torch, // [B, H,W,K]
    torch::Tensor num_occ_voxel_torch, // [B]
    // torch::Tensor vol_origin_torch, // [B, 3, 2]
    // torch::Tensor vol_bbox_torch, // [B, 3, 2]
    // torch::Tensor voxel_size_torch, // [B, 1]
    torch::Tensor cam_intr_torch, // [B, N, 3, 3]
    // torch::Tensor cam_intr_inv_torch, // [B, N, 3, 3]
    torch::Tensor cam_pose_torch, // [B, N, 3, 4]
    torch::Tensor cam_R_torch, // [B, N, 3, 3]
    torch::Tensor cam_T_torch, // [B, N, 3]
    torch::Tensor color_ims_torch, // [B, N, H, W]
    torch::Tensor depth_ims_torch,  // [B, N, H, W]
    torch::Tensor mask_ims_torch, // [B, N, H, W]
    const float obs_weight, 
    const float th_bbox, //default as 0.01m
    const float trunc_weight, // default as 5.0
    const float tsdf_th_low, const float tsdf_th_high, // default as 6;
    const int device
) {
    cudaSetDevice(device);
    CUDA_CHECK_ERRORS();
    
    int batch_num = (int) color_ims_torch.size(0);
    int num_views = (int) color_ims_torch.size(1);
    // all x-y-z volume's size.
    int vol_dim_x = (int) occupied_vol_torch.size(1);
    int vol_dim_y = (int) occupied_vol_torch.size(2);
    int vol_dim_z = (int) occupied_vol_torch.size(3);
    int im_h      = (int) color_ims_torch.size(2);
    int im_w      = (int) color_ims_torch.size(3);
    int vol_size  = vol_dim_x * vol_dim_y * vol_dim_z;
    // assert(vol_dim_x == vol_dim_y && vol_dim_y == vol_dim_z);
    // torch::Tensor cam_R = cam_pose_torch.slice(3, 0, 3); // [B,N,3,3]

    // output buffer (TSDF_Volume, Weight_Volume, Color_Volume).
    int  * occupied_vol = occupied_vol_torch.contiguous().data_ptr<int>();
    int * num_occ_voxel = num_occ_voxel_torch.contiguous().data_ptr<int>();
    // input buffer.
    float* cam_intr     = cam_intr_torch.contiguous().data_ptr<float>();
    // float* cam_int_inv  = cam_intr_inv_torch.contiguous().data_ptr<float>();
    float* cam_pose     = cam_pose_torch.contiguous().data_ptr<float>();
    float* cam_r        = cam_R_torch.contiguous().data_ptr<float>();
    // float* cam_r_inv    = cam_R_inv_torch.contiguous().data_ptr<float>();
    float* cam_t        = cam_T_torch.contiguous().data_ptr<float>();
    float* color_ims    = color_ims_torch.contiguous().data_ptr<float>();
    float* depth_ims    = depth_ims_torch.contiguous().data_ptr<float>();
    int * mask_ims      = mask_ims_torch.contiguous().data_ptr<int>();

    /* init variables */
    torch::Tensor vol_origin_torch_min = torch::full({batch_num, 3, 1}, 99999,  cam_intr_torch.options());
    torch::Tensor vol_origin_torch_max = torch::full({batch_num, 3, 1}, -99999, cam_intr_torch.options());
    // vol_origin_torch_min = vol_origin_torch_min * 0 + MAX_VALUE;
    // vol_origin_torch_max = vol_origin_torch_max * 0 + MIN_VALUE;
    torch::Tensor vol_bbox_torch   = torch::empty({batch_num, 3, 2}, cam_intr_torch.options());
    torch::Tensor voxel_size_torch = torch::empty({batch_num, 1}, cam_intr_torch.options());
    // assign the max, min;
    torch::Tensor vol_origin_torch = torch::cat({vol_origin_torch_min, vol_origin_torch_max}, 2);

    float* vol_origin = vol_origin_torch.contiguous().data_ptr<float>(); // [B, 3]
    float* vol_bbox   = vol_bbox_torch.contiguous().data_ptr<float>(); // [B, 3]
    float* voxel_size = voxel_size_torch.contiguous().data_ptr<float>(); // [B,1]

    torch::Tensor tsdf_vol_torch   = torch::full({batch_num, vol_dim_x, vol_dim_y, vol_dim_z}, 1, cam_intr_torch.options());
    torch::Tensor weight_vol_torch = torch::full({batch_num, vol_dim_x, vol_dim_y, vol_dim_z}, 0, cam_intr_torch.options());
    torch::Tensor color_vol_torch  = torch::full({batch_num, vol_dim_x, vol_dim_y, vol_dim_z}, 0, cam_intr_torch.options());

    float * tsdf_vol   = tsdf_vol_torch.contiguous().data_ptr<float>();
    float * weight_vol = weight_vol_torch.contiguous().data_ptr<float>();
    float * color_vol  = color_vol_torch.contiguous().data_ptr<float>();

    torch::Tensor cam_intr_inv_torch = torch::full({batch_num, num_views, 3, 3}, 0, cam_intr_torch.options());
    float* cam_int_inv = cam_intr_inv_torch.contiguous().data_ptr<float>();
    
    torch::Tensor cam_R_inv_torch = torch::full({batch_num, num_views, 3, 3}, 0, cam_R_torch.options());
    float* cam_r_inv = cam_R_inv_torch.contiguous().data_ptr<float>();

    int block_size, grid_size, min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, inverse_mat3x3, 0, 0);
    grid_size = (int) (batch_num * num_views + block_size - 1) / block_size;
    // step0. get inv matrix;
    inverse_mat3x3 <<< grid_size, block_size >>> (
        cam_intr, batch_num, num_views, cam_int_inv
    );

    inverse_mat3x3 <<< grid_size, block_size >>> (
        cam_r, batch_num, num_views, cam_r_inv
    );

    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, obtain_origin_bbox, 0, 0);
    grid_size = (int) (batch_num * num_views * im_h * im_w + block_size - 1) / block_size;
    // step1. obtain the original bbox (min,max);
    obtain_origin_bbox <<< grid_size, block_size >>> (
        depth_ims, cam_int_inv, cam_r_inv, cam_t, 
        (size_t) im_h, (size_t) im_w, 
        batch_num, num_views, vol_origin, th_bbox
    );

    // step2. obtain the resized bbox (min, max), object resolution no change;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, obtain_resized_bbox, 0, 0);
    grid_size = (int) (batch_num + block_size - 1) / block_size;
    obtain_resized_bbox <<< grid_size, block_size >>> (
        vol_origin, batch_num, vol_dim_x, vol_bbox, voxel_size
    );

    // step3. tsdf-fusion.
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, integrate_cuda, 0, 0);
    grid_size = (int) (batch_num * vol_size + block_size - 1) / block_size; //parallel in (batch num * num_views * vol_size)
    integrate_cuda <<< grid_size, block_size >>> (
        tsdf_vol, weight_vol, color_vol, occupied_vol, num_occ_voxel, 
        vol_bbox, cam_intr, cam_pose,
        color_ims, depth_ims, mask_ims,
        voxel_size, obs_weight, im_h, im_w,
        batch_num, num_views, vol_size, trunc_weight,
        tsdf_th_low, tsdf_th_high,
        vol_dim_x, vol_dim_y, vol_dim_z
    );

    CUDA_CHECK_ERRORS();
    
    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>{tsdf_vol_torch, weight_vol_torch, 
           color_vol_torch, vol_origin_torch, vol_bbox_torch, voxel_size_torch};
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
fusion_cuda_integrate_refined(
    torch::Tensor occupied_vol_torch, // [B, H,W,K]
    torch::Tensor num_occ_voxel_torch, // [B]
    // torch::Tensor vol_origin_torch, // [B, 3, 2]
    // torch::Tensor vol_bbox_torch, // [B, 3, 2]
    // torch::Tensor voxel_size_torch, // [B, 1]
    torch::Tensor cam_intr_torch, // [B, N, 3, 3]
    // torch::Tensor cam_intr_inv_torch, // [B, N, 3, 3]
    torch::Tensor cam_pose_torch, // [B, N, 3, 4]
    torch::Tensor cam_R_torch, // [B, N, 3, 3]
    torch::Tensor cam_T_torch, // [B, N, 3]
    // torch::Tensor color_ims_torch, // [B, N, H, W]
    torch::Tensor depth_ims_torch,  // [B, N, H, W]
    // torch::Tensor mask_ims_torch, // [B, N, H, W]
    const float obs_weight, 
    const float th_bbox, //default as 0.01m
    const float tsdf_th_low, const float tsdf_th_high, // default as 6;
    const int device
) {
    cudaSetDevice(device);
    CUDA_CHECK_ERRORS();
    
    int batch_num = (int) depth_ims_torch.size(0);
    int num_views = (int) depth_ims_torch.size(1);
    // all x-y-z volume's size.
    int vol_dim_x = (int) occupied_vol_torch.size(1);
    int vol_dim_y = (int) occupied_vol_torch.size(2);
    int vol_dim_z = (int) occupied_vol_torch.size(3);
    int im_h      = (int) depth_ims_torch.size(2);
    int im_w      = (int) depth_ims_torch.size(3);
    int vol_size  = vol_dim_x * vol_dim_y * vol_dim_z;
    // assert(vol_dim_x == vol_dim_y && vol_dim_y == vol_dim_z);
    // torch::Tensor cam_R = cam_pose_torch.slice(3, 0, 3); // [B,N,3,3]

    // output buffer (TSDF_Volume, Weight_Volume, Color_Volume).
    int  * occupied_vol = occupied_vol_torch.contiguous().data_ptr<int>();
    int * num_occ_voxel = num_occ_voxel_torch.contiguous().data_ptr<int>();
    // input buffer.
    float* cam_intr     = cam_intr_torch.contiguous().data_ptr<float>();
    // float* cam_int_inv  = cam_intr_inv_torch.contiguous().data_ptr<float>();
    float* cam_pose     = cam_pose_torch.contiguous().data_ptr<float>();
    float* cam_r        = cam_R_torch.contiguous().data_ptr<float>();
    // float* cam_r_inv    = cam_R_inv_torch.contiguous().data_ptr<float>();
    float* cam_t        = cam_T_torch.contiguous().data_ptr<float>();
    // float* color_ims    = color_ims_torch.contiguous().data_ptr<float>();
    float* depth_ims    = depth_ims_torch.contiguous().data_ptr<float>();
    // int * mask_ims      = mask_ims_torch.contiguous().data_ptr<int>();

    /* init variables */
    torch::Tensor vol_origin_torch_min = torch::full({batch_num, 3, 1}, 99999,  cam_intr_torch.options());
    torch::Tensor vol_origin_torch_max = torch::full({batch_num, 3, 1}, -99999, cam_intr_torch.options());
    // vol_origin_torch_min = vol_origin_torch_min * 0 + MAX_VALUE;
    // vol_origin_torch_max = vol_origin_torch_max * 0 + MIN_VALUE;
    torch::Tensor vol_bbox_torch   = torch::empty({batch_num, 3, 2}, cam_intr_torch.options());
    torch::Tensor voxel_size_torch = torch::empty({batch_num, 1}, cam_intr_torch.options());
    // assign the max, min;
    torch::Tensor vol_origin_torch = torch::cat({vol_origin_torch_min, vol_origin_torch_max}, 2);

    float* vol_origin = vol_origin_torch.contiguous().data_ptr<float>(); // [B, 3]
    float* vol_bbox   = vol_bbox_torch.contiguous().data_ptr<float>(); // [B, 3]
    float* voxel_size = voxel_size_torch.contiguous().data_ptr<float>(); // [B,1]

    torch::Tensor cam_intr_inv_torch = torch::full({batch_num, num_views, 3, 3}, 0, cam_intr_torch.options());
    float* cam_int_inv = cam_intr_inv_torch.contiguous().data_ptr<float>();
    
    torch::Tensor cam_R_inv_torch = torch::full({batch_num, num_views, 3, 3}, 0, cam_R_torch.options());
    float* cam_r_inv = cam_R_inv_torch.contiguous().data_ptr<float>();


    int block_size, grid_size, min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, inverse_mat3x3, 0, 0);
    grid_size = (int) (batch_num * num_views + block_size - 1) / block_size;
    // step0. get inv matrix;
    inverse_mat3x3 <<< grid_size, block_size >>> (
        cam_intr, batch_num, num_views, cam_int_inv
    );

    inverse_mat3x3 <<< grid_size, block_size >>> (
        cam_r, batch_num, num_views, cam_r_inv
    );

    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, obtain_origin_bbox, 0, 0);
    grid_size = (int) (batch_num * num_views * im_h * im_w + block_size - 1) / block_size;
    // step1. obtain the original bbox (min,max), this func can obtain the 
    // 3d position for each pixel in the image, and update the resized vol-box's corner positions.
    // -> vol_origin [B, 3], the started position.
    obtain_origin_bbox <<< grid_size, block_size >>> (
        depth_ims, cam_int_inv, cam_r_inv, cam_t, 
        (size_t) im_h, (size_t) im_w, 
        batch_num, num_views, vol_origin, th_bbox
    );

    // step2. obtain the resized bbox (min, max), object resolution no change;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, obtain_resized_bbox, 0, 0);
    grid_size = (int) (batch_num + block_size - 1) / block_size;
    obtain_resized_bbox <<< grid_size, block_size >>> (
        vol_origin, batch_num, vol_dim_x, vol_bbox, voxel_size
    );

    // step3. directly set the occupied values with 0 or 1, the pixels num is B*N_views*H*W
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, integrate, 0, 0);
    grid_size = (int) (batch_num * num_views * im_h * im_w + block_size - 1) / block_size;
    // update the occupied volume with 0,1;
    integrate <<< grid_size, block_size >>> (
        occupied_vol, num_occ_voxel, 
        vol_bbox, cam_int_inv, cam_r_inv, cam_t,
        depth_ims,
        voxel_size, obs_weight, im_h, im_w,
        batch_num, num_views, vol_size,
        tsdf_th_low, tsdf_th_high,
        vol_dim_x, vol_dim_y, vol_dim_z
    );

    // step 4. get the non-empty num of the bbox, atomic adding.
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, cal_non_empty_voxels, 0, 0);
    grid_size = (int) (batch_num * vol_size + block_size - 1) / block_size;
    cal_non_empty_voxels <<< grid_size, block_size >>> (
        occupied_vol, num_occ_voxel, batch_num, vol_size
    );

    CUDA_CHECK_ERRORS();   
    // only return the box's information.
    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>{vol_origin_torch, vol_bbox_torch, voxel_size_torch};
}