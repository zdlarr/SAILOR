/*
    Utils for rendering images in different views.
    reference to StereoPIFu cpp codes.
    Input: 
    1. vertices, B x N_v x 3.
    2. faces, B x N_f x 3.
    3. normals, B x N_f x 3 x 3. 
    4. uvs: B x N_f x 3 x 2.
    5. textures: B x H x W x 3
    6. Ks: B x 3 x 3, RTs: B x 3 x 4
    7. ambient: B x 3, light_dirs: B x 3
    9. H, W is obtained from K matrix.
*/

#include <math.h>
#include <vector>
#include <stdint.h>

#include "cuda_helper.h"
#include <string>
#include <stdio.h>
#include <iostream>

using namespace std;


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

__device__ float atomicCAS_f32(float *p, float cmp, float val) {
	return __int_as_float(atomicCAS((int *) p, __float_as_int(cmp), __float_as_int(val)));
}

__device__ void project_pts(
    const float * vert, // 3 * 1 vec
    const float *K, const float *RT, const bool is_ortho,
    float * grid, int * status) // 3 * 1,
{
    const int W = static_cast<int>(K[2] * 2);
    const int H = static_cast<int>(K[5] * 2);
    // [xyz] = R @ X + T
    float x = vert[0], y = vert[1], z = vert[2];
    float x_trans = x * RT[0] + y * RT[1] + z * RT[2] + RT[3];
    float y_trans = x * RT[4] + y * RT[5] + z * RT[6] + RT[7];
    float z_trans = x * RT[8] + y * RT[9] + z * RT[10] + RT[11];
    
    // K * [xyz]_trans. without divided by z_trans -> ortho prespective.
    float pixel_x, pixel_y, pixel_x_, pixel_y_, pixel_z_;
    if (is_ortho) {
        // when ortho, directly apply the f_x * x + d_x; f_y * y + d_y;
        pixel_x = K[0] * x_trans + K[2];
        pixel_y = K[4] * y_trans + K[5];
    } else { // when persp, apply dividing, i.e., divide by z_trans;
        // pixel_x = K[0] * x_trans / z_trans + K[2];
        // pixel_y = K[4] * y_trans / z_trans + K[5];
        pixel_x_ = K[0] * x_trans + K[1] * y_trans + K[2] * z_trans;
        pixel_y_ = K[3] * x_trans + K[4] * y_trans + K[5] * z_trans;
        pixel_z_ = K[6] * x_trans + K[7] * y_trans + K[8] * z_trans;
        pixel_x = pixel_x_ / pixel_z_;
        pixel_y = pixel_y_ / pixel_z_;
    }
    
    grid[0] = pixel_x; grid[1] = pixel_y; grid[2] = z_trans;
    if (pixel_x >= 0 && pixel_x < W && pixel_y >= 0 && pixel_y < H)
        *status = 1;
    return;
}

__global__ void project_vts_kernel(
    const int num_verts, const int batch_num,
    const float *vertices,
    const float *Ks, const float * RTs, const bool is_ortho,
    float *grids, int * status )
{
    // vertices: [B, N, 3]. Ks: [B, 3, 3], RTs: [B, 3, 4], grids: [B, N, 3] (xy,d)
    // status: [B, N]
    #pragma unroll
    CUDA_KERNEL_LOOP(i, num_verts * batch_num) {
        size_t t_id = i * 3; // [B*N, 3] each vertice contains 3 values.
        size_t batch_id = static_cast<size_t>(i / num_verts); 
        // calculate the projected results.
        project_pts(vertices + t_id, 
                    Ks + batch_id * 9, RTs + batch_id * 12, is_ortho,
                    grids + t_id, status + i);
    }
}

template<typename scalar_t>
__device__ __inline__ void normalize_vec3(scalar_t * vector) {
    scalar_t l = sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]) + 1e-6;
    vector[0] /= l;
    vector[1] /= l;
    vector[2] /= l;
}   


template<typename scalar_t>
__device__ __inline__ void cross_vec3(const scalar_t *v0, const scalar_t *v1, const scalar_t *v2, scalar_t *normal) {
    // calculate the crossed vector of the given vertices.
    scalar_t x0 = v1[0] - v0[0],
             y0 = v1[1] - v0[1],
             z0 = v1[2] - v0[2];

    scalar_t x1 = v2[0] - v1[0],
             y1 = v2[1] - v1[1],
             z1 = v2[2] - v1[2];
    
    normal[0] = y0 * z1 - z0 * y1;
    normal[1] = z0 * x1 - x0 * z1;
    normal[2] = x0 * y1 - y0 * x1;
    normalize_vec3(normal);
}

template<typename scalar_t>
__device__ __inline__ scalar_t dot_vec3(const scalar_t *v0, const scalar_t *v1) {
    return v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
}

template<typename scalar_t>
__device__ scalar_t product(const scalar_t *p0, const scalar_t *p1, const scalar_t *p2) {
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
}

template<typename scalar_t>
__device__ __inline__ bool is_clock_wise(const scalar_t * pa, const scalar_t * pb, const scalar_t * pc)
{
    // judge if the triangle is clock-wise. if ((p2.x - p1.x) *(p3.y - p1.y) -(p3.x - p1.x) *(p2.y-p1.y) < 0)
    if ( (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pc[0] - pa[0]) * (pb[1] - pa[1]) < 0 ) {
        return false;
    }
    return true;
}

template<typename scalar_t>
__device__ __inline__ bool is_in_tri(const scalar_t * query_pos, 
                          const scalar_t * pa, const scalar_t * pb, const scalar_t * pc,
                          scalar_t * coord)
{  
    // in 2D space.
    // given the query point, judge if the query pos in located in triangle.
    // suppose that the points p0, p1, p2 follow the inverse order, else adjust.
    // if (product(p0, p1, p2) < 0) return is_in_tri(query_pos,p0,p2,p1,coord);
    if (product(pa, pb, pc) == 0) return false;

    // scalar_t x0 = p1[0] - p0[0],
    //          y0 = p1[1] - p0[1],
    //          x1 = p2[0] - p0[0],
    //          y1 = p2[1] - p0[1],
    //          x  = query_pos[0] - p0[0],
    //          y  = query_pos[1] - p0[1];

    // scalar_t a = (x * y0 - x0 * y) / (x1 * y0 - x0 * y1),
    //          b = (x * y1 - x1 * y) / (x0 * y1 - x1 * y0);

    // weighted triangle inteploate.
    scalar_t a = ((query_pos[1] -  pb[1]) * (pc[0] - pb[0]) - (query_pos[0] - pb[0]) * (pc[1] - pb[1])) \
               / ((pa[1] - pb[1]) * (pc[0] - pb[0]) - (pa[0] - pb[0]) * (pc[1] - pb[1]));
    scalar_t b = ((query_pos[1] - pc[1]) * (pa[0] - pc[0]) - (query_pos[0] - pc[0]) * (pa[1] - pc[1])) \
               / ((pb[1] - pc[1]) * (pa[0] - pc[0]) - (pb[0] - pc[0]) * (pa[1] - pc[1]));

    coord[0] = a;
    coord[1] = b;
    coord[2] = scalar_t(1) - a - b;

    // if ( product(p0, p1, query_pos) > 0 && product(p1, p2, query_pos) > 0 && product(p2, p0, query_pos) > 0 )    
    //     return true;

    // return false;
    if (a >= 0 && b >= 0 && (a+b) <= 1) return true;
    return false;
}



__global__ void depth_test(
    const float * verts, // the 3d points of each verts. [B, N, 3]
    const int * faces, // the indexes of faces. [B, M, 3]
    const float * grids, // projected x & y & depth value. 
    const int * status, // record the status that if the points locate in triangle.
    const int verts_num, const int faces_num, const int batch_num,
    const int H, const int W,
    float * depths)
{
    #pragma unroll 
    CUDA_KERNEL_LOOP(i, faces_num * batch_num) {
        size_t t_id = i * 3;
        size_t batch_id = static_cast<size_t>(i / faces_num);
        // get the idx of each vert in the triangle.
        size_t v_idx0 = faces[t_id] + batch_id * verts_num, 
               v_idx1 = faces[t_id + 1] + batch_id * verts_num, 
               v_idx2 = faces[t_id + 2] + batch_id * verts_num;
        
        // when verts_idx == -1, the triangle mesh is invalid.
        if (v_idx0 < 0 || v_idx1 < 0 || v_idx2 < 0) continue;

        const float* v0 = verts + v_idx0 * 3,
                   * v1 = verts + v_idx1 * 3,
                   * v2 = verts + v_idx2 * 3;
        
        const float* g0 = grids + v_idx0 * 3,
                   * g1 = grids + v_idx1 * 3,
                   * g2 = grids + v_idx2 * 3;
        
        if( is_clock_wise(g0, g1, g2) ) continue; // when ccw, continue.

        // all vertice belong to a triangle should locate in image space.
        // if (! (status[v_idx0] && status[v_idx1] && status[v_idx2]))
        //     continue;
        
        // printf("verts: %f, %f, %f \n", v0[0], v0[1], v0[2]);
        // get the bbox of the projected triangles.
        float x_min = g0[0] < g1[0] ? g0[0] : g1[0],
              x_max = g0[0] > g1[0] ? g0[0] : g1[0],
              y_min = g0[1] < g1[1] ? g0[1] : g1[1],
              y_max = g0[1] > g1[1] ? g0[1] : g1[1];
              
        x_min = x_min < g2[0] ? x_min : g2[0];
        x_max = x_max > g2[0] ? x_max : g2[0];
        y_min = y_min < g2[1] ? y_min : g2[1];
        y_max = y_max > g2[1] ? y_max : g2[1];
        x_min = ceilf(fmaxf(0, x_min));
        y_min = ceilf(fmaxf(0, y_min));
        x_max = floorf(fminf(W - 1, x_max));
        y_max = floorf(fminf(H - 1, y_max));

        float query_pos[2], coord[3];

        for (int x = x_min; x <= x_max; x++) {
            for (int y = y_min; y <= y_max; y++) {
                // assign a new query point.
                query_pos[0] = x; query_pos[1] = y;
                if( is_in_tri(query_pos, g0, g1, g2, coord) ) { // if inside the triangle.
                    // printf("coord: %f, %f, %f \n", coord[0], coord[1], coord[2]);
                    size_t pixel_idx = batch_id * H * W + y * W + x;
                    // inteplate a depth value for the query point.
                    float z_value = g0[2] * coord[0] + g1[2] * coord[1] + g2[2] * coord[2];
                    // float pre_depth = depths[pixel_idx];
                    atomicMin(depths + pixel_idx, z_value);
                }

            }
        }

    }
}


// phong_color model: 
// ambient + diffuse_coef * light_dir * albedo + spec_coef * light_color * albedo

template<typename scalar_t>
__device__ void calc_ambient(const scalar_t * albedo, const scalar_t ambient, scalar_t * color) {
    color[0] = albedo[0] * ambient;
    color[1] = albedo[1] * ambient;
    color[2] = albedo[2] * ambient;
}

template<typename scalar_t>
__device__ scalar_t calc_diffuse_coef(const scalar_t * normal, const scalar_t * light_dir) {
    scalar_t cos_v = dot_vec3(normal, light_dir);
    cos_v = cos_v >= scalar_t(0) ? cos_v : scalar_t(0);
    cos_v = cos_v <= scalar_t(1) ? cos_v : scalar_t(1);
    return cos_v;
}

template<typename scalar_t>
__device__ scalar_t calc_specular_coef(const scalar_t * normal, const scalar_t * light_dir, const scalar_t *view_dir) {
    // reflect func: (-light_dir, normal)
    // reflection dir: -light_dir + 2 * N * N.dot(light_dir), update reflection dir.
    scalar_t n_dot_ldir = dot_vec3(normal, light_dir);
    scalar_t reflect_dir[3];
    reflect_dir[0] = -light_dir[0] + 2 * normal[0] * n_dot_ldir;
    reflect_dir[1] = -light_dir[1] + 2 * normal[1] * n_dot_ldir;
    reflect_dir[2] = -light_dir[2] + 2 * normal[2] * n_dot_ldir;

    scalar_t v_dot_r = dot_vec3(view_dir, reflect_dir);
    scalar_t spec = v_dot_r >= scalar_t(0) ? v_dot_r : scalar_t(0);
    spec = spec <= scalar_t(1) ? spec : scalar_t(1);
    spec = static_cast<scalar_t>(powf(float(spec), float(60)));
    return spec;
}


template<typename scalar_t>
__device__ __inline__ void calc_brdf(const scalar_t * albedo, // the albedo color.
                          const scalar_t * normal, const scalar_t *light_dir, const scalar_t * view_dir, // properties of the verts.
                          const scalar_t ambient, const scalar_t light_stren,                          
                          scalar_t * brdf, const bool calc_spec) {
    scalar_t ambient_albedo[3];
    calc_ambient(albedo, ambient, ambient_albedo);
    scalar_t diffuse_coef = calc_diffuse_coef(normal, light_dir),
             specular_coef = calc_specular_coef(normal, light_dir, view_dir);
    
    brdf[0] = ambient_albedo[0] + albedo[0] * light_stren * diffuse_coef;
    brdf[1] = ambient_albedo[1] + albedo[1] * light_stren * diffuse_coef;
    brdf[2] = ambient_albedo[2] + albedo[2] * light_stren * diffuse_coef;

    if (calc_spec) {
        brdf[0] += albedo[0] * light_stren * specular_coef;
        brdf[1] += albedo[1] * light_stren * specular_coef;
        brdf[2] += albedo[2] * light_stren * specular_coef;
    }

    brdf[0] = static_cast<scalar_t>(fmaxf(0, fminf(1, float(brdf[0]))));
    brdf[1] = static_cast<scalar_t>(fmaxf(0, fminf(1, float(brdf[1]))));
    brdf[2] = static_cast<scalar_t>(fmaxf(0, fminf(1, float(brdf[2]))));
}


// main render.
__global__ void render(
    const float * verts, // the 3d points of each verts. [B, N, 3]
    const int * faces, // the indexes of faces. [B, M, 3]
    const float* normals, // [B, M, 3, 3] normal vector for each faces.
    const float* uvs, // [B, M, 3, 2] uv coord.
    const float* textures, // [B, th, tw, 3] uv maps of the image.
    const float * grids, // projected x & y & depth value. 
    const int * status, // record the status that if the points locate in triangle.
    const int verts_num, const int faces_num, const int batch_num,
    const int H, const int W, const int H_T, const int W_T,
    const float ambient, const float light_stren, // the amients lights' strength and the light_directions.
    const float* view_dirs, const float* light_dirs, const bool calc_spec, // view_dir:[B,3], light_dir: [B, 3]
    const float* depths, int * masks, float * RGBs)
{
    #pragma unroll
    CUDA_KERNEL_LOOP(i, faces_num * batch_num) {
        size_t t_id = i * 3;
        size_t batch_id = static_cast<size_t>(i / faces_num);
        // get verts' idxs.
        size_t v_idx0 = faces[t_id] + batch_id * verts_num, 
               v_idx1 = faces[t_id + 1] + batch_id * verts_num, 
               v_idx2 = faces[t_id + 2] + batch_id * verts_num;

        if (v_idx0 < 0 || v_idx1 < 0 || v_idx2 < 0) continue;
            
        const float* g0 = grids + v_idx0 * 3,
                   * g1 = grids + v_idx1 * 3,
                   * g2 = grids + v_idx2 * 3,
                   * f_uv = uvs + i * 6,
                   * f_normal = normals + i * 9,
                   * light_dir = light_dirs + batch_id * 3,
                   * view_dir  = view_dirs + batch_id * 3;

        float tex_tri_area = ((f_uv[4] - f_uv[0]) * (f_uv[3] - f_uv[1]) - (f_uv[5] - f_uv[1]) * (f_uv[2] - f_uv[0])) * H_T * W_T;
        float rgb_tri_area = ((g2[0] - g0[0]) * (g1[1] - g0[1]) - (g2[1] - g0[1]) * (g1[0] - g0[0]));
        int filter_border_len = floorf(sqrt(abs(tex_tri_area / rgb_tri_area)) * 0.5);
        
        // judge whether the interploated depth is located in the triangle depth.
        float x_min = ceilf(fmaxf(fminf(g0[0], fminf(g1[0], g2[0])), 0)),
              x_max = floorf(fminf(fmaxf(g0[0], fmaxf(g1[0], g2[0])), W - 2)),
              y_min = ceilf(fmaxf(fminf(g0[1], fminf(g1[1], g2[1])), 0)),
              y_max = floorf(fminf(fmaxf(g0[1], fmaxf(g1[1], g2[1])), H - 2));
        float query_pos[2], coord[3], p_uv[2], p_normal[3], p_albedo[3], p_color[3];
        
        for (int x = x_min; x <= x_max; x++) {
            for (int y = y_min; y <= y_max; y++) {
                // assign a new query point.
                query_pos[0] = x; query_pos[1] = y;
                if( is_in_tri(query_pos, g0, g1, g2, coord) ) { // if inside the triangle.
                    size_t pixel_idx = batch_id * H * W + y * W + x;
                    // inteplate a depth value for the query point.
                    float z_value = g0[2] * coord[0] + g1[2] * coord[1] + g2[2] * coord[2],
                          z_mem   = depths[pixel_idx];
                    if (abs(z_value - z_mem) < 1e-5) { // the pixel's color is interploated by the triangle.
                        // interpolate the uv values.
                        p_uv[0] = coord[0] * f_uv[0] + coord[1] * f_uv[2] + coord[2] * f_uv[4];
                        p_uv[1] = coord[0] * f_uv[1] + coord[1] * f_uv[3] + coord[2] * f_uv[5];

                        // interpolate the normals.
                        p_normal[0] = coord[0] * f_normal[0] + coord[1] * f_normal[3] + coord[2] * f_normal[6];
                        p_normal[1] = coord[0] * f_normal[1] + coord[1] * f_normal[4] + coord[2] * f_normal[7];
                        p_normal[2] = coord[0] * f_normal[2] + coord[1] * f_normal[5] + coord[2] * f_normal[8];
                        normalize_vec3(p_normal);

                        // float cos_v = dot_vec3(p_normal, light_dir);
                        // cos_v = fmaxf(0.0f, cos_v);
                        // float scale = (ambient + light_stren * cos_v);

                        int tex_r = (H_T - 1) * (1.0 - p_uv[1]),
                            tex_c = (W_T - 1) * p_uv[0];
                        
                        // sampling uv mapping.
                        float r_ = 0.0f, g_ = 0.0f, b_ = 0.0f, cnt = 0.0f;
                        if (filter_border_len == 0) {
                            const float * tex_cptr  = textures + 3 * (batch_id * H_T * W_T + tex_r * W_T + tex_c);
                            r_ = tex_cptr[0]; g_ = tex_cptr[1]; b_ = tex_cptr[2];
                        }
                        else {                                
                            int start_x = fmaxf(0, tex_c - filter_border_len);
                            int end_x   = fminf(W_T, tex_c + filter_border_len + 1);
                            int start_y = fmaxf(0, tex_r - filter_border_len);
                            int end_y   = fminf(H_T, tex_r + filter_border_len + 1);
                            
                            for (int tx = start_x; tx < end_x; tx++) {
                                for (int ty = start_y; ty < end_y; ty++) {
                                    const float * tex_cptr  = textures + 3 * (batch_id * H_T * W_T + ty * W_T + tx);
                                    r_ += tex_cptr[0];
                                    g_ += tex_cptr[1];
                                    b_ += tex_cptr[2];
                                    cnt ++;
                                }
                            }
                            
                            r_ *= (1  / cnt);
                            g_ *= (1  / cnt);
                            b_ *= (1  / cnt);
                        }
                        // calculate the color of vertice.
                        p_albedo[0] = r_; p_albedo[1] = g_; p_albedo[2] = b_;
                        calc_brdf(p_albedo, p_normal, light_dir, view_dir, ambient, light_stren, p_color, calc_spec);

                        // update RGBs.
                        atomicAdd(RGBs + 4 * pixel_idx + 0, p_color[0]);
                        atomicAdd(RGBs + 4 * pixel_idx + 1, p_color[1]);
                        atomicAdd(RGBs + 4 * pixel_idx + 2, p_color[2]);
                        atomicAdd(RGBs + 4 * pixel_idx + 3, 1.0f);
                        // update masks
                        atomicOr(masks + pixel_idx, int(1));
                    }
                    
                }
            }
        }
        
    }
}


__host__ void render_tex_mesh(
    const at::Tensor vertices, // [B, N_v, 3]
    const at::Tensor faces, // [B, N_f, 3]
    const at::Tensor normals, // [B, M, 3, 3]
    const at::Tensor uvs, // [B, M, 3, 2] uv coord.
    const at::Tensor textures, // [B, th, tw, 3] uv maps of the image.
    const at::Tensor Ks, const at::Tensor RTs,
    const int H, const int W,
    const float ambient, const float light_stren, 
    const at::Tensor view_dirs, const at::Tensor light_dirs, const bool calc_spec, const bool is_ortho,// [B, 3]
    at::Tensor depths, at::Tensor RGBs, at::Tensor masks) // depths: [B, H, W]
{
    const int batch_size = vertices.size(0);
    const int verts_num  = vertices.size(1);
    const int faces_num  = faces.size(1);
    const int H_T = textures.size(1),
              W_T = textures.size(2);
    
    // input buffer.
    const float* verts_buffer = vertices.contiguous().data_ptr<float>();
    const int* faces_buffer = faces.contiguous().data_ptr<int>();
    const float* uvs_buffer = uvs.contiguous().data_ptr<float>();
    const float* texs_buffer = textures.contiguous().data_ptr<float>();
    const float* Ks_buffer = Ks.contiguous().data_ptr<float>();
    const float* RTs_buffer = RTs.contiguous().data_ptr<float>();
    const float* normals_buffer = normals.contiguous().data_ptr<float>();
    const float* ld_buffer = light_dirs.contiguous().data_ptr<float>();
    const float* vd_buffer = view_dirs.contiguous().data_ptr<float>();

    // output buffer.
    // float* grids_buffer = grids.contiguous().data_ptr<float>();
    float* depths_buffer = depths.contiguous().data_ptr<float>();
    int* masks_buffer = masks.contiguous().data_ptr<int>();
    float* RGBs_buffer  = RGBs.contiguous().data_ptr<float>();
    
    int n_status, n_grids, *status_buffer;
    float * grids_buffer;
    n_status = sizeof(int) * (batch_size * verts_num);
    n_grids = sizeof(float) * (batch_size * verts_num * 3);
    // allocate buffer for status
    cudaMallocManaged((void**)&status_buffer, n_status);
    cudaMallocManaged((void**)&grids_buffer, n_grids);

    // allocate grid, block sizes.
    int block_size, grid_size, min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, project_vts_kernel, 0, 0);
    grid_size = (verts_num * batch_size + block_size - 1) / block_size;  // process the projections.

    project_vts_kernel <<< grid_size, block_size >>> (
        verts_num, batch_size,
        verts_buffer, Ks_buffer, RTs_buffer, is_ortho,
        grids_buffer, status_buffer
    );

    // reallocate memory.
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, depth_test, 0, 0);
    grid_size = (faces_num * batch_size + block_size - 1) / block_size;
    depth_test <<< grid_size, block_size >>> (
        verts_buffer, faces_buffer, grids_buffer,
        status_buffer, verts_num, faces_num, batch_size, 
        H, W,
        depths_buffer
    );

    // render.
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, render, 0, 0);
    grid_size = (faces_num * batch_size + block_size - 1) / block_size;
    render <<< grid_size, block_size >>> (
        verts_buffer, faces_buffer, normals_buffer, uvs_buffer, texs_buffer, grids_buffer,
        status_buffer, verts_num, faces_num, batch_size, 
        H, W, H_T, W_T, ambient, light_stren, 
        vd_buffer, ld_buffer, calc_spec,
        depths_buffer, masks_buffer, RGBs_buffer
    );
    cudaFree(status_buffer);
    cudaFree(grids_buffer);

    CUDA_CHECK_ERRORS();
    return;
}


// __host__ void render_tex_mesh(
//     const at::Tensor vertices,
//     const at::Tensor faces,
//     const at::Tensor normals,
//     const at::Tensor uvs,
//     const at::Tensor texures,
//     const at::Tensor Ks, const at::Tensor RTs, // Ks: [B, 3, 3], RTs: [B, 3, 4]
//     const int H, const int W,    
//     const at::Tensor ambients, const at::Tensor light_dirs,
//     at::Tensor depths, at::Tensor rgbs, at::Tensor masks) {
    
// }