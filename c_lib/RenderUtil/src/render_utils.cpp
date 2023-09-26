/*
    A simple renderer to render meshes in different views.
*/

// Standard libs
#include <string>
#include <cstdio>
#include <cmath>
#include <vector>
#include <iostream>

// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
#include "cuda_helper.h"

void render_tex_mesh(const at::Tensor vertices, // [B, N_v, 3]
    const at::Tensor faces, // [B, N_f, 3]
    const at::Tensor normals, // [B, M, 3, 3]
    const at::Tensor uvs, const at::Tensor textures,
    const at::Tensor Ks, const at::Tensor RTs,
    const int H, const int W,
    const float ambients, const float light_stren, 
    const at::Tensor view_dirs, const at::Tensor light_dirs, const bool calc_spec, const bool is_ortho, // [B, 3]
    at::Tensor depths, at::Tensor RGBs, at::Tensor masks // depths: [B, H, W]
);

void interface_render(const at::Tensor vertices, // [B, N_v, 3]
    const at::Tensor faces, // [B, N_f, 3]
    const at::Tensor normals, 
    const at::Tensor uvs, const at::Tensor textures,
    const at::Tensor Ks, const at::Tensor RTs, // [B, 3, 3], [B, 3, 4]
    const int H, const int W,
    const float ambients, const float light_stren, 
    const at::Tensor view_dirs, const at::Tensor light_dirs, const bool calc_spec, const bool is_ortho, // [B, 3]
    at::Tensor depths, at::Tensor RGBs, at::Tensor masks // depths: [B, H, W]
) {
    CHECK_INPUT(vertices); CHECK_INPUT(faces);
    CHECK_INPUT(normals); CHECK_INPUT(uvs); CHECK_INPUT(textures);
    CHECK_INPUT(Ks); CHECK_INPUT(RTs);
    CHECK_INPUT(view_dirs); CHECK_INPUT(light_dirs);
    render_tex_mesh(vertices, faces, normals, 
                    uvs, textures, 
                    Ks, RTs, 
                    H, W, ambients, light_stren,
                    view_dirs, light_dirs, calc_spec, is_ortho,
                    depths, RGBs, masks);
}

// call in dataloader for generate different views' images realtime.
PYBIND11_MODULE(RenderUtils, m) {
    // m.def("render_tex_mesh",  &render_tex_mesh, "A function that render texed meshes");
    m.def("render_mesh", &interface_render, "Rendering meshes.");
}