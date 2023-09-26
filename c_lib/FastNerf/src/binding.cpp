#include "../include/infer_mlp.h"

#include <pybind11/pybind11.h>
namespace py=pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("infer_nerf", &infer_fast_rendering);
    m.def("infer_blending", &infer_upsampling);
    m.def("infer_density", &infer_density_mlp);
}