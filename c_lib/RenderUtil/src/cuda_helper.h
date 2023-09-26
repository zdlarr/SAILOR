#pragma once
#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <torch/torch.h>
#include <torch/extension.h>
// #include "cuda.h"
// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"
#include <ATen/ATen.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>

using namespace std;

#define CHECK_INPUT(x) \
  TORCH_CHECK(x.is_cuda(), #x "input must be a cuda tensor"); \
  TORCH_CHECK(x.is_contiguous(), #x "input must be contiguous.")

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x "input must be a cuda tensor.")

// #define CHECK_ERRORS(x) checkCudaErrors(x)

#define CUDA_CHECK_ERRORS()                                                    \
  do {                                                                         \
    cudaError_t err = cudaGetLastError();                                      \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",           \
              cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__,          \
              __FILE__);                                                       \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)
       
#endif // CUDA_HELPER_H