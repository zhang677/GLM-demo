#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <math.h>
#include <stdio.h>
#include "embedding.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <torch/extension.h>

#define DIV_UP(x, y) ((x) + (y) - 1) / (y) 

__global__ void gelu_kernel(half *x){

  int id = blockIdx.x * blockDim.x + threadIdx.x;
  float tmp = __half2float(x[id]);
  x[id] = __float2half(0.5 * tmp * (1.0 + tanhf(0.7978845608028654 * tmp *
                                       (1.0 + 0.044715 * tmp * tmp))));
}

void gelu_apply_wrapper(at::Tensor x){

  int seq_len = x.size(0);
  int bs = x.size(1);
  int hdim = x.size(2);

  half *x_ptr = reinterpret_cast<half *>(x.data_ptr<at::Half>());

  gelu_kernel<<<dim3(DIV_UP(seq_len * bs * hdim, 128)), dim3(128)>>>(x_ptr);
}