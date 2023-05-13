#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "embedding.h"
#include "gelu.h" 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("gelu_cuda", &gelu_apply_wrapper);
  m.def("embedding_cuda", &embedding_apply_wrapper);
}