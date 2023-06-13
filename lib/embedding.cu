#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <math.h>
#include <stdio.h>
#include "embedding.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <torch/extension.h>


__global__ void embedding_kernel(const int hn, const int hs,
                    const int *id, half *q, half *k, 
                    const half *cos, const half *sin){

  int sid = blockIdx.x / hn; 
  int hnum = blockIdx.x % hn;
  int hid = threadIdx.x;
  int block_offset = sid * (hn * hs) + hnum * hs;

  int eid = __ldg(&id[sid]);
  half cos_pos = __ldg(&cos[eid * hs + hid]);
  half sin_pos = __ldg(&sin[eid * hs + hid]);
  half q_data = q[block_offset + hid];
  half q_rotate = hid < (hs / 2) ? 
    __hneg(q[block_offset + hid + (hs / 2)]) :
    q[block_offset + hid - (hs / 2)];
  half k_data = k[block_offset + hid];
  half k_rotate = hid < (hs / 2) ? 
    __hneg(k[block_offset + hid + (hs / 2)]) :
    k[block_offset + hid - (hs / 2)];
    
  __syncthreads();

  q[block_offset + hid] = 
    __hadd(__hmul(q_data, cos_pos), __hmul(q_rotate, sin_pos));
  k[block_offset + hid] = 
    __hadd(__hmul(k_data, cos_pos), __hmul(k_rotate, sin_pos));
}

void embedding_apply_wrapper(at::Tensor q, at::Tensor k, 
                const at::Tensor cos, const at::Tensor sin, 
                const at::Tensor id){
  
  int seq_len = id.size(0);
  int bs = id.size(1);
  // 32
  int hn = q.size(2);
  // 64
  int hs = cos.size(2);

  half *q_ptr = reinterpret_cast<half *>(q.data_ptr<at::Half>());
  half *k_ptr = reinterpret_cast<half *>(k.data_ptr<at::Half>());
  half *cos_ptr = reinterpret_cast<half *>(cos.data_ptr<at::Half>());
  half *sin_ptr = reinterpret_cast<half *>(sin.data_ptr<at::Half>());
  int *id_ptr = id.data_ptr<int>();

  embedding_kernel<<<dim3(seq_len * hn), dim3(hs)>>>
    (hn, hs, id_ptr, q_ptr, k_ptr, cos_ptr, sin_ptr);
}
