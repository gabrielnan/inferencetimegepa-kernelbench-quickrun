import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


_CPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/ops/linear.h>

torch::Tensor matmul_gelu_softmax_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias
);
"""


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/linear.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {

__device__ __forceinline__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

__global__ void gelu_softmax_rows_kernel(
    const float* __restrict__ logits,
    float* __restrict__ out,
    int cols
) {
    extern __shared__ float scratch[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* in_row = logits + row * cols;
    float* out_row = out + row * cols;

    float value = -INFINITY;
    if (tid < cols) {
        value = gelu_exact(in_row[tid]);
    }
    scratch[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = scratch[tid + stride];
            scratch[tid] = scratch[tid] > other ? scratch[tid] : other;
        }
        __syncthreads();
    }
    float row_max = scratch[0];

    float ex = 0.0f;
    if (tid < cols) {
        ex = expf(value - row_max);
    }
    scratch[tid] = ex;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / scratch[0];
    if (tid < cols) {
        out_row[tid] = ex * inv_sum;
    }
}

int next_power_of_two(int value) {
    int power = 1;
    while (power < value) {
        power <<= 1;
    }
    return power;
}

}  // namespace

torch::Tensor matmul_gelu_softmax_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");

    auto logits = at::linear(x, weight, bias);
    int rows = static_cast<int>(logits.size(0));
    int cols = static_cast<int>(logits.size(1));
    TORCH_CHECK(cols <= 1024, "softmax kernel supports up to 1024 columns");

    auto out = torch::empty_like(logits);
    int threads = next_power_of_two(cols);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gelu_softmax_rows_kernel<<<rows, threads, threads * sizeof(float), stream>>>(
        logits.data_ptr<float>(),
        out.data_ptr<float>(),
        cols
    );
    return out;
}
"""


_ext = load_inline(
    name="kb_focus3_matmul_gelu_softmax_small_v2",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["matmul_gelu_softmax_forward"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return _ext.matmul_gelu_softmax_forward(x, self.linear.weight, self.linear.bias)
