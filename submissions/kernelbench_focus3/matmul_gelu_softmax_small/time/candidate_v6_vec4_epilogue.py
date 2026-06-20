import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


_CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor matmul_gelu_softmax_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias
);
"""


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {

__device__ __forceinline__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    unsigned mask = 0xffffffffu;
    value = fmaxf(value, __shfl_down_sync(mask, value, 16));
    value = fmaxf(value, __shfl_down_sync(mask, value, 8));
    value = fmaxf(value, __shfl_down_sync(mask, value, 4));
    value = fmaxf(value, __shfl_down_sync(mask, value, 2));
    value = fmaxf(value, __shfl_down_sync(mask, value, 1));
    return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
    unsigned mask = 0xffffffffu;
    value += __shfl_down_sync(mask, value, 16);
    value += __shfl_down_sync(mask, value, 8);
    value += __shfl_down_sync(mask, value, 4);
    value += __shfl_down_sync(mask, value, 2);
    value += __shfl_down_sync(mask, value, 1);
    return value;
}

__global__ void add_bias_gelu_softmax_rows_vec4_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int cols
) {
    __shared__ float warp_partials[8];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    const float* in_row = logits + row * cols;
    float* out_row = out + row * cols;

    float values[4];
    float local_max = -INFINITY;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int col = tid + i * blockDim.x;
        float value = -INFINITY;
        if (col < cols) {
            value = gelu_exact(in_row[col] + bias[col]);
        }
        values[i] = value;
        local_max = fmaxf(local_max, value);
    }

    float warp_max = warp_reduce_max(local_max);
    if (lane == 0) {
        warp_partials[warp] = warp_max;
    }
    __syncthreads();

    float block_max = -INFINITY;
    if (warp == 0) {
        block_max = (lane < (blockDim.x >> 5)) ? warp_partials[lane] : -INFINITY;
        block_max = warp_reduce_max(block_max);
        if (lane == 0) {
            warp_partials[0] = block_max;
        }
    }
    __syncthreads();
    block_max = warp_partials[0];

    float exps[4];
    float local_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int col = tid + i * blockDim.x;
        float ex = 0.0f;
        if (col < cols) {
            ex = expf(values[i] - block_max);
        }
        exps[i] = ex;
        local_sum += ex;
    }

    float warp_sum = warp_reduce_sum(local_sum);
    if (lane == 0) {
        warp_partials[warp] = warp_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp == 0) {
        block_sum = (lane < (blockDim.x >> 5)) ? warp_partials[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            warp_partials[0] = block_sum;
        }
    }
    __syncthreads();
    float inv_sum = 1.0f / warp_partials[0];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int col = tid + i * blockDim.x;
        if (col < cols) {
            out_row[col] = exps[i] * inv_sum;
        }
    }
}

cublasHandle_t get_handle(cudaStream_t stream) {
    static thread_local cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    }
    cublasSetStream(handle, stream);
    return handle;
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
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    int batch = static_cast<int>(x.size(0));
    int in_features = static_cast<int>(x.size(1));
    int out_features = static_cast<int>(weight.size(0));
    TORCH_CHECK(weight.size(1) == in_features, "weight shape mismatch");
    TORCH_CHECK(bias.size(0) == out_features, "bias shape mismatch");
    TORCH_CHECK(out_features <= 1024, "softmax kernel supports up to 1024 columns");

    auto logits = torch::empty({batch, out_features}, x.options());
    auto out = torch::empty_like(logits);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle = get_handle(stream);
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        out_features,
        batch,
        in_features,
        &alpha,
        weight.data_ptr<float>(),
        in_features,
        x.data_ptr<float>(),
        in_features,
        &beta,
        logits.data_ptr<float>(),
        out_features
    );
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasSgemm failed");

    add_bias_gelu_softmax_rows_vec4_kernel<<<batch, 256, 0, stream>>>(
        logits.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        out_features
    );
    return out;
}
"""


_ext = load_inline(
    name="kb_focus3_matmul_gelu_softmax_small_v6",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["matmul_gelu_softmax_forward"],
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["-lcublas"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return _ext.matmul_gelu_softmax_forward(x, self.linear.weight, self.linear.bias)
