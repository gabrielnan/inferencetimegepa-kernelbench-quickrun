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

__global__ void add_bias_gelu_softmax_rows_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int cols
) {
    extern __shared__ float scratch[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* row_logits = logits + row * cols;
    float* row_out = out + row * cols;

    float value = -INFINITY;
    if (tid < cols) {
        value = gelu_exact(row_logits[tid] + bias[tid]);
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

    float numerator = 0.0f;
    if (tid < cols) {
        numerator = expf(value - row_max);
    }
    scratch[tid] = numerator;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    if (tid < cols) {
        row_out[tid] = numerator / scratch[0];
    }
}

int next_power_of_two(int value) {
    int power = 1;
    while (power < value) {
        power <<= 1;
    }
    return power;
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

    int threads = next_power_of_two(out_features);
    add_bias_gelu_softmax_rows_kernel<<<batch, threads, threads * sizeof(float), stream>>>(
        logits.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        out_features
    );
    return out;
}
"""


_ext = load_inline(
    name="kb_focus3_matmul_gelu_softmax_small_v8_negative_control",
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
