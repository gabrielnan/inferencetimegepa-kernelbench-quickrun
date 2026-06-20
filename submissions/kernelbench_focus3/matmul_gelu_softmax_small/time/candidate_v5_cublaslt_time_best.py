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
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {

__device__ __forceinline__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

__global__ void gelu_softmax_rows_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
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
        value = gelu_exact(in_row[tid] + bias[tid]);
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
    float inv_sum = 1.0f / scratch[0];
    if (tid < cols) {
        out_row[tid] = numerator * inv_sum;
    }
}

int next_power_of_two(int value) {
    int power = 1;
    while (power < value) {
        power <<= 1;
    }
    return power;
}

const char* cublaslt_status_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "CUBLAS_STATUS_UNKNOWN";
    }
}

#define CUBLASLT_CHECK(expr) do { \
    cublasStatus_t _status = (expr); \
    TORCH_CHECK(_status == CUBLAS_STATUS_SUCCESS, #expr " failed: ", cublaslt_status_string(_status)); \
} while (0)

cublasLtHandle_t get_lt_handle() {
    static thread_local cublasLtHandle_t handle = nullptr;
    if (handle == nullptr) {
        CUBLASLT_CHECK(cublasLtCreate(&handle));
    }
    return handle;
}

void set_row_major(cublasLtMatrixLayout_t layout) {
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        layout,
        CUBLASLT_MATRIX_LAYOUT_ORDER,
        &order,
        sizeof(order)
    ));
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
    TORCH_CHECK(x.dim() == 2, "x must be rank 2");
    TORCH_CHECK(weight.dim() == 2, "weight must be rank 2");
    TORCH_CHECK(bias.dim() == 1, "bias must be rank 1");

    int batch = static_cast<int>(x.size(0));
    int in_features = static_cast<int>(x.size(1));
    int out_features = static_cast<int>(weight.size(0));
    TORCH_CHECK(weight.size(1) == in_features, "weight shape mismatch");
    TORCH_CHECK(bias.size(0) == out_features, "bias shape mismatch");
    TORCH_CHECK(out_features <= 1024, "softmax kernel supports up to 1024 columns");

    auto logits = torch::empty({batch, out_features}, x.options());
    auto out = torch::empty_like(logits);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasLtHandle_t handle = get_lt_handle();

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatrixLayout_t d_desc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;

    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_T;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        op_desc,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &transa,
        sizeof(transa)
    ));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        op_desc,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &transb,
        sizeof(transb)
    ));

    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_32F, batch, in_features, in_features));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_32F, out_features, in_features, in_features));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, batch, out_features, out_features));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_32F, batch, out_features, out_features));
    set_row_major(a_desc);
    set_row_major(b_desc);
    set_row_major(c_desc);
    set_row_major(d_desc);

    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    size_t workspace_size = 0;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(workspace_size)
    ));

    cublasLtMatmulHeuristicResult_t heuristic = {};
    int returned_results = 0;
    CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        handle,
        op_desc,
        a_desc,
        b_desc,
        c_desc,
        d_desc,
        preference,
        1,
        &heuristic,
        &returned_results
    ));
    TORCH_CHECK(returned_results > 0, "cuBLASLt returned no matmul heuristic");

    float alpha = 1.0f;
    float beta = 0.0f;
    CUBLASLT_CHECK(cublasLtMatmul(
        handle,
        op_desc,
        &alpha,
        x.data_ptr<float>(),
        a_desc,
        weight.data_ptr<float>(),
        b_desc,
        &beta,
        logits.data_ptr<float>(),
        c_desc,
        logits.data_ptr<float>(),
        d_desc,
        &heuristic.algo,
        nullptr,
        0,
        stream
    ));

    CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(d_desc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
    CUBLASLT_CHECK(cublasLtMatmulDescDestroy(op_desc));

    int threads = next_power_of_two(out_features);
    gelu_softmax_rows_kernel<<<batch, threads, threads * sizeof(float), stream>>>(
        logits.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        out_features
    );
    return out;
}
"""


_ext = load_inline(
    name="kb_focus3_matmul_gelu_softmax_small_v5",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["matmul_gelu_softmax_forward"],
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["-lcublasLt", "-lcublas"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return _ext.matmul_gelu_softmax_forward(x, self.linear.weight, self.linear.bias)
