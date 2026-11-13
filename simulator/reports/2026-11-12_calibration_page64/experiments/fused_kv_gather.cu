#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

// Pure bit copies. No floating-point arithmetic or attention approximations.
template<int PS>
__global__ void gather_kernel(
    const int* pages, const int* permutation,
    const uint4* pk, const uint4* pv, const uint4* hk, const uint4* hv,
    uint4* outk, uint4* outv, int plen, int capacity, int start,
    int slots, int page_stride_b, int page_stride_h) {
  const int bh = blockIdx.y;
  const int b = bh / 8;
  const int h = bh % 8;
  const int slot = blockIdx.x * 16 + threadIdx.x / 16;
  const int vector = threadIdx.x % 16;  // Eight BF16 elements per vector.
  if (slot >= slots) return;
  const int page = pages[b * page_stride_b + h * page_stride_h + slot / PS];
  const int physical = max(0, page * PS + slot % PS);
  const long long dst = ((long long)bh * slots + slot) * 16 + vector;
  if (physical < plen) {
    const int original = permutation[((long long)b * 8 + h) * plen + physical];
    const long long src = ((long long)original * 8 + h) * 16 + vector;
    outk[dst] = pk[src];
    outv[dst] = pv[src];
  } else {
    const int generated = min(capacity - 1, max(0, physical - plen));
    const long long src = (((long long)(b + start) * 8 + h) * capacity + generated) * 16 + vector;
    outk[dst] = hk[src];
    outv[dst] = hv[src];
  }
}

std::vector<torch::Tensor> gather(
    torch::Tensor pages, torch::Tensor permutation,
    torch::Tensor pk, torch::Tensor pv, torch::Tensor hk, torch::Tensor hv,
    int64_t start, int64_t page_size) {
  TORCH_CHECK(pages.is_cuda() && permutation.is_cuda() && pk.is_cuda() && hk.is_cuda());
  TORCH_CHECK(pages.scalar_type() == torch::kInt32 && permutation.scalar_type() == torch::kInt32);
  TORCH_CHECK(pk.scalar_type() == torch::kBFloat16 && pv.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(hk.scalar_type() == torch::kBFloat16 && hv.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(pk.is_contiguous() && pv.is_contiguous() && hk.is_contiguous() && hv.is_contiguous());
  TORCH_CHECK(permutation.is_contiguous() && pages.stride(2) == 1);
  TORCH_CHECK(pages.size(0) > 0 && pages.size(1) == 8 && pk.size(1) == 8 && pk.size(2) == 128);
  TORCH_CHECK(hk.size(0) >= start + pages.size(0) && hk.size(1) == 8 && hk.size(3) == 128);
  TORCH_CHECK(start >= 0);
  TORCH_CHECK(page_size == 32 || page_size == 64);
  const int slots = pages.size(2) * page_size;
  const int plen = pk.size(0);
  const int capacity = hk.size(2);
  TORCH_CHECK(permutation.dim() == 3 && permutation.size(0) == pages.size(0) && permutation.size(1) == 8 && permutation.size(2) == plen);
  auto outk = torch::empty({pages.size(0), 8, slots, 128}, pk.options());
  auto outv = torch::empty_like(outk);
  dim3 grid((slots + 15) / 16, pages.size(0) * 8);
  auto stream = at::cuda::getCurrentCUDAStream();
  #define LAUNCH(PS) gather_kernel<PS><<<grid, 256, 0, stream>>>( \
    pages.data_ptr<int>(), permutation.data_ptr<int>(), \
    reinterpret_cast<const uint4*>(pk.data_ptr()), reinterpret_cast<const uint4*>(pv.data_ptr()), \
    reinterpret_cast<const uint4*>(hk.data_ptr()), reinterpret_cast<const uint4*>(hv.data_ptr()), \
    reinterpret_cast<uint4*>(outk.data_ptr()), reinterpret_cast<uint4*>(outv.data_ptr()), \
    plen, capacity, start, slots, pages.stride(0), pages.stride(1))
  if (page_size == 32) { LAUNCH(32); } else { LAUNCH(64); }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {outk, outv};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gather", &gather); }
