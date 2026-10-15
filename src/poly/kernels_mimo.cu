#include "strata/poly/mimo_kernels.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::poly {
namespace {
constexpr int threads = 256;
__device__ float rotary(const float* x, int d, int rot, float base, int64_t pos) {
    if (d >= rot) return x[d];
    const int i = d % (rot / 2);
    const float angle = float(pos) * powf(base, -2.f * i / rot);
    float s, c;
    sincosf(angle, &s, &c);
    return d < rot / 2 ? x[d] * c - x[d + rot / 2] * s
                       : x[d] * c + x[d - rot / 2] * s;
}
__global__ void qkv_kernel(const float* fused, float* q, half* keys, half* vals,
                           int heads, int kv, int kd, int vd, int rot, float base, int64_t pos) {
    const int row = blockIdx.y, h = blockIdx.x, d = threadIdx.x;
    const int width = heads * kd + kv * (kd + vd);
    const float* x = fused + size_t(row) * width;
    if (h < heads && d < kd)
        q[(size_t(row) * heads + h) * kd + d] = rotary(x + h * kd, d, rot, base, pos + row);
    if (h < kv) {
        if (d < kd)
            keys[(size_t(pos + row) * kv + h) * kd + d] =
                __float2half(rotary(x + heads * kd + h * kd, d, rot, base, pos + row));
        if (d < vd)
            vals[(size_t(pos + row) * kv + h) * vd + d] =
                __float2half(x[heads * kd + kv * kd + h * vd + d]);
    }
}
// Baseline attention, bounded to 8K context. One block owns a query head and its
// score row; there is no shared scratch between tokens or calls.
__global__ void attend_kernel(const float* q, const half* keys, const half* vals,
                              const float* sinks, float* out, int heads, int kv, int kd, int vd,
                              int window, int64_t pos) {
    const int h = blockIdx.x, row = blockIdx.y, lane = threadIdx.x;
    const int kh = h / (heads / kv);
    const int end = int(pos) + row + 1;
    const int begin = window && end > window ? end - window : 0;
    const int count = end - begin;
    extern __shared__ float score[];
    __shared__ float reduce[threads];
    const float* query = q + (size_t(row) * heads + h) * kd;
    float mx = sinks ? sinks[h] : -INFINITY;
    for (int t = lane; t < count; t += threads) {
        const half* k = keys + (size_t(begin + t) * kv + kh) * kd;
        float dot = 0;
        for (int d = 0; d < kd; ++d) dot += query[d] * __half2float(k[d]);
        score[t] = dot * rsqrtf(float(kd));
        mx = fmaxf(mx, score[t]);
    }
    reduce[lane] = mx;
    __syncthreads();
    for (int step = threads / 2; step; step /= 2) {
        if (lane < step) reduce[lane] = fmaxf(reduce[lane], reduce[lane + step]);
        __syncthreads();
    }
    mx = reduce[0];
    float sum = lane == 0 && sinks ? expf(sinks[h] - mx) : 0.f;
    for (int t = lane; t < count; t += threads) {
        score[t] = expf(score[t] - mx);
        sum += score[t];
    }
    // All lanes must read the old reduction before its storage is reused.
    __syncthreads();
    reduce[lane] = sum;
    __syncthreads();
    for (int step = threads / 2; step; step /= 2) {
        if (lane < step) reduce[lane] += reduce[lane + step];
        __syncthreads();
    }
    if (lane < vd) {
        float value = 0;
        for (int t = 0; t < count; ++t)
            value += score[t] * __half2float(vals[(size_t(begin + t) * kv + kh) * vd + lane]);
        out[(size_t(row) * heads + h) * vd + lane] = value / reduce[0];
    }
}
void check(const char* name) {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(std::string(name) + ": " + cudaGetErrorString(error));
}
void validate(int heads, int kv, int kd, int vd, int64_t pos, int rows) {
    if (heads <= 0 || kv <= 0 || heads % kv || kd <= 0 || kd > threads || vd <= 0 || vd > threads ||
        rows <= 0 || pos < 0 || pos > 8192 - rows)
        throw std::invalid_argument("MiMo attention: invalid geometry or context exceeds 8192");
}
}
void mimo_qkv(const float* fused, float* q, uint16_t* keys, uint16_t* vals,
              int heads, int kv, int kd, int vd, int rot, float base, int64_t pos, int rows, void* stream) {
    validate(heads, kv, kd, vd, pos, rows);
    if (rot <= 0 || rot > kd || rot % 2 || !std::isfinite(base) || base <= 0)
        throw std::invalid_argument("MiMo attention: invalid rotary geometry");
    qkv_kernel<<<dim3(heads, rows), threads, 0, (cudaStream_t) stream>>>(
        fused, q, (half*) keys, (half*) vals, heads, kv, kd, vd, rot, base, pos);
    check("mimo_qkv");
}
void mimo_attend(const float* q, const uint16_t* keys, const uint16_t* vals, const float* sinks, float* out,
                 int heads, int kv, int kd, int vd, int window, int64_t pos, int rows, void* stream) {
    validate(heads, kv, kd, vd, pos, rows);
    if (window < 0) throw std::invalid_argument("MiMo attention: negative window");
    const int count = window && window < pos + rows ? window : int(pos + rows);
    attend_kernel<<<dim3(heads, rows), threads, size_t(count) * sizeof(float), (cudaStream_t) stream>>>(
        q, (const half*) keys, (const half*) vals, sinks, out, heads, kv, kd, vd, window, pos);
    check("mimo_attend");
}
}
