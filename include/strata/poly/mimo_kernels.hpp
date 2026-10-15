#pragma once
#include <cstdint>

namespace strata::poly {
// Fused rows [Q, K, V]. Partial NeoX RoPE rotates the first rot channels of Q/K;
// V is unrotated. Caches use absolute positions, FP16, with distinct K/V widths.
void mimo_qkv(const float* fused, float* q, uint16_t* keys, uint16_t* vals,
              int heads, int kv, int kd, int vd, int rot, float base,
              int64_t pos, int rows, void* stream);
// Causal GQA. window=0 means full attention; otherwise includes exactly window
// keys including the current token. A sink contributes to the softmax denominator
// but has no value vector. Full caches deliberately avoid ring/checkpoint aliasing.
void mimo_attend(const float* q, const uint16_t* keys, const uint16_t* vals,
                 const float* sinks, float* out, int heads, int kv, int kd, int vd,
                 int window, int64_t pos, int rows, void* stream);
}
