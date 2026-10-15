#include "mimo2.hpp"
#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/poly/mimo_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {
void need(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(std::string("unsupported MiMo2 model: ") + why);
}
int per_layer(const MetaValue& v, int il) {
    if (v.type != MetaType::ARRAY) return (int) v.u;
    need(size_t(il) < v.items.size(), "per-layer metadata ends before the trunk");
    return (int) v.items[size_t(il)].u;
}
}
void Mimo2::load_geometry() {
    need(opt_.draft_model.empty(), "MTP/DFlash is not implemented; omit --draft-model");
    need(opt_.max_context > 0 && opt_.max_context <= 8192, "initial context limit is 8192 tokens");
    opt_.draft = false;
    load_shape();
    auto required = [&](const char* key) -> const MetaValue& {
        const auto* v = meta_in(*gg_, key);
        need(v != nullptr, key);
        return *v;
    };
    heads_ = per_layer(required("attention.head_count"), 0);
    kd_ = shape_i("attention.key_length");
    vd_ = shape_i("attention.value_length");
    rot_ = shape_i("rope.dimension_count");
    window_ = shape_i("attention.sliding_window");
    base_ = shape_f("rope.freq_base");
    const auto* swa_base = meta_in(*gg_, "rope.freq_base_swa");
    base_swa_ = swa_base ? float(swa_base->f) : 10000.f;
    if (const auto* v = meta_in(*gg_, "attention.value_scale")) value_scale_ = float(v->f);
    if (const auto* v = meta_in(*gg_, "expert_weights_scale")) weight_scale_ = float(v->f);
    if (const auto* v = meta_in(*gg_, "expert_gating_func")) need(v->u == 2, "router is not sigmoid");
    if (const auto* v = meta_in(*gg_, "expert_shared_count")) need(v->u == 0, "shared experts");
    if (const auto* v = meta_in(*gg_, "expert_group_count")) need(v->u == 1, "grouped expert routing");
    if (const auto* v = meta_in(*gg_, "expert_group_used_count")) need(v->u == 1, "grouped expert routing");
    if (const auto* v = meta_in(*gg_, "rope.scaling.type"))
        need(v->s == "none", "scaled RoPE is not implemented");
    need(heads_ > 0 && heads_ <= 256 && kd_ > 0 && kd_ <= 256 && vd_ > 0 && vd_ <= 256,
         "invalid attention widths");
    need(rot_ > 0 && rot_ <= kd_ && rot_ % 2 == 0 && window_ > 0, "invalid rotary or sliding window");
    need(std::isfinite(base_) && base_ > 0 && std::isfinite(base_swa_) && base_swa_ > 0 &&
         std::isfinite(value_scale_) && value_scale_ > 0 && std::isfinite(weight_scale_) && weight_scale_ > 0,
         "invalid attention or expert scale");
    kv_.resize(n_layer_);
    swa_.resize(n_layer_);
    const auto& kv = required("attention.head_count_kv");
    const auto& pattern = required("attention.sliding_window_pattern");
    for (int il = 0; il < n_layer_; ++il) {
        need(per_layer(required("attention.head_count"), il) == heads_, "varying query head counts");
        kv_[il] = per_layer(kv, il);
        swa_[il] = per_layer(pattern, il);
        need(kv_[il] > 0 && heads_ % kv_[il] == 0 && (swa_[il] == 0 || swa_[il] == 1),
             "invalid per-layer KV heads or sliding window pattern");
        fused_width_ = std::max(fused_width_, heads_ * kd_ + kv_[il] * (kd_ + vd_));
    }
    const auto* gate = tensor(0, "ffn_gate.weight");
    need(gate && gate->shape.size() == 2, "missing first dense FFN");
    n_ff_ = (int) gate->shape[1];
    clamp_exp_.assign(n_layer_, INFINITY);
    clamp_shexp_.assign(n_layer_, INFINITY);
    std::fprintf(stderr, "strata-poly: experimental MiMo2 text trunk; MTP disabled; context %d\n", opt_.max_context);
}
void Mimo2::begin_pass() { blocks_.assign(layers_.size(), Block{}); }
void Mimo2::load_layer(int il) {
    auto& L = layers_[il];
    auto& B = blocks_[il];
    B.kv = kv_[il];
    B.window = swa_[il] ? window_ : 0;
    B.base = swa_[il] ? base_swa_ : base_;
    auto get = [&](const char* name, int a, int b, Want kind) {
        return upload(tensor(il, name), blk(il, name), a, b, kind);
    };
    L.fixed = false; // Position is passed by value to the baseline attention kernels.
    L.attn_norm = get("attn_norm.weight", n_embd_, 1, kF32);
    L.ffn_norm = get("ffn_norm.weight", n_embd_, 1, kF32);
    B.qkv = get("attn_qkv.weight", n_embd_, heads_ * kd_ + B.kv * (kd_ + vd_), kAnyMat);
    B.wo = get("attn_output.weight", heads_ * vd_, n_embd_, kAnyMat);
    if (tensor(il, "attn_sinks.weight")) B.sinks = get("attn_sinks.weight", heads_, 1, kF32);
    if (!tensor(il, "ffn_gate_inp.weight")) {
        B.gate = get("ffn_gate.weight", n_embd_, n_ff_, kAnyMat);
        B.up = get("ffn_up.weight", n_embd_, n_ff_, kAnyMat);
        B.down = get("ffn_down.weight", n_ff_, n_embd_, kAnyMat);
        return;
    }
    load_moe(il, false, true);
    B.bias.assign(n_expert_, 0.f);
    if (const auto* b = tensor(il, "exp_probs_b.bias")) {
        need(b->type == kTypeF32 && b->elements() == (uint64_t) n_expert_, "expert bias must be F32");
        B.bias_dev = get("exp_probs_b.bias", n_expert_, 1, kF32);
        home(*b).payload(*b);
        std::memcpy(B.bias.data(), home(*b).data(*b), n_expert_ * sizeof(float));
    }
}
int Mimo2::widest() const { return std::max({fused_width_, heads_ * vd_, n_ff_}); }
void Mimo2::assign_scratch() {
    qkv_ = fbuf(size_t(rows_) * fused_width_);
    q_ = fbuf(size_t(rows_) * heads_ * kd_);
    att_ = fbuf(size_t(rows_) * heads_ * vd_);
}
void Mimo2::alloc_caches() {
    for (int il = lo_; il < hi_; ++il) {
        auto& B = blocks_[il];
        B.keys = (uint16_t*) kv_alloc(size_t(B.kv) * kd_ * sizeof(uint16_t), 1);
        B.vals = (uint16_t*) kv_alloc(size_t(B.kv) * vd_ * sizeof(uint16_t), 1);
    }
}
void Mimo2::layer_head(Layer& L, int il, int n) {
    auto& B = blocks_[il];
    rms_norm(r_, fp(L.attn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n * n_embd_, 1, st_);
    lin(B.qkv, xn_, qkv_, n, xq_);
    mimo_qkv(qkv_, q_, B.keys, B.vals, heads_, B.kv, kd_, vd_, rot_, B.base, pos_, n, st_);
    mimo_attend(q_, B.keys, B.vals, fp(B.sinks), att_, heads_, B.kv, kd_, vd_, B.window, pos_, n, st_);
    lin(B.wo, att_, ao_, n);
    axpy(ao_, value_scale_, r_, n * n_embd_, true, st_);
    rms_norm(r_, fp(L.ffn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n * n_embd_, 1, st_);
    if (L.moe) {
        if (!batched(L, n)) moe_router(L, n);
    } else {
        lin(B.gate, xn_, g_, n, xq_);
        lin(B.up, xn_, u_, n, xq_);
        swiglu_clamp(g_, u_, h_, n * n_ff_, INFINITY, st_);
        lin(B.down, h_, fo_, n);
        axpy(fo_, 1.f, r_, n * n_embd_, true, st_);
    }
}
void Mimo2::layer_tail(Layer&, int, int n) { axpy(fo_, 1.f, r_, n * n_embd_, true, st_); }
void Mimo2::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + size_t(first) * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}
void Mimo2::moe_shared(const Layer&, int, int n) {
    ck(cudaMemsetAsync(fo_, 0, size_t(n) * n_embd_ * sizeof(float), cs(st_)), "clear MiMo MoE sum");
}
void Mimo2::route(int il, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    float* pick = probs + n_expert_;
    for (int e = 0; e < n_expert_; ++e) {
        probs[e] = 1.f / (1.f + std::exp(-logits[e]));
        pick[e] = probs[e] + blocks_[il].bias[e];
    }
    float sum = 0;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e) if (pick[e] > pick[best]) best = e;
        ids[k] = best;
        sum += w[k] = probs[best];
        pick[best] = -INFINITY;
    }
    for (int k = 0; k < n_used_; ++k) w[k] = w[k] / std::max(sum, 6.103515625e-5f) * weight_scale_;
}
bool Mimo2::route_card(int il, int n, int32_t* ids, float* w) {
    route_rows(rl_, n_expert_, n_used_, n, 1, 1.f, fp(blocks_[il].bias_dev), nullptr,
               true, weight_scale_, ids, w, st_);
    return true;
}
size_t Mimo2::card_row_bytes() const {
    size_t bytes = 0;
    for (int il = lo_; il < hi_; ++il) bytes += size_t(blocks_[il].kv) * (kd_ + vd_) * sizeof(uint16_t);
    return bytes;
}
void Mimo2::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first > opt_.max_context - n) throw std::logic_error("MiMo save_rows bounds");
    ck(cudaStreamSynchronize(cs(st_)), "save MiMo rows");
    for (int il = lo_; il < hi_; ++il) {
        const auto& B = blocks_[il];
        for (int v = 0; v < 2; ++v) {
            const size_t row = size_t(B.kv) * (v ? vd_ : kd_) * sizeof(uint16_t);
            if (n) ck(cudaMemcpy(dst, (uint8_t*) (v ? B.vals : B.keys) + size_t(first) * row,
                                  size_t(n) * row, cudaMemcpyDeviceToHost), "save MiMo KV");
            dst += size_t(n) * row;
        }
    }
}
void Mimo2::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || count < 0 || first < from || n < 0 || first > opt_.max_context - n ||
        first - from > count || n > count - (first - from)) throw std::logic_error("MiMo load_rows bounds");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load MiMo rows");
    for (int il = lo_; il < hi_; ++il) {
        const auto& B = blocks_[il];
        for (int v = 0; v < 2; ++v) {
            const size_t row = size_t(B.kv) * (v ? vd_ : kd_) * sizeof(uint16_t);
            if (n) ck(cudaMemcpy((uint8_t*) (v ? B.vals : B.keys) + size_t(first) * row,
                                  src + size_t(first - from) * row, size_t(n) * row, cudaMemcpyHostToDevice),
                      "load MiMo KV");
            src += size_t(count) * row;
        }
    }
}
}
