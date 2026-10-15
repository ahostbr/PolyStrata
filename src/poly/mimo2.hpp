#pragma once
#include "engine.hpp"

namespace strata::poly {
// Experimental text-only MiMo2 trunk. Appended MTP layers are skipped. All KV
// rows are kept at absolute positions so the existing checkpoint API applies.
class Mimo2 final : public Engine {
protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    size_t recurrent_floats() const override { return 0; }
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override;
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;
    void save_layer(int, int64_t, uint8_t*) override {}
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;
private:
    struct Block {
        int kv = 0, window = 0;
        float base = 0;
        Dev qkv, wo, sinks, gate, up, down, bias_dev;
        std::vector<float> bias;
        uint16_t *keys = nullptr, *vals = nullptr;
    };
    std::vector<Block> blocks_;
    std::vector<int> kv_, swa_;
    int heads_ = 0, kd_ = 0, vd_ = 0, rot_ = 0, window_ = 0, fused_width_ = 0;
    float base_ = 0, base_swa_ = 0, value_scale_ = 1, weight_scale_ = 1;
    float *qkv_ = nullptr, *q_ = nullptr, *att_ = nullptr;
};
}
