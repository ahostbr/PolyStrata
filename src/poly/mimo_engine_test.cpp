#include "engine.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>
using namespace strata::poly;
template<class T> std::vector<T> read(const std::string& file) {
    std::ifstream in(file, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot read " + file);
    const auto bytes = in.tellg();
    if (bytes <= 0 || size_t(bytes) % sizeof(T)) throw std::runtime_error("bad fixture size");
    std::vector<T> v(size_t(bytes)/sizeof(T));
    in.seekg(0); in.read((char*)v.data(), bytes);
    if (!in) throw std::runtime_error("short fixture read");
    return v;
}
void compare(const float* a, const float* b, int n, float tol, const char* what) {
    float worst=0;
    double squared=0, reference=0;
    bool failed=false;
    for (int i=0; i<n; ++i) {
        worst=std::max(worst, std::abs(a[i]-b[i]));
        squared+=double(a[i]-b[i])*(a[i]-b[i]); reference+=double(b[i])*b[i];
        failed |= !std::isfinite(a[i]) || std::abs(a[i]-b[i]) > tol*(1+std::abs(b[i]));
    }
    const double relative=std::sqrt(squared/std::max(reference,1e-20));
    failed |= relative > tol;
    std::printf("%s %s max absolute error %.6g, relative L2 %.6g\n",failed?"FAIL":"PASS",what,worst,relative);
    if (failed) throw std::runtime_error(std::string(what)+" logits differ");
}
int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr,"usage: poly_mimo_engine_test FIXTURE_PREFIX\n"); return 2; }
    try {
        std::string prefix=argv[1];
        auto tokens=read<int32_t>(prefix+".tokens");
        auto expected=read<float>(prefix+".logits");
        for (bool draft : {false,true}) {
            Options invalid; invalid.model=prefix+".gguf"; invalid.draft=false;
            invalid.max_context=draft?256:8193;
            if (draft) invalid.draft_model=prefix+".gguf";
            bool rejected=false;
            try { auto engine=Engine::open(invalid); }
            catch (const std::runtime_error& e) {
                rejected=std::string(e.what()).find(draft?"MTP/DFlash":"context limit")!=std::string::npos;
            }
            if (!rejected) throw std::runtime_error("MiMo context/draft guard failed");
        }
        std::puts("PASS MiMo context and draft guards");
        for (bool gpu : {false,true}) {
            Options o; o.model=prefix+".gguf"; o.max_context=256; o.prompt_batch=64;
            o.gpu_experts=gpu; o.expert_swaps=0; o.stage_mib=16; o.stage_ahead=64; o.threads=4;
            auto e=Engine::open(o);
            if (e->drafts() || expected.size()!=tokens.size()*size_t(e->n_vocab()))
                throw std::runtime_error("fixture geometry or MTP skipping failed");
            const int vocab=e->n_vocab();
            for (size_t at=0; at<tokens.size();) {
                int n=int(std::min(size_t(64),tokens.size()-at));
                e->eval(tokens.data()+at,n,true); at+=n;
                // Reference dequantizes weights to F32; the engine also quantizes activations for
                // each product. Bound accumulated quantization error by both elementwise and L2 tests.
                compare(e->logits(),expected.data()+(at-1)*vocab,vocab,.015f,gpu?"GPU batch/reference":"CPU batch/reference");
            }
            std::vector<float> batch(e->logits(),e->logits()+vocab);
            e->reset();
            for (size_t i=0; i<tokens.size(); ++i) e->eval(tokens.data()+i,1,i+1==tokens.size());
            compare(e->logits(),expected.data()+(tokens.size()-1)*vocab,vocab,.015f,
                    gpu?"GPU decode/reference":"CPU decode/reference");
            compare(e->logits(),batch.data(),vocab,.02f,gpu?"GPU decode/batch":"CPU decode/batch");
            // Save a prefix's KV and state, overwrite it, then restore a subrange of a larger snapshot.
            e->reset(); e->eval(tokens.data(),64,false); e->eval(tokens.data()+64,64,true);
            std::vector<uint8_t> rows(128*e->row_bytes()), state(e->state_bytes());
            e->save_rows(0,128,rows.data()); e->save_state(state.data());
            e->reset(); std::vector<int32_t> other(64,2);
            e->eval(other.data(),64,false); e->eval(other.data(),64,false);
            e->load_rows(rows.data(),0,128,0,64); e->load_rows(rows.data(),0,128,64,64);
            e->load_state(state.data(),128); e->eval(tokens.data()+128,9,true);
            compare(e->logits(),batch.data(),vocab,1e-6f,gpu?"GPU checkpoint":"CPU checkpoint");
            if (gpu) {
                if (e->cached_experts() == 0 || e->stats().batch_cached == 0)
                    throw std::runtime_error("GPU cache path was not exercised");
                // Leave no expert slots; a layer-major read must stage eligible experts.
                e->resize_cache(1 << 20);
                if (e->cached_experts() != 0) throw std::runtime_error("cache shrink failed");
                e->reset();
                int turns=0;
                const std::function<bool(int,int)> progress=[&](int done, int total) {
                    if (total!=4 || done!=++turns) throw std::runtime_error("expected a layer-major read");
                    return true;
                };
                if (!e->read(tokens.data(), (int)tokens.size()-1, nullptr, 0, &progress))
                    throw std::runtime_error("read interrupted");
                if (turns!=3) throw std::runtime_error("layer-major path was not exercised");
                e->eval(tokens.data()+tokens.size()-1,1,true);
                compare(e->logits(),expected.data()+(tokens.size()-1)*vocab,vocab,.015f,"staged read/reference");
                if (e->stats().batch_staged == 0) throw std::runtime_error("staging was not exercised");
            }
        }
        std::puts("MiMo engine integration: PASS"); return 0;
    } catch(const std::exception& ex) { std::fprintf(stderr,"FAIL: %s\n",ex.what()); return 1; }
}
