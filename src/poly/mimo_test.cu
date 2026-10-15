// Small deterministic CPU references; no model download required.
#include "strata/poly/mimo_kernels.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace strata::poly;
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t n) { check(cudaMalloc((void**)&p, n * sizeof(T))); }
    ~Buffer() { cudaFree(p); }
    void put(const std::vector<T>& v) { check(cudaMemcpy(p, v.data(), v.size()*sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get(size_t n) const {
        std::vector<T> v(n);
        check(cudaMemcpy(v.data(), p, n*sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};
void close(float got, float expected, float tol, const char* name) {
    if (!std::isfinite(got) || std::abs(got - expected) > tol * (1 + std::abs(expected))) {
        std::fprintf(stderr, "%s: got %.9g expected %.9g\n", name, got, expected);
        throw std::runtime_error("MiMo parity failed");
    }
}
float rotate(const float* x, int d, int rot, double base, int pos) {
    if (d >= rot) return x[d];
    const int i = d % (rot / 2);
    const double angle = pos * std::pow(base, -2.0 * i / rot);
    return float(d < rot / 2 ? x[d]*std::cos(angle) - x[d+rot/2]*std::sin(angle)
                            : x[d]*std::cos(angle) + x[d-rot/2]*std::sin(angle));
}
void run(int heads, int kv, int tokens, int start, int rows, int window, bool sink, float base) {
    constexpr int kd = 192, vd = 128, rot = 64;
    const int width = heads*kd + kv*(kd+vd);
    std::vector<float> fused(size_t(tokens)*width), sinks(heads);
    for (size_t i=0; i<fused.size(); ++i) fused[i] = float(std::sin(double(i)*0.137)*0.7);
    for (int h=0; h<heads; ++h) sinks[h] = h % 3 == 0 ? 90.f : (h % 3 == 1 ? -90.f : 0.3f);
    Buffer<float> f(fused.size()), q(size_t(tokens)*heads*kd), out(size_t(rows)*heads*vd), s(heads);
    Buffer<uint16_t> k(size_t(tokens)*kv*kd), v(size_t(tokens)*kv*vd);
    f.put(fused); s.put(sinks);
    // Store in uneven calls, so nonzero absolute positions and packed row strides are exercised.
    for (int p=0; p<tokens;) {
        int n=std::min(37, tokens-p);
        mimo_qkv(f.p+size_t(p)*width, q.p+size_t(p)*heads*kd, k.p, v.p, heads, kv, kd, vd, rot,
                  base, p, n, nullptr);
        p+=n;
    }
    auto qs=q.get(size_t(tokens)*heads*kd);
    auto ks=k.get(size_t(tokens)*kv*kd), vs=v.get(size_t(tokens)*kv*vd);
    for (int t=0; t<tokens; ++t) {
        const float* x=fused.data()+size_t(t)*width;
        for (int h=0; h<heads; ++h) for (int d=0; d<kd; ++d)
            close(qs[(size_t(t)*heads+h)*kd+d], rotate(x+h*kd,d,rot,base,t), 4e-4f, "partial RoPE Q");
        for (int h=0; h<kv; ++h) {
            for (int d=0; d<kd; ++d)
                close(__half2float(reinterpret_cast<const half*>(ks.data())[(size_t(t)*kv+h)*kd+d]),
                      rotate(x+heads*kd+h*kd,d,rot,base,t), 9e-4f, "partial RoPE K");
            for (int d=0; d<vd; ++d)
                close(__half2float(reinterpret_cast<const half*>(vs.data())[(size_t(t)*kv+h)*vd+d]),
                      __half2float(__float2half(x[heads*kd+kv*kd+h*vd+d])), 0, "unrotated V");
        }
    }
    mimo_attend(q.p+size_t(start)*heads*kd, k.p, v.p, sink?s.p:nullptr, out.p,
                 heads, kv, kd, vd, window, start, rows, nullptr);
    auto got=out.get(size_t(rows)*heads*vd);
    const half* keys=reinterpret_cast<const half*>(ks.data());
    const half* vals=reinterpret_cast<const half*>(vs.data());
    for (int r=0; r<rows; ++r) for (int h=0; h<heads; ++h) {
        int end=start+r+1, begin=window?std::max(0,end-window):0, kh=h/(heads/kv);
        std::vector<double> scores(end-begin);
        double mx=sink?sinks[h]:-INFINITY;
        for (int t=begin; t<end; ++t) {
            double dot=0;
            for (int d=0; d<kd; ++d)
                dot+=double(qs[(size_t(start+r)*heads+h)*kd+d])*__half2float(keys[(size_t(t)*kv+kh)*kd+d]);
            scores[t-begin]=dot/std::sqrt(double(kd));
            mx=std::max(mx,scores[t-begin]);
        }
        double denom=sink?std::exp(sinks[h]-mx):0;
        for (double& score:scores) denom+=score=std::exp(score-mx);
        for (int d=0; d<vd; ++d) {
            double expected=0;
            for (int t=begin; t<end; ++t) expected+=scores[t-begin]*__half2float(vals[(size_t(t)*kv+kh)*vd+d]);
            close(got[(size_t(r)*heads+h)*vd+d],float(expected/denom),3e-5f,"causal sink attention");
        }
    }
    // A prefilled batch must agree with sequential decode even with future keys present.
    for (int r=0; r<rows; ++r)
        mimo_attend(q.p+size_t(start+r)*heads*kd,k.p,v.p,sink?s.p:nullptr,
                     out.p+size_t(r)*heads*vd,heads,kv,kd,vd,window,start+r,1,nullptr);
    auto sequential=out.get(got.size());
    for (size_t i=0; i<got.size(); ++i) close(sequential[i],got[i],0,"batch/decode");
    std::printf("PASS heads=%d kv=%d pos=%d rows=%d window=%d sinks=%d\n",heads,kv,start,rows,window,sink);
}
int main() {
    try {
        run(64,8,137,0,4,128,true,10000.f);
        run(64,8,137,125,12,128,true,10000.f);
        run(64,4,137,125,12,0,false,10000000.f);
        run(4,2,8192,8191,1,0,true,10000000.f);
        bool rejected=false;
        try { mimo_attend(nullptr,nullptr,nullptr,nullptr,nullptr,4,2,192,128,0,8192,1,nullptr); }
        catch (const std::invalid_argument&) { rejected=true; }
        if (!rejected) throw std::runtime_error("context guard did not reject overflow");
        std::puts("MiMo primitive parity: PASS");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
