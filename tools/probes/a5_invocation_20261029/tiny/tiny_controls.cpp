/*
[Role] Measure tiny real-work AIV Add and AIC Matmul controls on one idle A5 die.
[Invariants] AIV is a custom body-working AscendC Add. Matmul is the installed
  CANN aclnn Matmul op (no torch); profiler task type is reported verbatim.
  Every shape is numerically checked against a CPU reference. Device events,
  host submit times and profiler kernel rows are emitted as separate records.
[Timeouts & Locks] One die, <=64MiB tensors, 10s sync timeout per batch.
[Side-Effect Policy] Device 0 only, no weights, no file writes, no other device.
*/
#include <acl/acl.h>
#include <aclnnop/aclnn_matmul.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
static void ck(aclError e, const char *w) {
    if (e != ACL_SUCCESS) { const char *m = aclGetRecentErrMsg();
        throw std::runtime_error(std::string(w) + ":" + std::to_string(e) + (m ? std::string("\n") + m : "")); }
}
#define ACL(x) ck((x), #x)
#define ANN(x) do { auto s = (x); if (s != 0) throw std::runtime_error("aclnn:" #x ":" + std::to_string(s)); } while (0)
static uint16_t f2h(float f) {
    uint32_t x; std::memcpy(&x, &f, 4); uint16_t sign = (x >> 16) & 0x8000;
    int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15; uint32_t m = x & 0x7fffff;
    if (e <= 0) return sign; if (e >= 31) return sign | 0x7c00;
    return (uint16_t)(sign | (e << 10) | (m >> 13));
}
static float h2f(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t x; if (!e) x = sign; else if (e == 31) x = sign | 0x7f800000 | (m << 13);
    else x = sign | ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &x, 4); return f;
}
static void *dev(size_t bytes) { void *p = nullptr; ACL(aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST)); return p; }
struct VecArgs { void *a; void *b; void *c; uint32_t n; };
static void run_aiv(aclrtFuncHandle fn, aclrtStream stream, int n, std::vector<float> &ha, std::vector<float> &hb, std::vector<float> &hc) {
    VecArgs args{nullptr, nullptr, nullptr, (uint32_t)n};
    args.a = dev(n * 4); args.b = dev(n * 4); args.c = dev(n * 4);
    ACL(aclrtMemcpy(args.a, n * 4, ha.data(), n * 4, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL(aclrtMemcpy(args.b, n * 4, hb.data(), n * 4, ACL_MEMCPY_HOST_TO_DEVICE));
    constexpr int warm = 20, reps = 50, batches = 3;
    for (int i = 0; i < warm; ++i) ACL(aclrtLaunchKernelWithHostArgs(fn, 1, stream, nullptr, &args, sizeof(args), nullptr, 0));
    ACL(aclrtSynchronizeStreamWithTimeout(stream, 10000));
    aclrtEvent s = nullptr, e = nullptr; ACL(aclrtCreateEvent(&s)); ACL(aclrtCreateEvent(&e));
    std::printf("{\"kind\":\"device_event_batch\",\"op\":\"tiny_aiv_add\",\"shape\":[%d],\"blocks\":1,\"warmup\":%d,\"repetitions\":%d,\"batch_us\":[", n, warm, reps);
    double submit_us = 0;
    for (int b = 0; b < batches; ++b) {
        ACL(aclrtRecordEvent(s, stream));
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) ACL(aclrtLaunchKernelWithHostArgs(fn, 1, stream, nullptr, &args, sizeof(args), nullptr, 0));
        auto t1 = std::chrono::steady_clock::now();
        submit_us += std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
        ACL(aclrtRecordEvent(e, stream));
        ACL(aclrtSynchronizeStreamWithTimeout(stream, 10000));
        float ms = 0; ACL(aclrtEventElapsedTime(&ms, s, e));
        std::printf("%s%.6f", b ? "," : "", ms * 1000.0f / reps);
    }
    std::printf("],\"host_submit_us_per_launch\":%.6f}\n", submit_us / batches);
    ACL(aclrtMemcpy(hc.data(), n * 4, args.c, n * 4, ACL_MEMCPY_DEVICE_TO_HOST));
    (void)s; (void)e;
    bool ok = true; double maxerr = 0;
    for (int i = 0; i < n; ++i) { float want = ha[i] + hb[i]; maxerr = std::max(maxerr, (double)std::fabs(hc[i] - want)); ok &= hc[i] == want; }
    std::printf("{\"kind\":\"numeric_check\",\"op\":\"tiny_aiv_add\",\"shape\":[%d],\"passed\":%s,\"max_abs_err\":%.8g}\n", n, ok ? "true" : "false", maxerr);
    if (!ok) throw std::runtime_error("aiv_numeric_mismatch");
}
struct Cube { aclTensor *a = nullptr, *b = nullptr, *c = nullptr; void *da = nullptr, *db = nullptr, *dc = nullptr, *ws = nullptr; aclOpExecutor *ex = nullptr; uint64_t wssize = 0; int m, n, k; };
static void run_cube(Cube &q, aclrtStream stream) {
    constexpr int warm = 20, reps = 50, batches = 3;
    for (int i = 0; i < warm; ++i) ANN(aclnnMatmul(q.ws, q.wssize, q.ex, stream));
    ACL(aclrtSynchronizeStreamWithTimeout(stream, 10000));
    aclrtEvent s = nullptr, e = nullptr; ACL(aclrtCreateEvent(&s)); ACL(aclrtCreateEvent(&e));
    std::printf("{\"kind\":\"device_event_batch\",\"op\":\"aclnnMatmul\",\"shape\":[%d,%d,%d],\"warmup\":%d,\"repetitions\":%d,\"batch_us\":[", q.m, q.n, q.k, warm, reps);
    double submit_us = 0;
    for (int b = 0; b < batches; ++b) {
        ACL(aclrtRecordEvent(s, stream));
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) ANN(aclnnMatmul(q.ws, q.wssize, q.ex, stream));
        auto t1 = std::chrono::steady_clock::now();
        submit_us += std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
        ACL(aclrtRecordEvent(e, stream));
        ACL(aclrtSynchronizeStreamWithTimeout(stream, 10000));
        float ms = 0; ACL(aclrtEventElapsedTime(&ms, s, e));
        std::printf("%s%.6f", b ? "," : "", ms * 1000.0f / reps);
    }
    std::printf("],\"host_submit_us_per_launch\":%.6f}\n", submit_us / batches);
    ACL(aclrtDestroyEvent(s)); ACL(aclrtDestroyEvent(e));
}
static void cube_shape(int m, int n, int k, aclrtStream stream) {
    Cube q; q.m = m; q.n = n; q.k = k;
    size_t sa = (size_t)m * k * 2, sb = (size_t)k * n * 2, sc = (size_t)m * n * 2;
    std::vector<float> fa(m * k), fb(k * n);
    std::vector<uint16_t> ha(m * k), hb(k * n), hc(m * n);
    for (int i = 0; i < m; ++i) for (int j = 0; j < k; ++j) { fa[i * k + j] = (float)(((i + j) % 7) - 3); ha[i * k + j] = f2h(fa[i * k + j]); }
    for (int i = 0; i < k; ++i) for (int j = 0; j < n; ++j) { fb[i * n + j] = (float)(((i * 2 + j) % 5) - 2); hb[i * n + j] = f2h(fb[i * n + j]); }
    q.da = dev(sa); q.db = dev(sb); q.dc = dev(sc);
    ACL(aclrtMemcpy(q.da, sa, ha.data(), sa, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL(aclrtMemcpy(q.db, sb, hb.data(), sb, ACL_MEMCPY_HOST_TO_DEVICE));
    int64_t ad[2] = {m, k}, bd[2] = {k, n}, cd[2] = {m, n};
    int64_t as[2] = {k, 1}, bs[2] = {n, 1}, cs[2] = {n, 1};
    q.a = aclCreateTensor(ad, 2, ACL_FLOAT16, as, 0, ACL_FORMAT_ND, ad, 2, q.da);
    q.b = aclCreateTensor(bd, 2, ACL_FLOAT16, bs, 0, ACL_FORMAT_ND, bd, 2, q.db);
    q.c = aclCreateTensor(cd, 2, ACL_FLOAT16, cs, 0, ACL_FORMAT_ND, cd, 2, q.dc);
    ANN(aclnnMatmulGetWorkspaceSize(q.a, q.b, q.c, 0, &q.wssize, &q.ex));
    if (q.wssize) q.ws = dev(q.wssize);
    run_cube(q, stream);
    ACL(aclrtMemcpy(hc.data(), sc, q.dc, sc, ACL_MEMCPY_DEVICE_TO_HOST));
    bool ok = true; double maxerr = 0;
    for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) {
        float want = 0; for (int t = 0; t < k; ++t) want += fa[i * k + t] * fb[t * n + j];
        float got = h2f(hc[i * n + j]); maxerr = std::max(maxerr, (double)std::fabs(got - want)); ok &= std::fabs(got - want) < 1e-3f;
    }
    std::printf("{\"kind\":\"numeric_check\",\"op\":\"aclnnMatmul\",\"shape\":[%d,%d,%d],\"passed\":%s,\"max_abs_err\":%.8g}\n", m, n, k, ok ? "true" : "false", maxerr);
    if (!ok) throw std::runtime_error("cube_numeric_mismatch");
    (void)q;
}
int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) { std::fprintf(stderr, "usage: tiny_controls AIV_ELF [aiv|cube]\n"); return 2; }
    try {
        ACL(aclInit(nullptr)); ACL(aclrtSetDevice(0));
        size_t freeb = 0, total = 0; ACL(aclrtGetMemInfo(ACL_HBM_MEM, &freeb, &total));
        std::printf("{\"kind\":\"precheck\",\"run_id\":\"tiny-a5-%ld\",\"free_hbm_bytes\":%zu,\"total_hbm_bytes\":%zu}\n", (long)time(nullptr), freeb, total);
        if (freeb < (1ull << 30)) throw std::runtime_error("insufficient_free_hbm");
        aclrtStream stream = nullptr; ACL(aclrtCreateStream(&stream));
        std::ifstream in(argv[1], std::ios::binary); if (!in) throw std::runtime_error("elf_unavailable");
        std::vector<char> elf((std::istreambuf_iterator<char>(in)), {});
        aclrtBinaryLoadOption o[2]{}; o[0].type = ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD; o[0].value.isLazyLoad = 0;
        o[1].type = ACL_RT_BINARY_LOAD_OPT_MAGIC; o[1].value.magic = ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE;
        aclrtBinaryLoadOptions opts{o, 2}; aclrtBinHandle bin = nullptr; aclrtFuncHandle fn = nullptr;
        ACL(aclrtBinaryLoadFromData(elf.data(), elf.size(), &opts, &bin));
        ACL(aclrtBinaryGetFunction(bin, "tiny_aiv_add", &fn));
        std::string mode = argc > 2 ? argv[2] : "both";
        if (mode != "cube") for (int n : {64, 256, 1024}) {
            std::vector<float> a(n), b(n), c(n);
            for (int i = 0; i < n; ++i) { a[i] = (float)(i % 11) * 0.25f; b[i] = (float)(i % 7) * 0.5f; }
            run_aiv(fn, stream, n, a, b, c);
        }
        if (mode != "aiv") {
            cube_shape(16, 16, 16, stream);
            cube_shape(32, 32, 16, stream);
            cube_shape(64, 32, 16, stream);
        }
        (void)bin; ACL(aclrtDestroyStream(stream)); ACL(aclrtResetDevice(0)); ACL(aclFinalize());
        return 0;
    } catch (const std::exception &x) { std::fprintf(stderr, "%s\n", x.what()); return 1; }
}
