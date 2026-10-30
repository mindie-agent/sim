/*
[Role] Drive pure-AIC tiny AscendC Matmul ELF on one explicitly idle A5 die.
  Profiler TaskDuration is the primary duration source; ACL event batches and
  host submit spans are recorded as separate physical boundaries.
[Invariants] Loads the supplied cube-magic ELF, fixes one block per launch,
  runs warmup then repeated batches, and checks every output against a CPU
  reference before accepting a shape. No tensor-core API fallback.
[Timeouts & Locks] 10s stream-sync budget per batch; host probe bounded by the
  caller's outer SSH timeout. Exclusive use of device 0 for the process.
[Side-Effect Policy] Loads only the supplied kernel ELF, allocates <=64MiB
  device tensors, prints JSONL to stdout. No files, weights or other devices.
*/
#include <acl/acl.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
static void ck(aclError e, const char *w) {
    if (e != ACL_SUCCESS) {
        const char *m = aclGetRecentErrMsg();
        throw std::runtime_error(std::string(w) + ":" + std::to_string(e) +
                                 (m ? std::string("\n") + m : ""));
    }
}
#define ACL(x) ck((x), #x)
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
static void *dev(size_t bytes) {
    void *p = nullptr; ACL(aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST)); return p;
}
struct Args { void *a; void *b; void *c; };
struct Loaded { aclrtBinHandle bin = nullptr; aclrtFuncHandle fn = nullptr; };
static Loaded load_cube(const char *path, const char *symbol) {
    aclrtBinaryLoadOption o[2]{};
    o[0].type = ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD; o[0].value.isLazyLoad = 0;
    o[1].type = ACL_RT_BINARY_LOAD_OPT_MAGIC; o[1].value.magic = ACL_RT_BINARY_MAGIC_ELF_CUBE_CORE;
    aclrtBinaryLoadOptions opts{o, 2};
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("binary_unavailable");
    std::vector<char> data((std::istreambuf_iterator<char>(in)), {});
    Loaded l;
    ACL(aclrtBinaryLoadFromData(data.data(), data.size(), &opts, &l.bin));
    ACL(aclrtBinaryGetFunction(l.bin, symbol, &l.fn));
    return l;
}
static void one_shape(const std::string &run_id, const char *symbol, Loaded &l, int m, int n, int k,
                      aclrtStream stream) {
    constexpr int warm = 20, reps = 50, batches = 3;
    const size_t sa = (size_t)m * k * 2, sb = (size_t)k * n * 2, sc = (size_t)m * n * 4;
    std::vector<uint16_t> ha(m * k), hb(k * n);
    std::vector<float> hc(m * n);
    for (int i = 0; i < m * k; ++i) ha[i] = f2h((float)(((i % 7) - 3)) * 0.25f);
    for (int i = 0; i < k * n; ++i) hb[i] = f2h((float)(((i % 5) - 2)) * 0.5f);
    Args args{dev(sa), dev(sb), dev(sc)};
    ACL(aclrtMemcpy(args.a, sa, ha.data(), sa, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL(aclrtMemcpy(args.b, sb, hb.data(), sb, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL(aclrtMemcpy(args.c, sc, hc.data(), sc, ACL_MEMCPY_HOST_TO_DEVICE));
    for (int i = 0; i < warm; ++i)
        ACL(aclrtLaunchKernelWithHostArgs(l.fn, 1, stream, nullptr, &args, sizeof(args), nullptr, 0));
    ACL(aclrtSynchronizeStreamWithTimeout(stream, 10000));
    aclrtEvent s = nullptr, e = nullptr;
    ACL(aclrtCreateEvent(&s)); ACL(aclrtCreateEvent(&e));
    std::printf("{\"kind\":\"device_event_batch\",\"run_id\":\"%s\",\"op\":\"%s\",\"shape\":[%d,%d,%d],"
                "\"blocks\":1,\"warmup\":%d,\"repetitions\":%d,\"batch_us\":[",
                run_id.c_str(), symbol, m, n, k, warm, reps);
    double submit_us = 0;
    for (int b = 0; b < batches; ++b) {
        ACL(aclrtRecordEvent(s, stream));
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i)
            ACL(aclrtLaunchKernelWithHostArgs(l.fn, 1, stream, nullptr, &args, sizeof(args), nullptr, 0));
        auto t1 = std::chrono::steady_clock::now();
        submit_us += std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
        ACL(aclrtRecordEvent(e, stream));
        ACL(aclrtSynchronizeStreamWithTimeout(stream, 10000));
        float ms = 0; ACL(aclrtEventElapsedTime(&ms, s, e));
        std::printf("%s%.6f", b ? "," : "", ms * 1000.0f / reps);
    }
    std::printf("],\"host_submit_us_per_launch\":%.6f}\n", submit_us / batches);
    ACL(aclrtDestroyEvent(s)); ACL(aclrtDestroyEvent(e));
    ACL(aclrtMemcpy(hc.data(), sc, args.c, sc, ACL_MEMCPY_DEVICE_TO_HOST));
    bool ok = true; double maxerr = 0;
    for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) {
        float want = 0;
        for (int t = 0; t < k; ++t) want += h2f(ha[i * k + t]) * h2f(hb[t * n + j]);
        float got = hc[i * n + j];
        maxerr = std::max(maxerr, (double)std::fabs(got - want));
        ok &= std::fabs(got - want) < 5e-3f;
    }
    std::printf("{\"kind\":\"numeric_check\",\"run_id\":\"%s\",\"op\":\"%s\",\"shape\":[%d,%d,%d],"
                "\"passed\":%s,\"max_abs_err\":%.8g}\n",
                run_id.c_str(), symbol, m, n, k, ok ? "true" : "false", maxerr);
    if (!ok) throw std::runtime_error("aic_numeric_mismatch");
    ACL(aclrtFree(args.a)); ACL(aclrtFree(args.b)); ACL(aclrtFree(args.c));
}
int main(int argc, char **argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: tiny_aic_launch CUBE.elf RUN_ID\n"); return 2; }
    const std::string run_id = argv[2];
    try {
        ACL(aclInit(nullptr)); ACL(aclrtSetDevice(0));
        size_t freeb = 0, total = 0;
        ACL(aclrtGetMemInfo(ACL_HBM_MEM, &freeb, &total));
        std::printf("{\"kind\":\"precheck\",\"run_id\":\"%s\",\"free_hbm_bytes\":%zu,"
                    "\"total_hbm_bytes\":%zu}\n", run_id.c_str(), freeb, total);
        if (freeb < (1ull << 30)) throw std::runtime_error("insufficient_free_hbm");
        aclrtStream stream = nullptr; ACL(aclrtCreateStream(&stream));
        Loaded l16 = load_cube(argv[1], "tiny_aic_matmul16");
        Loaded l32 = load_cube(argv[1], "tiny_aic_matmul32");
        Loaded l64 = load_cube(argv[1], "tiny_aic_matmul64");
        one_shape(run_id, "tiny_aic_matmul16", l16, 16, 16, 16, stream);
        one_shape(run_id, "tiny_aic_matmul32", l32, 32, 32, 32, stream);
        one_shape(run_id, "tiny_aic_matmul64", l64, 64, 64, 64, stream);
        ACL(aclrtBinaryUnLoad(l16.bin)); ACL(aclrtBinaryUnLoad(l32.bin)); ACL(aclrtBinaryUnLoad(l64.bin));
        ACL(aclrtDestroyStream(stream)); ACL(aclrtResetDevice(0)); ACL(aclFinalize());
        return 0;
    } catch (const std::exception &x) { std::fprintf(stderr, "%s\n", x.what()); return 1; }
}
