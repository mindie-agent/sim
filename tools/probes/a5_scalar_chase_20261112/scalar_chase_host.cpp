/*
[Role] Host driver for the A5 native scalar dependent-GM pointer-chase probe.
[Invariants] Uses ACL device events for the primary duration, host submit time
  separately; D0 is the fixed-start control; checks final indices.
[Timeouts & Locks] One device/stream; bounded sync and finite memory.
[Side-Effect Policy] Device buffers only; writes no files.
*/
#include <acl/acl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <numeric>
#include <random>
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

struct Args { void *t0; void *t1; void *out; int32_t n; int32_t s0; int32_t s1; };
struct Loaded { aclrtBinHandle bin = nullptr; aclrtFuncHandle fn = nullptr; };

static std::vector<int32_t> cycle(int32_t n, uint64_t seed) {
    std::vector<int32_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<int32_t> tab(n);
    for (int32_t i = 0; i < n; ++i) tab[order[i]] = order[(i + 1) % n];
    return tab;
}

static int32_t advance(const std::vector<int32_t> &a,
                       const std::vector<int32_t> &b, int32_t start, int depth) {
    int32_t idx = start;
    for (int d = 0; d < depth; ++d) { idx = a[idx]; idx = b[idx]; }
    return idx;
}

static Loaded load(const std::string &path, const char *symbol) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("binary_unavailable");
    std::vector<char> data((std::istreambuf_iterator<char>(in)), {});
    aclrtBinaryLoadOption o[2]{};
    o[0].type = ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD; o[0].value.isLazyLoad = 0;
    o[1].type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
    o[1].value.magic = ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE;
    aclrtBinaryLoadOptions opts{o, 2};
    Loaded l;
    ACL(aclrtBinaryLoadFromData(data.data(), data.size(), &opts, &l.bin));
    ACL(aclrtBinaryGetFunction(l.bin, symbol, &l.fn));
    return l;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

struct Case { const char *name; int depth; int chains; };

static void run_case(const Case &c, const std::string &binary, int32_t n,
                     const std::vector<int32_t> &h0, const std::vector<int32_t> &h1,
                     const std::vector<int32_t> &expected, aclrtStream stream,
                     const std::string &footprint) {
    Loaded l = load(binary, c.name);
    Args args{nullptr, nullptr, nullptr, n, 0, c.chains == 2 ? n / 2 : 0};
    size_t bytes = (size_t)n * sizeof(int32_t);
    ACL(aclrtMalloc(&args.t0, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL(aclrtMalloc(&args.t1, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL(aclrtMalloc(&args.out, 2 * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL(aclrtMemcpy(args.t0, bytes, h0.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL(aclrtMemcpy(args.t1, bytes, h1.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    for (int i = 0; i < 5; ++i)
        ACL(aclrtLaunchKernelWithHostArgs(l.fn, 1, stream, nullptr, &args,
                                          sizeof(args), nullptr, 0));
    ACL(aclrtSynchronizeStreamWithTimeout(stream, 30000));

    aclrtEvent s = nullptr, e = nullptr;
    ACL(aclrtCreateEvent(&s)); ACL(aclrtCreateEvent(&e));
    std::vector<double> ev_us, host_us;
    int32_t out[2] = {0, 0};
    for (int r = 0; r < 7; ++r) {
        ACL(aclrtRecordEvent(s, stream));
        auto h0t = std::chrono::steady_clock::now();
        ACL(aclrtLaunchKernelWithHostArgs(l.fn, 1, stream, nullptr, &args,
                                          sizeof(args), nullptr, 0));
        auto h1t = std::chrono::steady_clock::now();
        ACL(aclrtRecordEvent(e, stream));
        ACL(aclrtSynchronizeStreamWithTimeout(stream, 30000));
        float ms = 0;
        ACL(aclrtEventElapsedTime(&ms, s, e));
        ev_us.push_back((double)ms * 1000.0);
        host_us.push_back(std::chrono::duration<double, std::micro>(h1t - h0t).count());
        ACL(aclrtMemcpy(out, sizeof(out), args.out, sizeof(out),
                        ACL_MEMCPY_DEVICE_TO_HOST));
        if (out[0] != expected[0] || (c.chains == 2 && out[1] != expected[1]))
            throw std::runtime_error("final_index_mismatch:" + std::string(c.name));
    }
    ACL(aclrtDestroyEvent(s)); ACL(aclrtDestroyEvent(e));
    std::printf("{\"kind\":\"case\",\"footprint\":\"%s\",\"n\":%d,"
                "\"working_set_bytes\":%zu,\"depth\":%d,\"chains\":%d,"
                "\"dependent_loads_per_launch\":%d,\"event_us\":[",
                footprint.c_str(), n, (size_t)8 * n, c.depth, c.chains,
                2 * c.depth * c.chains);
    for (size_t i = 0; i < ev_us.size(); ++i) std::printf("%s%.9f", i ? "," : "", ev_us[i]);
    std::printf("],\"host_submit_us\":[");
    for (size_t i = 0; i < host_us.size(); ++i) std::printf("%s%.9f", i ? "," : "", host_us[i]);
    std::printf("],\"expected\":[%d,%d],\"observed\":[%d,%d],\"final_ok\":true}\n",
                expected[0], expected[1], out[0], out[1]);
    ACL(aclrtFree(args.t0)); ACL(aclrtFree(args.t1)); ACL(aclrtFree(args.out));
    ACL(aclrtBinaryUnLoad(l.bin));
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: scalar_chase DEV_ELF RUN_ID\n");
        return 2;
    }
    try {
        ACL(aclInit(nullptr)); ACL(aclrtSetDevice(0));
        const char *soc = aclrtGetSocName();
        size_t freeb = 0, total = 0;
        ACL(aclrtGetMemInfo(ACL_HBM_MEM, &freeb, &total));
        std::printf("{\"kind\":\"precheck\",\"run_id\":\"%s\",\"soc\":\"%s\","
                    "\"free_hbm_bytes\":%zu,\"total_hbm_bytes\":%zu}\n",
                    argv[2], soc ? soc : "unknown", freeb, total);
        if (freeb < (2ull << 30)) throw std::runtime_error("insufficient_free_hbm");
        aclrtStream stream = nullptr; ACL(aclrtCreateStream(&stream));

        struct Foot { const char *name; int32_t n; uint64_t seed0, seed1; };
        const Foot feet[] = {{"small_16k", 2048, 0x51a7, 0x7c31},
                             {"large_128m", 16777216, 0x9e37, 0xc2b2}};
        const Case cases[] = {
            {"chase_d0_c1", 0, 1}, {"chase_d128_c1", 128, 1},
            {"chase_d256_c1", 256, 1}, {"chase_d512_c1", 512, 1},
            {"chase_d0_c2", 0, 2}, {"chase_d128_c2", 128, 2},
            {"chase_d256_c2", 256, 2}, {"chase_d512_c2", 512, 2}};
        for (const Foot &f : feet) {
            auto h0 = cycle(f.n, f.seed0);
            auto h1 = cycle(f.n, f.seed1);
            for (const Case &c : cases) {
                std::vector<int32_t> exp(2, 0);
                exp[0] = advance(h0, h1, 0, c.depth);
                exp[1] = c.chains == 2 ? advance(h0, h1, f.n / 2, c.depth) : 0;
                run_case(c, argv[1], f.n, h0, h1, exp, stream, f.name);
            }
        }
        ACL(aclrtDestroyStream(stream)); ACL(aclrtResetDevice(0)); ACL(aclFinalize());
        return 0;
    } catch (const std::exception &e) {
        std::printf("{\"kind\":\"error\",\"message\":\"%s\"}\n", e.what());
        return 1;
    }
}
