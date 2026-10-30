/*
[Role] Launch empty AIV/AIC kernels and record device-event batches on one
  explicitly idle A5 die; profiler rows are the primary duration source.
[Invariants] No tensor allocation, no compute/copy kernel body; fixed argument
  layout; warmup then repeated batches; events never converted into kernel time.
  Precheck requires >=1 GiB free HBM and the operator-selected idle die only.
[Timeouts & Locks] 10s stream-sync budget per batch; outer ssh/msprof timeout
  >=120s. Exclusive use of one die for the process lifetime.
[Side-Effect Policy] Loads only the supplied kernel ELF, launches it, prints
  JSONL to stdout. No files, weights, runtime defaults or other devices touched.
*/
#include <acl/acl.h>
#include <cstdio>
#include <cstdlib>
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
struct Loaded { aclrtBinHandle bin = nullptr; aclrtFuncHandle fn = nullptr; };
static Loaded load(const char *path, const char *symbol, unsigned magic) {
    aclrtBinaryLoadOption o[2]{};
    o[0].type = ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD; o[0].value.isLazyLoad = 0;
    o[1].type = ACL_RT_BINARY_LOAD_OPT_MAGIC; o[1].value.magic = magic;
    aclrtBinaryLoadOptions opts{o, 2};
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("binary_unavailable");
    std::vector<char> data((std::istreambuf_iterator<char>(in)), {});
    Loaded l;
    ACL(aclrtBinaryLoadFromData(data.data(), data.size(), &opts, &l.bin));
    ACL(aclrtBinaryGetFunction(l.bin, symbol, &l.fn));
    std::printf("{\"event\":\"loaded\",\"path\":\"%s\",\"symbol\":\"%s\",\"bytes\":%zu}\n",
                path, symbol, data.size());
    return l;
}
static void run(Loaded &l, const char *name, int block, aclrtStream stream) {
    constexpr int warm = 50, reps = 100, batches = 5;
    uint32_t dummy = 0;
    for (int i = 0; i < warm; ++i)
        ACL(aclrtLaunchKernelWithHostArgs(l.fn, block, stream, nullptr, &dummy, sizeof(dummy), nullptr, 0));
    ACL(aclrtSynchronizeStreamWithTimeout(stream, 10000));
    aclrtEvent s = nullptr, e = nullptr;
    ACL(aclrtCreateEvent(&s)); ACL(aclrtCreateEvent(&e));
    std::printf("{\"event\":\"batch\",\"kernel\":\"%s\",\"blocks\":%d,\"warmup\":%d,"
                "\"repetitions\":%d,\"device_event_us\":[", name, block, warm, reps);
    for (int b = 0; b < batches; ++b) {
        ACL(aclrtRecordEvent(s, stream));
        for (int i = 0; i < reps; ++i)
            ACL(aclrtLaunchKernelWithHostArgs(l.fn, block, stream, nullptr, &dummy, sizeof(dummy), nullptr, 0));
        ACL(aclrtRecordEvent(e, stream));
        ACL(aclrtSynchronizeStreamWithTimeout(stream, 10000));
        float ms = 0; ACL(aclrtEventElapsedTime(&ms, s, e));
        std::printf("%s%.6f", b ? "," : "", ms * 1000.0f / reps);
    }
    std::printf("]}\n");
    ACL(aclrtDestroyEvent(s)); ACL(aclrtDestroyEvent(e));
}
int main(int argc, char **argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: empty_launch AIV.elf AIC.elf\n"); return 2; }
    try {
        ACL(aclInit(nullptr)); ACL(aclrtSetDevice(0));
        size_t freeb = 0, total = 0;
        ACL(aclrtGetMemInfo(ACL_HBM_MEM, &freeb, &total));
        std::printf("{\"event\":\"precheck\",\"free_hbm_bytes\":%zu,\"total_hbm_bytes\":%zu}\n", freeb, total);
        if (freeb < (1ull << 30)) throw std::runtime_error("insufficient_free_hbm");
        aclrtStream stream = nullptr; ACL(aclrtCreateStream(&stream));
        Loaded aiv = load(argv[1], "empty_aiv", ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE);
        Loaded aic = load(argv[2], "empty_aic", ACL_RT_BINARY_MAGIC_ELF_CUBE_CORE);
        for (int blocks : {1, 8, 64}) { run(aiv, "empty_aiv", blocks, stream); run(aic, "empty_aic", blocks, stream); }
        ACL(aclrtBinaryUnLoad(aiv.bin)); ACL(aclrtBinaryUnLoad(aic.bin));
        ACL(aclrtDestroyStream(stream)); ACL(aclrtResetDevice(0)); ACL(aclFinalize());
        return 0;
    } catch (const std::exception &x) { std::fprintf(stderr, "%s\n", x.what()); return 1; }
}
