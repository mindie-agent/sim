/*
[Role] Measure compiled primitive kernels without importing a model runtime.
[Invariants] No weights; identical binary and argument layout across shapes.
  Event batches amortize host submission; every written element is checked.
[Timeouts & Locks] External timeout plus a 10s device synchronization budget.
[Side-Effect Policy] Device 0, <=16 MiB buffers; stdout is JSONL observations.
*/
#include <acl/acl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

static void check(aclError error, const char *operation) {
    if (error != ACL_SUCCESS) {
        const char *details = aclGetRecentErrMsg();
        throw std::runtime_error(std::string(operation) + ":" + std::to_string(error) +
                                 (details ? std::string("\n") + details : ""));
    }
}
#define ACL(call) check((call), #call)

struct StoreArgs {
    void *lock = nullptr, *workspace = nullptr, *output = nullptr, *input = nullptr;
    int32_t heads = 0, tokens = 0, iterations = 0, gx = 64, gy = 1, gz = 1;
    void *debug = nullptr;
};
struct StridedArgs {
    void *lock = nullptr, *workspace = nullptr, *output = nullptr, *input = nullptr;
    int32_t heads = 0, tokens = 0, iterations = 0, leading_dim = 0, gx = 64, gy = 1, gz = 1;
    void *debug = nullptr;
};
struct GatingArgs {
    void *lock = nullptr, *workspace = nullptr;
    void *g = nullptr, *beta_output = nullptr, *a_log = nullptr;
    void *a = nullptr, *b = nullptr, *bias = nullptr;
    int32_t seq_len = 1, heads = 0, tokens = 0;
    float beta = 1, threshold = 20;
    int32_t iterations = 0, gx = 64, gy = 1, gz = 1;
    void *debug = nullptr;
};
struct Shape {
    int tokens, heads, units = 64, leading_dim = 0, offset = 0;
};

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr, "usage: native_kernel_probe BINARY "
                            "strided_store|strided_store_ld|gating [reverse|CASES]\n");
        return 2;
    }
    try {
        const std::string primitive = argv[2];
        const bool gating = primitive == "gating", strided = primitive == "strided_store_ld";
        if (!gating && !strided && primitive != "strided_store")
            throw std::runtime_error("unknown_probe");
        ACL(aclInit(nullptr));
        ACL(aclrtSetDevice(0));
        size_t free_bytes = 0, total = 0;
        ACL(aclrtGetMemInfo(ACL_HBM_MEM, &free_bytes, &total));
        constexpr size_t budget = 16 * 1024 * 1024;
        if (free_bytes < budget + 512 * 1024 * 1024)
            throw std::runtime_error("insufficient_free_device_memory");
        aclrtStream stream = nullptr;
        ACL(aclrtCreateStream(&stream));
        aclrtBinaryLoadOption options[2]{};
        options[0].type = ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD;
        options[0].value.isLazyLoad = 0;
        options[1].type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
        options[1].value.magic = ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE;
        aclrtBinaryLoadOptions load_options{options, 2};
        aclrtBinHandle binary = nullptr;
        std::ifstream input(argv[1], std::ios::binary);
        if (!input) throw std::runtime_error("binary_file_unavailable");
        std::vector<char> payload((std::istreambuf_iterator<char>(input)), {});
        ACL(aclrtBinaryLoadFromData(payload.data(), payload.size(), &load_options, &binary));
        aclrtFuncHandle function = nullptr;
        ACL(aclrtBinaryGetFunction(binary, gating ? "fused_gdn_gating_kernel" : argv[2], &function));
        void *storage = nullptr;
        ACL(aclrtMalloc(&storage, budget, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL(aclrtMemset(storage, budget, 0, budget));
        auto address = [&](size_t offset) { return static_cast<char *>(storage) + offset; };
        std::vector<uint16_t> seed(64 * 8, 0x3e80);
        ACL(aclrtMemcpy(address(15 * 1024 * 1024), seed.size() * 2, seed.data(),
                        seed.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE));
        aclrtEvent start = nullptr, end = nullptr;
        ACL(aclrtCreateEvent(&start));
        ACL(aclrtCreateEvent(&end));
        std::vector<Shape> shapes = gating
            ? std::vector<Shape>{{64, 48}, {4096, 48}, {16384, 48}}
            : std::vector<Shape>{{4096, 32}, {8192, 32}, {4096, 64},
                                 {8192, 64}, {12288, 40}, {4225, 33}};
        if (argc == 4) {
            if (std::string(argv[3]) == "reverse") {
                std::reverse(shapes.begin(), shapes.end());
            } else {
                std::ifstream cases(argv[3]);
                if (!cases) throw std::runtime_error("cases_file_unavailable");
                shapes.clear();
                Shape shape{};
                while (cases >> shape.tokens >> shape.heads >> shape.units >>
                       shape.leading_dim >> shape.offset)
                    shapes.push_back(shape);
                if (!cases.eof() || shapes.empty()) throw std::runtime_error("invalid_cases");
            }
        }
        for (auto [tokens, heads, units, leading_dim, offset] : shapes) {
            if (!leading_dim) leading_dim = heads;
            if (tokens < 1 || heads < 1 || units < 1 || units > 64 || offset < 0 ||
                leading_dim < heads || (size_t(tokens) * leading_dim + offset) * 4 >
                4 * 1024 * 1024 || (!strided && leading_dim != heads) ||
                (gating && offset != 0))
                throw std::runtime_error("case_exceeds_probe_geometry_or_budget");
            const size_t elements = size_t(tokens) * heads;
            const size_t physical_elements = size_t(tokens) * leading_dim + offset;
            ACL(aclrtMemset(storage, 14 * 1024 * 1024, 0, 14 * 1024 * 1024));
            StoreArgs store;
            store.output = address(offset * 2);
            store.input = address(15 * 1024 * 1024);
            store.heads = heads; store.tokens = tokens;
            store.iterations = (tokens + units * 64 - 1) / (units * 64);
            store.gx = units;
            StridedArgs strided_args;
            strided_args.output = store.output; strided_args.input = store.input;
            strided_args.heads = heads; strided_args.tokens = tokens;
            strided_args.iterations = store.iterations;
            strided_args.leading_dim = leading_dim; strided_args.gx = units;
            GatingArgs gate;
            gate.g = storage; gate.beta_output = address(4 * 1024 * 1024);
            gate.a_log = address(6 * 1024 * 1024); gate.bias = address(7 * 1024 * 1024);
            gate.a = address(8 * 1024 * 1024); gate.b = address(12 * 1024 * 1024);
            gate.heads = heads; gate.tokens = tokens; gate.iterations = store.iterations; gate.gx = units;
            void *args = gating ? static_cast<void *>(&gate) :
                         strided ? static_cast<void *>(&strided_args) : static_cast<void *>(&store);
            size_t args_size = gating ? sizeof(gate) : strided ? sizeof(strided_args) : sizeof(store);
            auto launch = [&]() {
                ACL(aclrtLaunchKernelWithHostArgs(function, units, stream, nullptr,
                                                  args, args_size, nullptr, 0));
            };
            constexpr int warmup = 512;
            for (int i = 0; i < warmup; ++i) launch();
            ACL(aclrtSynchronizeStreamWithTimeout(stream, 10000));
            constexpr int repetitions = 20;
            std::vector<float> trials;
            for (int trial = 0; trial < 5; ++trial) {
                ACL(aclrtRecordEvent(start, stream));
                for (int i = 0; i < repetitions; ++i) launch();
                ACL(aclrtRecordEvent(end, stream));
                ACL(aclrtSynchronizeStreamWithTimeout(stream, 10000));
                float elapsed = 0;
                ACL(aclrtEventElapsedTime(&elapsed, start, end));
                trials.push_back(elapsed * 1000 / repetitions);
            }
            std::vector<uint16_t> output(physical_elements);
            ACL(aclrtMemcpy(output.data(), physical_elements * 2, gating ? gate.beta_output : storage,
                            physical_elements * 2, ACL_MEMCPY_DEVICE_TO_HOST));
            for (size_t i = 0; i < physical_elements; ++i) {
                const bool written = i >= size_t(offset) && (i - offset) % leading_dim < size_t(heads);
                if (output[i] != (written ? 0x3f00 : 0))
                    throw std::runtime_error("output_or_padding_mismatch");
            }
            if (gating) {
                std::vector<float> decay(elements);
                ACL(aclrtMemcpy(decay.data(), elements * 4, storage,
                                elements * 4, ACL_MEMCPY_DEVICE_TO_HOST));
                for (float value : decay)
                    if (std::abs(value + std::log(2.0f)) > 1e-5f)
                        throw std::runtime_error("decay_output_mismatch");
            }
            std::printf("{\"primitive\":\"%s\",\"tokens\":%d,\"heads\":%d,"
                        "\"units\":%d,\"leading_dim\":%d,\"offset_elements\":%d,"
                        "\"warmup\":%d,\"repetitions\":%d,\"durations_us\":[",
                        argv[2], tokens, heads, units, leading_dim, offset, warmup, repetitions);
            for (size_t i = 0; i < trials.size(); ++i)
                std::printf("%s%.6f", i ? "," : "", trials[i]);
            std::printf("],\"all_elements_verified\":true}\n");
            std::fflush(stdout);
        }
        ACL(aclrtDestroyEvent(start)); ACL(aclrtDestroyEvent(end));
        ACL(aclrtFree(storage)); ACL(aclrtBinaryUnLoad(binary));
        ACL(aclrtDestroyStream(stream)); ACL(aclrtResetDevice(0)); ACL(aclFinalize());
        rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        std::printf("{\"peak_rss_kib\":%ld,\"tensor_budget_bytes\":%zu}\n", usage.ru_maxrss, budget);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
