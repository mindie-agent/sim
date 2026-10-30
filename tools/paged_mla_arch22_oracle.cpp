// Compile with the pinned ops-transformer attention/common/op_host/split_core.cpp.
// This driver only supplies q1 dense MLA inputs and exports the unmodified host result.
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include "split_core.h"

int main(int argc, char **argv) {
    if (argc < 4) return 2;
    using namespace optiling;
    const auto cores = std::stoul(argv[1]);
    BaseInfo b{};
    b.bSize = argc - 3;
    b.n2Size = 1;
    b.gSize = std::stoul(argv[2]);
    b.s1Size = 1;
    b.isS1G = false; // BNSD caller; q1 makes the single M group identical.
    b.actualLenKvDims = b.bSize;
    for (int i = 3; i < argc; ++i) {
        auto length = std::stoul(argv[i]) + 1;
        b.actualSeqS2Size.push_back(length);
        b.s2Size = std::max(b.s2Size, static_cast<uint32_t>(length));
    }
    SplitParam p{};
    p.mBaseSize = 256;
    p.s2BaseSize = 512;
    p.gS1BaseSizeOfFd = 8;
    SplitResult r(cores, 2);
    SplitCore(cores, b, p, r);
    std::cout << "{\"cores\":" << cores << ",\"heads\":" << b.gSize
              << ",\"key_tile\":512,\"contexts\":[";
    for (unsigned i = 0; i < b.bSize; ++i) {
        if (i) std::cout << ",";
        std::cout << b.actualSeqS2Size[i] - 1;
    }
    std::cout << "],\"ends\":[";
    for (unsigned i = 0; i < r.usedCoreNum; ++i) {
        if (i) std::cout << ",";
        std::cout << "[" << r.bN2End[i] << "," << r.gS1End[i] << "," << r.s2End[i] << "]";
    }
    std::cout << "],\"fd\":[";
    for (unsigned i = 0; i < r.numOfFdHead; ++i) {
        if (i) std::cout << ",";
        std::cout << "[" << r.fdRes.bN2IdxOfFdHead[i] << "," << r.fdRes.gS1IdxOfFdHead[i]
                  << "," << r.fdRes.s2SplitNumOfFdHead[i]
                  << "," << r.fdRes.gS1SplitNumOfFdHead[i]
                  << "," << r.fdRes.gS1LastPartSizeOfFdHead[i] << "]";
    }
    std::cout << "],\"fd_ends\":[";
    for (unsigned i = 0; i < r.usedVecNumOfFd; ++i) {
        if (i) std::cout << ",";
        std::cout << "[" << r.fdRes.gS1IdxEndOfFdHead[i]
                  << "," << r.fdRes.gS1IdxEndOfFdHeadSplit[i] << "]";
    }
    std::cout << "]}\n";
}
