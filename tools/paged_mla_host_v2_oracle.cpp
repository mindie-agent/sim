/* [Role] Execute original SplitCoreV2 CPU metadata for q1 dense paged MLA.
 * [Invariants] Context excludes the current token; inactive KV length is zero.
 * [Timeouts & Locks] One bounded input geometry, no threads or locks.
 * [Side-Effect Policy] Print source metadata; no device or tensor allocation.
 * Compile alongside the pinned original split_core_v2.cpp and header.
 */
#include <iostream>
#include <sstream>
#include <string>
#include "split_core_v2.h"

int main(int argc, char **argv) {
    if (argc != 5) return 2;
    std::vector<unsigned> contexts;
    std::istringstream input(argv[1]);
    std::string value;
    while (std::getline(input, value, ',')) contexts.push_back(std::stoul(value));
    const unsigned heads = std::stoul(argv[2]), cores = std::stoul(argv[3]);
    const unsigned capacity = std::stoul(argv[4]);
    if (contexts.empty() || capacity < contexts.size()) return 2;
    split_core_v2::BaseInfo base{};
    base.bSize=capacity; base.n2Size=1; base.gSize=heads; base.s1Size=1;
    base.isS1G=false;
    base.actualLenQDims=capacity; base.actualLenKvDims=capacity;
    base.actualSeqS1Size.assign(capacity,1); base.actualSeqS2Size.assign(capacity,0);
    for (unsigned i=0;i<contexts.size();++i) base.actualSeqS2Size[i]=contexts[i]+1;
    split_core_v2::SplitParam params{};
    params.mBaseSize=64; params.s2BaseSize=128; params.gS1BaseSizeOfFd=8;
    params.streamK=true; params.fdTolerance=9; params.fdLeastBlock=0;
    split_core_v2::FAMetaData result{cores,2};
    split_core_v2::SplitCore(cores,base,params,result);
    std::cout << "{\"used_aic\":" << result.usedCoreNum << ",\"ends\":[";
    for (unsigned i=0;i<result.usedCoreNum;++i) {
        if(i) std::cout << ',';
        std::cout << '[' << result.bN2End[i] << ',' << result.mEnd[i] << ',' << result.s2End[i] << ']';
    }
    std::cout << "],\"fd\":[";
    for(unsigned i=0;i<result.fdRes.fdNum;++i) {
        if(i) std::cout << ',';
        std::cout << "{\"sequence\":" << result.fdRes.fdBN2Idx[i]
                  << ",\"m_tile\":" << result.fdRes.fdMIdx[i]
                  << ",\"splits\":" << result.fdRes.fdS2SplitNum[i] << '}';
    }
    std::cout << "],\"fd_aiv\":[";
    for(unsigned i=0;i<result.fdRes.fdUsedVecNum;++i) {
        if(i) std::cout << ',';
        std::cout << '[' << result.fdRes.taskIdx[i] << ',' << result.fdRes.mStart[i]
                  << ',' << result.fdRes.mLen[i] << ']';
    }
    std::cout << "]}\n";
}
