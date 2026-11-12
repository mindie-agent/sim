/*
[Role] A5 native scalar GM pointer-chase response probe.
[Invariants] Two dependent int32 GlobalTensor::GetValue loads per step:
  idx = table0.GetValue(idx); idx = table1.GetValue(idx).  The tables are
  random single-cycle permutations; final indices are written to GM and
  checked on the host.  Depth is compile-time, chains are independent 1 or 2.
[Timeouts & Locks] One AIV block; device sync bounded by host launcher.
[Side-Effect Policy] Writes only the two-int checksum buffer supplied.
*/
#include "kernel_operator.h"
using namespace AscendC;

template <int DEPTH, int CHAINS>
__aicore__ inline void RunChase(GM_ADDR t0, GM_ADDR t1, GM_ADDR out,
                                int32_t n, int32_t s0, int32_t s1) {
    GlobalTensor<int32_t> a, b;
    a.SetGlobalBuffer((__gm__ int32_t *)t0, n);
    b.SetGlobalBuffer((__gm__ int32_t *)t1, n);
    int32_t i0 = s0, i1 = s1;
#pragma unroll 1
    for (int d = 0; d < DEPTH; ++d) {
        i0 = a.GetValue(i0);
        i0 = b.GetValue(i0);
        if constexpr (CHAINS == 2) {
            i1 = a.GetValue(i1);
            i1 = b.GetValue(i1);
        }
    }
    GlobalTensor<int32_t> o;
    o.SetGlobalBuffer((__gm__ int32_t *)out, 2);
    o.SetValue(0, i0);
    if constexpr (CHAINS == 2) o.SetValue(1, i1);
}

#define DEF(D, C) \
extern "C" __global__ __aicore__ void chase_d##D##_c##C( \
    GM_ADDR t0, GM_ADDR t1, GM_ADDR out, int32_t n, int32_t s0, int32_t s1) { \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); \
    RunChase<D, C>(t0, t1, out, n, s0, s1); \
}
DEF(0, 1) DEF(128, 1) DEF(256, 1) DEF(512, 1)
DEF(0, 2) DEF(128, 2) DEF(256, 2) DEF(512, 2)
