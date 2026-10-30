/*
[Role] Tiny body-working AIV Add kernel for A5 launch/duration control.
[Invariants] n floats, single AIV block; reads A/B once, writes C once.
  Body is a real vector Add, not an empty kernel and not a host-only call.
[Timeouts & Locks] Host control bounds each stream sync to 10s.
[Side-Effect Policy] Writes only the output buffer supplied by the caller.
*/
#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void tiny_aiv_add(GM_ADDR ga, GM_ADDR gb, GM_ADDR gc, uint32_t n) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inA, inB;
    TQue<QuePosition::VECOUT, 1> outC;
    pipe.InitBuffer(inA, 1, n * sizeof(float));
    pipe.InitBuffer(inB, 1, n * sizeof(float));
    pipe.InitBuffer(outC, 1, n * sizeof(float));
    GlobalTensor<float> a, b, c;
    a.SetGlobalBuffer((__gm__ float *)ga, n);
    b.SetGlobalBuffer((__gm__ float *)gb, n);
    c.SetGlobalBuffer((__gm__ float *)gc, n);
    auto la = inA.AllocTensor<float>();
    auto lb = inB.AllocTensor<float>();
    DataCopy(la, a, n);
    DataCopy(lb, b, n);
    inA.EnQue(la); inB.EnQue(lb);
    la = inA.DeQue<float>(); lb = inB.DeQue<float>();
    auto lc = outC.AllocTensor<float>();
    Add(lc, la, lb, n);
    outC.EnQue(lc);
    lc = outC.DeQue<float>();
    DataCopy(c, lc, n);
    outC.FreeTensor(lc); inA.FreeTensor(la); inB.FreeTensor(lb);
}
