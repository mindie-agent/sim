/*
[Role] Tiny body-working AIC-only AscendC Matmul controls for A5 launch and
  device-duration measurement. Each symbol fixes one MxNxK geometry.
[Invariants] Compiled for dav-c310-cube with AIC-only task metadata; reads A/B
  once and writes C once per launch. Static tiling is compile-time; no AIV
  client, no empty body, no host fallback.
[Timeouts & Locks] Device execution is bounded by the host probe (<=120s).
[Side-Effect Policy] Writes only the output buffer supplied by the caller.
*/
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

using AType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

template <int M, int N, int K>
__aicore__ inline void RunTinyCube(GM_ADDR a, GM_ADDR b, GM_ADDR c)
{
    using Shape = Std::tuple<Std::integral_constant<int, M>,
                             Std::integral_constant<int, N>,
                             Std::integral_constant<int, K>>;
    using Base = Std::tuple<Std::integral_constant<int, 16>,
                            Std::integral_constant<int, 16>,
                            Std::integral_constant<int, 16>>;
    static constexpr auto cfg = GetMatmulApiTiling<AType, BType, CType, BiasType, Shape, Shape, Base>(CFG_NORM);

    TPipe pipe;
    Matmul<AType, BType, CType, BiasType, cfg> mm;
    GlobalTensor<half> ga;
    GlobalTensor<half> gb;
    GlobalTensor<float> gc;
    ga.SetGlobalBuffer((__gm__ half *)a, M * K);
    gb.SetGlobalBuffer((__gm__ half *)b, K * N);
    gc.SetGlobalBuffer((__gm__ float *)c, M * N);
    mm.SetOrgShape(M, N, K);
    mm.SetTensorA(ga, false);
    mm.SetTensorB(gb, false);
    mm.IterateAll(gc);
    mm.End();
}

extern "C" __global__ __aicore__ void tiny_aic_matmul16(GM_ADDR a, GM_ADDR b, GM_ADDR c)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    RunTinyCube<16, 16, 16>(a, b, c);
}

extern "C" __global__ __aicore__ void tiny_aic_matmul32(GM_ADDR a, GM_ADDR b, GM_ADDR c)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    RunTinyCube<32, 32, 32>(a, b, c);
}

extern "C" __global__ __aicore__ void tiny_aic_matmul64(GM_ADDR a, GM_ADDR b, GM_ADDR c)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    RunTinyCube<64, 64, 64>(a, b, c);
}
