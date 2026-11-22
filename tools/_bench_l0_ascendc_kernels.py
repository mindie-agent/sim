"""Native AscendC C++ kernel templates for L0 hardware primitive calibration.

[Role] Hold the self-contained ``bisheng -xasc`` C++ microkernel source strings
  consumed by ``tools/_bench_l0_ascendc.py`` (Domains 1-4 on-chip pipes,
  pure UB vector/sort, Arch35 SIMT, intra/cross-core sync, and MTE2 DMA).
[Invariants] Pure immutable string templates with zero runtime logic, I/O, or
  host-specific paths; supports both ``dav-2201`` (A2/A3) and ``dav-3510`` (A5).
[Timeouts & Locks] Import-time constants only; zero locks or execution.
[Side-Effect Policy] Pure module constants; zero side effects.
"""
from __future__ import annotations

VEC_AND_P12_CPP = r'''
#include "kernel_operator.h"
extern "C" __global__ __vector__ void p12_flag_mte2_v(GM_ADDR out, int iters) {
    for (int i = 0; i < iters; ++i) {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    }
}
extern "C" __global__ __vector__ void p12_flag_v_mte3(GM_ADDR out, int iters) {
    for (int i = 0; i < iters; ++i) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    }
}
extern "C" __global__ __vector__ void p12_pipe_v_barrier(GM_ADDR out, int iters) {
    for (int i = 0; i < iters; ++i) { AscendC::PipeBarrier<PIPE_V>(); }
}
extern "C" __global__ __cube__ void p12_flag_mte1_m(GM_ADDR out, int iters) {
    for (int i = 0; i < iters; ++i) {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
    }
}
extern "C" __global__ __cube__ void p12_flag_m_fix(GM_ADDR out, int iters) {
    for (int i = 0; i < iters; ++i) {
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
    }
}
extern "C" __global__ __cube__ void p06_pipe_m_barrier(GM_ADDR out, int iters) {
    for (int i = 0; i < iters; ++i) { AscendC::PipeBarrier<PIPE_M>(); }
}
#if defined(_MINDIE_ARCH35)
extern "C" __global__ __cube__ void p14_mutex_pipe_m(GM_ADDR out, int iters) {
    for (int i = 0; i < iters; ++i) {
        AscendC::Mutex::Lock<PIPE_M>(0);
        AscendC::Mutex::Unlock<PIPE_M>(0);
    }
}
extern "C" __global__ __cube__ void p14_mutex_xpipe(GM_ADDR out, int iters) {
    for (int i = 0; i < iters; ++i) {
        AscendC::Mutex::Lock<PIPE_MTE1>(0);
        AscendC::Mutex::Unlock<PIPE_MTE1>(0);
        AscendC::Mutex::Lock<PIPE_M>(0);
        AscendC::Mutex::Unlock<PIPE_M>(0);
    }
}
#endif
extern "C" __global__ __vector__ void p07_ub_alu_fp32(GM_ADDR in_gm, GM_ADDR out_gm, int iters) {
    constexpr int N = 8192;
    AscendC::TPipe pipe; AscendC::TBuf<AscendC::TPosition::VECCALC> b0, b1, b2;
    pipe.InitBuffer(b0, N * sizeof(float)); pipe.InitBuffer(b1, N * sizeof(float)); pipe.InitBuffer(b2, N * sizeof(float));
    auto a = b0.Get<float>(); auto s = b1.Get<float>(); auto c = b2.Get<float>();
    AscendC::GlobalTensor<float> gIn, gOut;
    gIn.SetGlobalBuffer((__gm__ float*)in_gm + AscendC::GetBlockIdx() * N, N);
    gOut.SetGlobalBuffer((__gm__ float*)out_gm + AscendC::GetBlockIdx() * N, N);
    AscendC::DataCopy(a, gIn, N); AscendC::DataCopy(s, gIn, N); AscendC::DataCopy(c, gIn, N);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    for (int i = 0; i < iters; ++i) { AscendC::MulAddDst(c, a, s, N); AscendC::PipeBarrier<PIPE_V>(); }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::DataCopy(gOut, c, N);
}
extern "C" __global__ __vector__ void p08_ub_exp_fp32(GM_ADDR in_gm, GM_ADDR out_gm, int iters) {
    constexpr int N = 4096;
    AscendC::TPipe pipe; AscendC::TBuf<AscendC::TPosition::VECCALC> bufA, bufB, bufC;
    pipe.InitBuffer(bufA, N * sizeof(float)); pipe.InitBuffer(bufB, N * sizeof(float)); pipe.InitBuffer(bufC, N * sizeof(float));
    auto a = bufA.Get<float>(); auto b = bufB.Get<float>(); auto c = bufC.Get<float>();
    AscendC::GlobalTensor<float> gIn, gOut;
    gIn.SetGlobalBuffer((__gm__ float*)in_gm + AscendC::GetBlockIdx() * N, N);
    gOut.SetGlobalBuffer((__gm__ float*)out_gm + AscendC::GetBlockIdx() * N, N);
    AscendC::DataCopy(a, gIn, N);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    for (int i = 0; i < iters; ++i) {
        AscendC::Exp(b, a, N); AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(c, a, N); AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Add(b, b, c, N); AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::DataCopy(gOut, b, N);
}
extern "C" __global__ __vector__ void p09_ub_reduce_fp32(GM_ADDR in_gm, GM_ADDR out_gm, int iters) {
    constexpr int ROWS = 128; constexpr int COLS = 64; constexpr int N = ROWS * COLS;
    AscendC::TPipe pipe; AscendC::TBuf<AscendC::TPosition::VECCALC> bufIn, bufOut;
    pipe.InitBuffer(bufIn, N * sizeof(float)); pipe.InitBuffer(bufOut, ROWS * sizeof(float));
    auto x = bufIn.Get<float>(); auto y = bufOut.Get<float>();
    AscendC::GlobalTensor<float> gIn, gOut;
    gIn.SetGlobalBuffer((__gm__ float*)in_gm + AscendC::GetBlockIdx() * N, N);
    gOut.SetGlobalBuffer((__gm__ float*)out_gm + AscendC::GetBlockIdx() * ROWS, ROWS);
    AscendC::DataCopy(x, gIn, N);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    for (int i = 0; i < iters; ++i) { AscendC::WholeReduceSum<float>(y, x, COLS, ROWS, 1, 1, 8); AscendC::PipeBarrier<PIPE_V>(); }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::DataCopy(gOut, y, ROWS);
}
extern "C" __global__ __vector__ void p09_ub_cast_f32_bf16(GM_ADDR in_gm, GM_ADDR out_gm, int iters) {
    constexpr int N = 8192;
    AscendC::TPipe pipe; AscendC::TBuf<AscendC::TPosition::VECCALC> bufF, bufB;
    pipe.InitBuffer(bufF, N * sizeof(float)); pipe.InitBuffer(bufB, N * sizeof(bfloat16_t));
    auto xf = bufF.Get<float>(); auto xb = bufB.Get<bfloat16_t>();
    AscendC::GlobalTensor<float> gIn, gOut;
    gIn.SetGlobalBuffer((__gm__ float*)in_gm + AscendC::GetBlockIdx() * N, N);
    gOut.SetGlobalBuffer((__gm__ float*)out_gm + AscendC::GetBlockIdx() * N, N);
    AscendC::DataCopy(xf, gIn, N);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    for (int i = 0; i < iters; ++i) {
        AscendC::Cast(xb, xf, AscendC::RoundMode::CAST_RINT, N); AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(xf, xb, AscendC::RoundMode::CAST_NONE, N); AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::DataCopy(gOut, xf, N);
}
extern "C" {
int run_p12_mte2_v(void* o, int it, void* st, int bd) { p12_flag_mte2_v<<<bd, nullptr, st>>>((uint8_t*)o, it); return 0; }
int run_p12_v_mte3(void* o, int it, void* st, int bd) { p12_flag_v_mte3<<<bd, nullptr, st>>>((uint8_t*)o, it); return 0; }
int run_p12_pipe_v(void* o, int it, void* st, int bd) { p12_pipe_v_barrier<<<bd, nullptr, st>>>((uint8_t*)o, it); return 0; }
int run_p12_mte1_m(void* o, int it, void* st, int bd) { p12_flag_mte1_m<<<bd, nullptr, st>>>((uint8_t*)o, it); return 0; }
int run_p12_m_fix(void* o, int it, void* st, int bd) { p12_flag_m_fix<<<bd, nullptr, st>>>((uint8_t*)o, it); return 0; }
int run_p06_pipe_m(void* o, int it, void* st, int bd) { p06_pipe_m_barrier<<<bd, nullptr, st>>>((uint8_t*)o, it); return 0; }
int run_p07_alu(void* i, void* o, int it, void* st, int bd) { p07_ub_alu_fp32<<<bd, nullptr, st>>>((uint8_t*)i, (uint8_t*)o, it); return 0; }
int run_p08_exp(void* i, void* o, int it, void* st, int bd) { p08_ub_exp_fp32<<<bd, nullptr, st>>>((uint8_t*)i, (uint8_t*)o, it); return 0; }
int run_p09_red(void* i, void* o, int it, void* st, int bd) { p09_ub_reduce_fp32<<<bd, nullptr, st>>>((uint8_t*)i, (uint8_t*)o, it); return 0; }
int run_p09_cast(void* i, void* o, int it, void* st, int bd) { p09_ub_cast_f32_bf16<<<bd, nullptr, st>>>((uint8_t*)i, (uint8_t*)o, it); return 0; }
#if defined(_MINDIE_ARCH35)
int run_p14_mutex(void* o, int it, void* st, int bd) { p14_mutex_pipe_m<<<bd, nullptr, st>>>((uint8_t*)o, it); return 0; }
int run_p14_mutex_xpipe(void* o, int it, void* st, int bd) { p14_mutex_xpipe<<<bd, nullptr, st>>>((uint8_t*)o, it); return 0; }
#endif
}
'''

P13_SYNC_CPP = r'''
#include "kernel_operator.h"
extern "C" __global__ __vector__ void p13_syncall_aiv(GM_ADDR out, int iters) {
    for (int i = 0; i < iters; ++i) { AscendC::SyncAll(); }
}
extern "C" __global__ __cube__ void p13_syncall_aic(GM_ADDR out, int iters) {
    for (int i = 0; i < iters; ++i) { AscendC::SyncAll(); }
}
extern "C" __global__ __mix__(1, 2) void p13_mode2_paired(GM_ADDR out, int iters) {
    for (int i = 0; i < iters; ++i) {
        if ASCEND_IS_AIC {
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0);
            AscendC::CrossCoreWaitFlag(1);
        } else {
            AscendC::CrossCoreWaitFlag(0);
            AscendC::CrossCoreSetFlag<0x2, PIPE_V>(1);
        }
    }
}
#if defined(_MINDIE_ARCH35)
extern "C" __global__ __mix__(1, 2) void p13_mode4_intrablock(GM_ADDR out, int iters) {
    constexpr uint64_t MODE = 4UL;
    constexpr uint16_t COMPANION_AIV_OFFSET = 16;
    for (int i = 0; i < iters; ++i) {
        if ASCEND_IS_AIC {
            AscendC::CrossCoreSetFlag<MODE, PIPE_FIX>(0);
            AscendC::CrossCoreSetFlag<MODE, PIPE_FIX>(COMPANION_AIV_OFFSET);
            AscendC::CrossCoreWaitFlag<MODE, PIPE_FIX>(1);
            AscendC::CrossCoreWaitFlag<MODE, PIPE_FIX>(1 + COMPANION_AIV_OFFSET);
        } else {
            AscendC::CrossCoreWaitFlag<MODE, PIPE_V>(0);
            AscendC::CrossCoreSetFlag<MODE, PIPE_V>(1);
        }
    }
}
#endif
extern "C" {
int run_p13_aiv(void* o, int it, void* st, int bd) { p13_syncall_aiv<<<bd, nullptr, st>>>((uint8_t*)o, it); return 0; }
int run_p13_aic(void* o, int it, void* st, int bd) { p13_syncall_aic<<<bd, nullptr, st>>>((uint8_t*)o, it); return 0; }
int run_p13_mode2(void* o, int it, void* st, int bd) { p13_mode2_paired<<<bd, nullptr, st>>>((uint8_t*)o, it); return 0; }
#if defined(_MINDIE_ARCH35)
int run_p13_mode4(void* o, int it, void* st, int bd) { p13_mode4_intrablock<<<bd, nullptr, st>>>((uint8_t*)o, it); return 0; }
#endif
}
'''

P02_P03_CPP = r'''
#include "kernel_operator.h"
extern "C" __global__ __cube__ void p02_load_l0a(GM_ADDR in_gm, int iters, int do_trans, int fractals, int use_barrier) {
    constexpr int MAX_FRACTALS = 64; constexpr int ELEMS = MAX_FRACTALS * 256;
    AscendC::TPipe pipe; AscendC::TBuf<AscendC::TPosition::A1> l1Buf; AscendC::TBuf<AscendC::TPosition::A2> l0aBuf;
    pipe.InitBuffer(l1Buf, ELEMS * sizeof(half)); pipe.InitBuffer(l0aBuf, ELEMS * sizeof(half));
    auto l1 = l1Buf.Get<half>(); auto l0a = l0aBuf.Get<half>();
    AscendC::GlobalTensor<half> gIn; gIn.SetGlobalBuffer((__gm__ half*)in_gm, ELEMS);
    AscendC::DataCopy(l1, gIn, ELEMS);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
#if defined(_MINDIE_ARCH35)
    AscendC::LoadData2DParamsV2 params;
    params.mStartPosition = 0; params.kStartPosition = 0;
    params.mStep = (fractals <= 16) ? 4 : 8; params.kStep = (fractals <= 16) ? 4 : 8;
    params.srcStride = params.kStep; params.dstStride = params.kStep; params.ifTranspose = (do_trans != 0);
#else
    AscendC::LoadData2DParams params;
    params.startIndex = 0; params.repeatTimes = static_cast<uint8_t>(fractals); params.srcStride = 1;
    params.sid = 0; params.dstGap = 0; params.ifTranspose = (do_trans != 0); params.addrMode = 0;
#endif
    for (int i = 0; i < iters; ++i) { AscendC::LoadData(l0a, l1, params); if (use_barrier != 0) AscendC::PipeBarrier<PIPE_MTE1>(); }
    AscendC::PipeBarrier<PIPE_MTE1>();
}
extern "C" __global__ __cube__ void p02_load_l0b(GM_ADDR in_gm, int iters, int do_trans, int fractals, int use_barrier) {
    constexpr int MAX_FRACTALS = 64; constexpr int ELEMS = MAX_FRACTALS * 256;
    AscendC::TPipe pipe; AscendC::TBuf<AscendC::TPosition::B1> l1Buf; AscendC::TBuf<AscendC::TPosition::B2> l0bBuf;
    pipe.InitBuffer(l1Buf, ELEMS * sizeof(half)); pipe.InitBuffer(l0bBuf, ELEMS * sizeof(half));
    auto l1 = l1Buf.Get<half>(); auto l0b = l0bBuf.Get<half>();
    AscendC::GlobalTensor<half> gIn; gIn.SetGlobalBuffer((__gm__ half*)in_gm, ELEMS);
    AscendC::DataCopy(l1, gIn, ELEMS);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
#if defined(_MINDIE_ARCH35)
    AscendC::LoadData2DParamsV2 params;
    params.mStartPosition = 0; params.kStartPosition = 0;
    params.mStep = (fractals <= 16) ? 4 : 8; params.kStep = (fractals <= 16) ? 4 : 8;
    params.srcStride = params.kStep; params.dstStride = params.kStep; params.ifTranspose = (do_trans != 0);
#else
    AscendC::LoadData2DParams params;
    params.startIndex = 0; params.repeatTimes = static_cast<uint8_t>(fractals); params.srcStride = 1;
    params.sid = 0; params.dstGap = 0; params.ifTranspose = (do_trans != 0); params.addrMode = 0;
#endif
    for (int i = 0; i < iters; ++i) { AscendC::LoadData(l0b, l1, params); if (use_barrier != 0) AscendC::PipeBarrier<PIPE_MTE1>(); }
    AscendC::PipeBarrier<PIPE_MTE1>();
}
extern "C" __global__ __cube__ void p03_fixpipe_gm(GM_ADDR out_gm, int iters, int m_rows, int n_cols, int use_barrier) {
    constexpr int MAX_ELEMS = 128 * 128;
    AscendC::TPipe pipe; AscendC::TBuf<AscendC::TPosition::CO1> l0cBuf;
    pipe.InitBuffer(l0cBuf, MAX_ELEMS * sizeof(float));
    auto l0c = l0cBuf.Get<float>();
    AscendC::GlobalTensor<bfloat16_t> gOut;
    gOut.SetGlobalBuffer((__gm__ bfloat16_t*)out_gm + AscendC::GetBlockIdx() * MAX_ELEMS, MAX_ELEMS);
#if defined(_MINDIE_ARCH35)
    AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> params;
#else
    AscendC::FixpipeParamsV220 params;
#endif
    params.mSize = static_cast<uint16_t>(m_rows); params.nSize = static_cast<uint16_t>(n_cols);
    params.srcStride = static_cast<uint16_t>(m_rows); params.dstStride = static_cast<uint32_t>(n_cols);
    params.quantPre = QuantMode_t::F322BF16;
    for (int i = 0; i < iters; ++i) {
        AscendC::Fixpipe<bfloat16_t, float, AscendC::CFG_ROW_MAJOR>(gOut, l0c, params);
        if (use_barrier != 0) AscendC::PipeBarrier<PIPE_FIX>();
    }
    AscendC::PipeBarrier<PIPE_FIX>();
}
extern "C" {
int run_p02_l0a(void* i, int it, int tr, int fr, int bar, void* st, int bd) { p02_load_l0a<<<bd, nullptr, st>>>((uint8_t*)i, it, tr, fr, bar); return 0; }
int run_p02_l0b(void* i, int it, int tr, int fr, int bar, void* st, int bd) { p02_load_l0b<<<bd, nullptr, st>>>((uint8_t*)i, it, tr, fr, bar); return 0; }
int run_p03_fix_gm(void* o, int it, int m, int n, int bar, void* st, int bd) { p03_fixpipe_gm<<<bd, nullptr, st>>>((uint8_t*)o, it, m, n, bar); return 0; }
}
'''

P06_MMAD_CPP = r'''
#include "kernel_operator.h"
template <typename TA, typename TB, typename TC, int M, int N, int K, bool EN_HF32>
__aicore__ inline void p06_mmad_impl(GM_ADDR in_gm, int iters) {
    constexpr int A_BYTES = M * K * sizeof(TA);
    constexpr int B_BYTES = K * N * sizeof(TB);
    constexpr int C_BYTES = M * N * sizeof(TC);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> l1aBuf; AscendC::TBuf<AscendC::TPosition::B1> l1bBuf;
    AscendC::TBuf<AscendC::TPosition::A2> l0aBuf; AscendC::TBuf<AscendC::TPosition::B2> l0bBuf;
    AscendC::TBuf<AscendC::TPosition::CO1> l0cBuf;
    pipe.InitBuffer(l1aBuf, A_BYTES); pipe.InitBuffer(l1bBuf, B_BYTES);
    pipe.InitBuffer(l0aBuf, A_BYTES); pipe.InitBuffer(l0bBuf, B_BYTES);
    pipe.InitBuffer(l0cBuf, C_BYTES);
    auto l1a = l1aBuf.Get<half>(); auto l1b = l1bBuf.Get<half>();
    auto l0aH = l0aBuf.Get<half>(); auto l0bH = l0bBuf.Get<half>();
    AscendC::GlobalTensor<half> gA, gB;
    gA.SetGlobalBuffer((__gm__ half*)in_gm, A_BYTES / 2);
    gB.SetGlobalBuffer((__gm__ half*)in_gm, B_BYTES / 2);
    AscendC::DataCopy(l1a, gA, A_BYTES / 2); AscendC::DataCopy(l1b, gB, B_BYTES / 2);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
#if defined(_MINDIE_ARCH35)
    AscendC::LoadData2DParamsV2 pa, pb;
    pa.mStartPosition = 0; pa.kStartPosition = 0; pa.mStep = 4; pa.kStep = 8; pa.srcStride = 8; pa.dstStride = 8; pa.ifTranspose = false;
    pb.mStartPosition = 0; pb.kStartPosition = 0; pb.mStep = 8; pb.kStep = 8; pb.srcStride = 8; pb.dstStride = 8; pb.ifTranspose = false;
#else
    AscendC::LoadData2DParams pa, pb;
    pa.startIndex = 0; pa.repeatTimes = A_BYTES / 512; pa.srcStride = 1; pa.sid = 0; pa.dstGap = 0; pa.ifTranspose = false; pa.addrMode = 0;
    pb.startIndex = 0; pb.repeatTimes = B_BYTES / 512; pb.srcStride = 1; pb.sid = 0; pb.dstGap = 0; pb.ifTranspose = false; pb.addrMode = 0;
#endif
    AscendC::LoadData(l0aH, l1a, pa); AscendC::LoadData(l0bH, l1b, pb);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0); AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
    auto l0a = l0aH.ReinterpretCast<TA>(); auto l0b = l0bH.ReinterpretCast<TB>(); auto l0c = l0cBuf.Get<TC>();
    if constexpr (EN_HF32) { AscendC::SetHF32Mode(AscendC::HF32Mode::ENABLE); }
    AscendC::MmadParams mp;
    mp.m = M; mp.n = N; mp.k = K; mp.isBias = false; mp.cmatrixInitVal = true; mp.cmatrixSource = false;
    for (int i = 0; i < iters; ++i) { AscendC::Mmad(l0c, l0a, l0b, mp); }
    AscendC::PipeBarrier<PIPE_M>();
    if constexpr (EN_HF32) { AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE); }
}
extern "C" __global__ __cube__ void p06_mmad_bf16(GM_ADDR i, int it) { p06_mmad_impl<bfloat16_t, bfloat16_t, float, 64, 128, 128, false>(i, it); }
extern "C" __global__ __cube__ void p06_mmad_fp16(GM_ADDR i, int it) { p06_mmad_impl<half, half, float, 64, 128, 128, false>(i, it); }
extern "C" __global__ __cube__ void p06_mmad_fp32(GM_ADDR i, int it) { p06_mmad_impl<float, float, float, 64, 128, 64, false>(i, it); }
extern "C" __global__ __cube__ void p06_mmad_hf32(GM_ADDR i, int it) { p06_mmad_impl<float, float, float, 64, 128, 64, true>(i, it); }
extern "C" __global__ __cube__ void p06_mmad_int8(GM_ADDR i, int it) { p06_mmad_impl<int8_t, int8_t, int32_t, 64, 128, 256, false>(i, it); }
#if defined(_MINDIE_ARCH35)
template <typename TA, typename TB, int M, int N, int K, bool IS_FP4>
__aicore__ inline void p06_mmad_mx_impl(GM_ADDR in_gm, GM_ADDR mx_gm, int iters) {
    constexpr int A_BYTES = IS_FP4 ? (M * K / 2) : (M * K);
    constexpr int B_BYTES = IS_FP4 ? (K * N / 2) : (K * N);
    constexpr int C_BYTES = M * N * sizeof(float);
    constexpr int MX_BUF_BYTES = 16384;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> l1aBuf, l1aMxBuf;
    AscendC::TBuf<AscendC::TPosition::B1> l1bBuf, l1bMxBuf;
    AscendC::TBuf<AscendC::TPosition::A2> l0aBuf;
    AscendC::TBuf<AscendC::TPosition::B2> l0bBuf;
    AscendC::TBuf<AscendC::TPosition::CO1> l0cBuf;
    pipe.InitBuffer(l1aBuf, A_BYTES); pipe.InitBuffer(l1aMxBuf, MX_BUF_BYTES);
    pipe.InitBuffer(l1bBuf, B_BYTES); pipe.InitBuffer(l1bMxBuf, MX_BUF_BYTES);
    pipe.InitBuffer(l0aBuf, A_BYTES); pipe.InitBuffer(l0bBuf, B_BYTES);
    pipe.InitBuffer(l0cBuf, C_BYTES);
    auto l1aU8 = l1aBuf.Get<uint8_t>(); auto l1bU8 = l1bBuf.Get<uint8_t>();
    auto l1aMxU8 = l1aMxBuf.Get<uint8_t>(); auto l1bMxU8 = l1bMxBuf.Get<uint8_t>();
    AscendC::GlobalTensor<uint8_t> gIn, gMx;
    gIn.SetGlobalBuffer((__gm__ uint8_t*)in_gm, 65536);
    gMx.SetGlobalBuffer((__gm__ uint8_t*)mx_gm, MX_BUF_BYTES);
    AscendC::DataCopy(l1aU8, gIn, A_BYTES); AscendC::DataCopy(l1bU8, gIn, B_BYTES);
    AscendC::DataCopy(l1aMxU8, gMx, MX_BUF_BYTES); AscendC::DataCopy(l1bMxU8, gMx, MX_BUF_BYTES);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
    auto l1a = l1aU8.ReinterpretCast<TA>(); auto l1b = l1bU8.ReinterpretCast<TB>();
    auto l1aMx = l1aMxU8.ReinterpretCast<fp8_e8m0_t>(); auto l1bMx = l1bMxU8.ReinterpretCast<fp8_e8m0_t>();
    auto l0a = l0aBuf.Get<TA>(); auto l0b = l0bBuf.Get<TB>(); auto l0c = l0cBuf.Get<float>();
    AscendC::LoadData2DParamsV2 pa, pb;
    pa.mStartPosition = 0; pa.kStartPosition = 0; pa.mStep = 4; pa.kStep = 8; pa.srcStride = 8; pa.dstStride = 8; pa.ifTranspose = false;
    pb.mStartPosition = 0; pb.kStartPosition = 0; pb.mStep = 8; pb.kStep = 8; pb.srcStride = 8; pb.dstStride = 8; pb.ifTranspose = false;
    AscendC::LoadData2DMxParams pma, pmb;
    pma.xStartPosition = 0; pma.yStartPosition = 0; pma.xStep = 4; pma.yStep = 4; pma.srcStride = 4; pma.dstStride = 4;
    pmb.xStartPosition = 0; pmb.yStartPosition = 0; pmb.xStep = 4; pmb.yStep = 8; pmb.srcStride = 8; pmb.dstStride = 8;
    AscendC::LoadData(l0a, l1a, l1aMx, pa, pma); AscendC::LoadData(l0b, l1b, l1bMx, pb, pmb);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0); AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
    AscendC::MmadParams mp;
    mp.m = M; mp.n = N; mp.k = K; mp.isBias = false; mp.cmatrixInitVal = true; mp.cmatrixSource = false;
    for (int i = 0; i < iters; ++i) { AscendC::MmadMx(l0c, l0a, l0b, mp); }
    AscendC::PipeBarrier<PIPE_M>();
}
extern "C" __global__ __cube__ void p06_mmad_fp8e4(GM_ADDR i, int it) { p06_mmad_impl<fp8_e4m3fn_t, fp8_e4m3fn_t, float, 64, 128, 256, false>(i, it); }
extern "C" __global__ __cube__ void p06_mmad_fp8e5(GM_ADDR i, int it) { p06_mmad_impl<fp8_e5m2_t, fp8_e5m2_t, float, 64, 128, 256, false>(i, it); }
extern "C" __global__ __cube__ void p06_mmad_hif8(GM_ADDR i, int it) { p06_mmad_impl<hifloat8_t, hifloat8_t, float, 64, 128, 256, false>(i, it); }
extern "C" __global__ __cube__ void p06_mmad_mxfp8(GM_ADDR i, GM_ADDR mx, int it) { p06_mmad_mx_impl<fp8_e4m3fn_t, fp8_e4m3fn_t, 64, 128, 256, false>(i, mx, it); }
extern "C" __global__ __cube__ void p06_mmad_mxfp4(GM_ADDR i, GM_ADDR mx, int it) { p06_mmad_mx_impl<fp4x2_e2m1_t, fp4x2_e2m1_t, 64, 128, 512, true>(i, mx, it); }
#endif
extern "C" {
int run_p06_bf16(void* i, int it, void* st, int bd) { p06_mmad_bf16<<<bd, nullptr, st>>>((uint8_t*)i, it); return 0; }
int run_p06_fp16(void* i, int it, void* st, int bd) { p06_mmad_fp16<<<bd, nullptr, st>>>((uint8_t*)i, it); return 0; }
int run_p06_fp32(void* i, int it, void* st, int bd) { p06_mmad_fp32<<<bd, nullptr, st>>>((uint8_t*)i, it); return 0; }
int run_p06_hf32(void* i, int it, void* st, int bd) { p06_mmad_hf32<<<bd, nullptr, st>>>((uint8_t*)i, it); return 0; }
int run_p06_int8(void* i, int it, void* st, int bd) { p06_mmad_int8<<<bd, nullptr, st>>>((uint8_t*)i, it); return 0; }
#if defined(_MINDIE_ARCH35)
int run_p06_fp8e4(void* i, int it, void* st, int bd) { p06_mmad_fp8e4<<<bd, nullptr, st>>>((uint8_t*)i, it); return 0; }
int run_p06_fp8e5(void* i, int it, void* st, int bd) { p06_mmad_fp8e5<<<bd, nullptr, st>>>((uint8_t*)i, it); return 0; }
int run_p06_hif8(void* i, int it, void* st, int bd) { p06_mmad_hif8<<<bd, nullptr, st>>>((uint8_t*)i, it); return 0; }
int run_p06_mxfp8(void* i, void* mx, int it, void* st, int bd) { p06_mmad_mxfp8<<<bd, nullptr, st>>>((uint8_t*)i, (uint8_t*)mx, it); return 0; }
int run_p06_mxfp4(void* i, void* mx, int it, void* st, int bd) { p06_mmad_mxfp4<<<bd, nullptr, st>>>((uint8_t*)i, (uint8_t*)mx, it); return 0; }
#endif
}
'''

P10_SORT_CPP = r'''
#include "kernel_operator.h"
extern "C" __global__ __vector__ void p10_ub_sort32_mrg(GM_ADDR in_gm, GM_ADDR out_gm, int iters) {
    constexpr int N = 1024; constexpr int REPS = N / 32;
    AscendC::TPipe pipe; AscendC::TBuf<AscendC::TPosition::VECCALC> bVal, bIdx, bA, bB;
    pipe.InitBuffer(bVal, N * sizeof(float)); pipe.InitBuffer(bIdx, N * sizeof(int32_t));
    pipe.InitBuffer(bA, 2 * N * sizeof(float)); pipe.InitBuffer(bB, 2 * N * sizeof(float));
    auto vals = bVal.Get<float>(); auto idxs = bIdx.Get<int32_t>();
    auto bufA = bA.Get<float>(); auto bufB = bB.Get<float>();
    AscendC::GlobalTensor<float> gIn, gOut;
    gIn.SetGlobalBuffer((__gm__ float*)in_gm + AscendC::GetBlockIdx() * N, N);
    gOut.SetGlobalBuffer((__gm__ float*)out_gm + AscendC::GetBlockIdx() * N, N);
    AscendC::DataCopy(vals, gIn, N);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::CreateVecIndex(idxs, 0, N); AscendC::PipeBarrier<PIPE_V>();
    auto uidxs = idxs.ReinterpretCast<uint32_t>();
    for (int it = 0; it < iters; ++it) {
        AscendC::Sort32(bufA, vals, uidxs, REPS); AscendC::PipeBarrier<PIPE_V>();
        for (int g = 0; g < 8; ++g) {
            AscendC::MrgSort4Info p;
            p.elementLengths[0] = 32; p.elementLengths[1] = 32; p.elementLengths[2] = 32; p.elementLengths[3] = 32;
            p.ifExhaustedSuspension = false; p.validBit = 0xF; p.repeatTimes = 1;
            AscendC::MrgSortSrcList<float> s; int base = g * 256;
            s.src1 = bufA[base]; s.src2 = bufA[base + 64]; s.src3 = bufA[base + 128]; s.src4 = bufA[base + 192];
            AscendC::MrgSort<float>(bufB[base], s, p);
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (int g = 0; g < 2; ++g) {
            AscendC::MrgSort4Info p;
            p.elementLengths[0] = 128; p.elementLengths[1] = 128; p.elementLengths[2] = 128; p.elementLengths[3] = 128;
            p.ifExhaustedSuspension = false; p.validBit = 0xF; p.repeatTimes = 1;
            AscendC::MrgSortSrcList<float> s; int base = g * 1024;
            s.src1 = bufB[base]; s.src2 = bufB[base + 256]; s.src3 = bufB[base + 512]; s.src4 = bufB[base + 768];
            AscendC::MrgSort<float>(bufA[base], s, p);
        }
        AscendC::PipeBarrier<PIPE_V>();
        {
            AscendC::MrgSort4Info p;
            p.elementLengths[0] = 512; p.elementLengths[1] = 512; p.elementLengths[2] = 0; p.elementLengths[3] = 0;
            p.ifExhaustedSuspension = false; p.validBit = 0x3; p.repeatTimes = 1;
            AscendC::MrgSortSrcList<float> s;
            s.src1 = bufA[0]; s.src2 = bufA[1024]; s.src3 = bufA[0]; s.src4 = bufA[0];
            AscendC::MrgSort<float>(bufB[0], s, p);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::DataCopy(gOut, bufB, N);
}
extern "C" int run_p10_sort(void* i, void* o, int it, void* st, int bd) { p10_ub_sort32_mrg<<<bd, nullptr, st>>>((uint8_t*)i, (uint8_t*)o, it); return 0; }
'''

DMA_HW_CPP = r'''
#include "kernel_operator.h"
extern "C" __global__ __vector__ void dma_stream_read_aligned(
    GM_ADDR src_gm, GM_ADDR sink_gm, int64_t total_bytes, int byte_offset, int chunk_bytes, int reps
) {
    AscendC::TPipe pipe; AscendC::TBuf<AscendC::TPosition::VECCALC> b0, b1;
    pipe.InitBuffer(b0, 32768); pipe.InitBuffer(b1, 32768);
    auto u0 = b0.Get<uint8_t>(); auto u1 = b1.Get<uint8_t>();
    int bid = AscendC::GetBlockIdx(); int bnum = AscendC::GetBlockNum();
    int64_t per_core = ((total_bytes - byte_offset) / bnum / chunk_bytes) * chunk_bytes;
    AscendC::GlobalTensor<uint8_t> gSrc;
    gSrc.SetGlobalBuffer((__gm__ uint8_t*)src_gm + byte_offset + (int64_t)bid * per_core, per_core);
    for (int r = 0; r < reps; ++r) {
        for (int64_t k = 0; k < per_core / chunk_bytes; ++k) {
            AscendC::DataCopy((k & 1) ? u1 : u0, gSrc[k * chunk_bytes], chunk_bytes);
        }
    }
    AscendC::PipeBarrier<PIPE_MTE2>();
}
extern "C" __global__ __cube__ void dma_aic_mte2_stream(
    GM_ADDR src_gm, int64_t total_bytes, int chunk_bytes, int reps
) {
    AscendC::TPipe pipe; AscendC::TBuf<AscendC::TPosition::A1> b0, b1;
    pipe.InitBuffer(b0, 65536); pipe.InitBuffer(b1, 65536);
    auto l0 = b0.Get<uint8_t>(); auto l1 = b1.Get<uint8_t>();
    int bid = AscendC::GetBlockIdx(); int bnum = AscendC::GetBlockNum();
    int64_t per_core = (total_bytes / bnum / chunk_bytes) * chunk_bytes;
    AscendC::GlobalTensor<uint8_t> gSrc;
    gSrc.SetGlobalBuffer((__gm__ uint8_t*)src_gm + (int64_t)bid * per_core, per_core);
    for (int r = 0; r < reps; ++r) {
        for (int64_t k = 0; k < per_core / chunk_bytes; ++k) {
            AscendC::DataCopy((k & 1) ? l1 : l0, gSrc[k * chunk_bytes], chunk_bytes);
        }
    }
    AscendC::PipeBarrier<PIPE_MTE2>();
}
extern "C" __global__ __vector__ void dma_2d_strided_read(
    GM_ADDR src_gm, int total_rows, int row_bytes, int stride_bytes, int rows_per_burst, int reps
) {
    AscendC::TPipe pipe; AscendC::TBuf<AscendC::TPosition::VECCALC> b0, b1;
    pipe.InitBuffer(b0, 65536); pipe.InitBuffer(b1, 65536);
    auto u0 = b0.Get<uint8_t>(); auto u1 = b1.Get<uint8_t>();
    int bid = AscendC::GetBlockIdx(); int bnum = AscendC::GetBlockNum();
    int rows_core = (total_rows / bnum / rows_per_burst) * rows_per_burst;
    int row_start = bid * rows_core;
    AscendC::DataCopyParams p;
    p.blockCount = static_cast<uint16_t>(rows_per_burst);
    p.blockLen = static_cast<uint16_t>(row_bytes / 32);
    p.srcStride = static_cast<uint16_t>((stride_bytes - row_bytes) / 32);
    p.dstStride = 0;
    AscendC::GlobalTensor<uint8_t> gSrc;
    gSrc.SetGlobalBuffer((__gm__ uint8_t*)src_gm, (int64_t)total_rows * stride_bytes);
    int bursts = rows_core / rows_per_burst;
    for (int rep = 0; rep < reps; ++rep) {
        for (int b = 0; b < bursts; ++b) {
            int64_t off = (int64_t)(row_start + b * rows_per_burst) * stride_bytes;
            AscendC::DataCopy((b & 1) ? u1 : u0, gSrc[off], p);
        }
    }
    AscendC::PipeBarrier<PIPE_MTE2>();
}
extern "C" {
int run_dma_align(void* s, void* k, long long tb, int off, int cb, int reps, void* st, int bd) {
    dma_stream_read_aligned<<<bd, nullptr, st>>>((uint8_t*)s, (uint8_t*)k, tb, off, cb, reps); return 0;
}
int run_aic_mte2(void* s, long long tb, int cb, int reps, void* st, int bd) {
    dma_aic_mte2_stream<<<bd, nullptr, st>>>((uint8_t*)s, tb, cb, reps); return 0;
}
int run_dma_2d(void* s, int nr, int rb, int sb, int rpb, int reps, void* st, int bd) {
    dma_2d_strided_read<<<bd, nullptr, st>>>((uint8_t*)s, nr, rb, sb, rpb, reps); return 0;
}
}
'''

ARCH35_SIMT_CPP = r'''
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
constexpr uint32_t SIMT_THREADS = 512;
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREADS) inline void simt_alu_vf(__gm__ float* out, int iters) {
    uint32_t tid = AscendC::Simt::GetThreadIdx();
    float a0 = 1.0001f + tid * 1e-5f; float a1 = 1.0002f + tid * 1e-5f;
    float a2 = 1.0003f + tid * 1e-5f; float a3 = 1.0004f + tid * 1e-5f;
    for (int i = 0; i < iters; ++i) {
        a0 = a0 * 0.99999f + 0.00001f; a1 = a1 * 0.99999f + 0.00001f;
        a2 = a2 * 0.99999f + 0.00001f; a3 = a3 * 0.99999f + 0.00001f;
    }
    out[AscendC::Simt::GetBlockIdx() * SIMT_THREADS + tid] = a0 + a1 + a2 + a3;
}
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREADS) inline void simt_bar_vf(__gm__ float* out, int iters) {
    uint32_t tid = AscendC::Simt::GetThreadIdx();
    for (int i = 0; i < iters; ++i) { AscendC::Simt::ThreadBarrier(); }
    if (tid == 0) out[AscendC::Simt::GetBlockIdx()] = (float)iters;
}
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREADS) inline void simt_empty_vf(__gm__ float* out, int tag) {
    if (AscendC::Simt::GetThreadIdx() == 0) out[AscendC::Simt::GetBlockIdx()] = (float)tag;
}
extern "C" __global__ __vector__ void p11_simt_alu(GM_ADDR out_gm, int iters) {
    AscendC::Simt::VF_CALL<simt_alu_vf>(AscendC::Simt::Dim3(SIMT_THREADS), (__gm__ float*)out_gm, iters);
}
extern "C" __global__ __vector__ void p11_simt_bar(GM_ADDR out_gm, int iters) {
    AscendC::Simt::VF_CALL<simt_bar_vf>(AscendC::Simt::Dim3(SIMT_THREADS), (__gm__ float*)out_gm, iters);
}
extern "C" __global__ __vector__ void p11_simt_launch(GM_ADDR out_gm, int iters) {
    for (int i = 0; i < iters; ++i) {
        AscendC::Simt::VF_CALL<simt_empty_vf>(AscendC::Simt::Dim3(SIMT_THREADS), (__gm__ float*)out_gm, i);
    }
}
extern "C" {
int run_p11_alu(void* o, int it, void* st, int bd) { p11_simt_alu<<<bd, nullptr, st>>>((uint8_t*)o, it); return 0; }
int run_p11_bar(void* o, int it, void* st, int bd) { p11_simt_bar<<<bd, nullptr, st>>>((uint8_t*)o, it); return 0; }
int run_p11_launch(void* o, int it, void* st, int bd) { p11_simt_launch<<<bd, nullptr, st>>>((uint8_t*)o, it); return 0; }
}
'''
