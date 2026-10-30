"""Explicit source selection for the KDA GATE_PREPARE and FWD_H stages.

[Role] Bind ``runtime.kernels.lowerings.kda`` to the retained vLLM-Ascend
  ``chunk_kda_fwd`` stage 0 (BF16 gate cumsum + native prepare) and stage 2
  (FWD_H), without changing the KDA semantic boundary.
[Invariants] Selection is by implementation name, never model label, tile
  geometry or measured duration.  GATE_PREPARE absorbs the generic prefill
  ``delta_gates`` and local score/solve exactly once; the external beta sigmoid
  and generic Post-WU stay separate.  Decode is not certified by this
  prefill-only implementation.  Existing selections remain available.
[Timeouts & Locks] Pure metadata validation; no I/O or locks.
[Side-Effect Policy] No device work, source reads or global mutation.
"""
from mindie_sim.schema.common import ModelingError
from .mechanism import PlannedRecipe
from . import family_kda_state
from .operators import kda_prepare, kda_state

IMPLEMENTATION = kda_prepare.IMPLEMENTATION
CAPABILITIES = frozenset({"prefill_gate_prepare", "prefill_fwd_h", "in_kernel_gates"})
SOURCE = {
    "repository": "vllm-ascend",
    "revision": "ae923905948f360f4b7ab06df52d0789a0c9316e",
    "paths": [
        "csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd.cpp",
        "csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd_common.h",
        "csrc/attention/kda_gate_cumsum/op_kernel/kda_gate_cumsum_kernel.h",
        "csrc/attention/chunk_kda_fwd/op_kernel/arch35/chunk_kda_fwd_prepare.h",
        "csrc/attention/chunk_kda_fwd/op_host/chunk_kda_fwd_tiling.cpp",
        "csrc/attention/chunk_kda_fwd/op_host/arch35/chunk_kda_fwd_tiling_impl.h",
        "vllm_ascend/ops/kimi_kda.py",
    ],
    "sha256": {
        "csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd.cpp":
            "0ea787730736f4ed92a35e3bafa6e54417c84716fc6aed381a339259abfa74d0",
        "csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd_common.h":
            "e98c17e528d673a9b6c2f0bbbfce83746bcccf0afaeb413ea0ac41a2809e74f8",
        "csrc/attention/kda_gate_cumsum/op_kernel/kda_gate_cumsum_kernel.h":
            "8abaec8e5cec1344e2fe8047271816115b20cbcf41463dc69195ae859f96162e",
        "csrc/attention/chunk_kda_fwd/op_kernel/arch35/chunk_kda_fwd_prepare.h":
            "b808a1a1c6e21c0a3098433764675a1f507b04bc8f77d22eaffedd2c4de866c7",
        "csrc/attention/chunk_kda_fwd/op_host/chunk_kda_fwd_tiling.cpp":
            "72148ca10b26ba4f34919c11e973c3bbcd696cfbc9a938c3cc3bf724e41f813c",
        "csrc/attention/chunk_kda_fwd/op_host/arch35/chunk_kda_fwd_tiling_impl.h":
            "e7542eff7113d97ce069b7ec4f57b01809f4894f1e3b5e801add5f557991fe59",
    },
    "symbols": [
        "KdaForward::DispatchStage(KDA_STAGE_GATE_PREPARE)",
        "KdaGateCumsum::DispatchKdaGateCumsum<bfloat16_t>",
        "KdaPrepare::RunChunkKdaPrepare",
        "KdaPrepare::ChunkKdaFwdPrepareKernel::ProcessPreAivHeadPair",
        "KdaPrepare::ChunkKdaFwdPrepareKernel::ProcessPreAicHeadPair",
        "KdaPrepare::ComputeAkkMergeCubeWorkspace",
        "KdaPrepare::ForwardSubDiag16StridedRegbase",
        "KimiGatedDeltaNetAttention._run_prefill",
    ],
    "lines": {
        "chunk_kda_fwd.cpp": "15-19,35-46",
        "chunk_kda_fwd_common.h": "160-199",
        "kda_gate_cumsum_kernel.h": "180-207,355-540",
        "chunk_kda_fwd_prepare.h": "60-120,2936-3070,3505-3545,3722-3755,"
                                  "4009-4030,4378-4478,4636-4710,4731-4826",
        "chunk_kda_fwd_tiling.cpp": "219-293,324-354",
    },
    "scope": "source_selected_GATE_PREPARE_plus_FWD_H_not_full_call_not_capture_identity",
    "source_state": "committed",
}


def recipes():
    return {
        kda_prepare.RECIPE: PlannedRecipe(kda_prepare.build),
        kda_state.RECIPE: PlannedRecipe(kda_state.build),
    }


def validate(*, key_heads, value_heads, key_width, value_width, model):
    if key_heads != value_heads:
        raise ModelingError("kda_prepare_requires_equal_key_value_heads")
    if key_width != 128 or value_width != 128:
        raise ModelingError("kda_prepare_requires_dk_dv_128")
    if model.activation_dtype != "bf16":
        raise ModelingError("kda_prepare_requires_bf16_inputs")


def bind_gate_prepare(ops, *, gate, local_ops, lengths, heads, key_heads, dk, dv,
                      chunk, dtype, raw_g_dtype="bf16", gate_dtype="fp32",
                      beta_dtype="fp32", requests=(), pcp=1,
                      speculative=False):
    return kda_prepare.bind_gate_prepare(
        ops, gate=gate, local_ops=local_ops, lengths=lengths, heads=heads,
        key_heads=key_heads, dk=dk, dv=dv, chunk=chunk, dtype=dtype,
        raw_g_dtype=raw_g_dtype, gate_dtype=gate_dtype,
        beta_dtype=beta_dtype, requests=requests, pcp=pcp,
        speculative=speculative)


def bind_prefill_scans(scans, *, lengths, heads, key_heads, dk, dv, chunk,
                       dtype, state_dtype, requests, pcp=1, speculative=False):
    return family_kda_state.bind_prefill_scans(
        scans, lengths=lengths, heads=heads, key_heads=key_heads, dk=dk,
        dv=dv, chunk=chunk, dtype=dtype, state_dtype=state_dtype,
        requests=requests, pcp=pcp, speculative=speculative)


__all__ = [
    "CAPABILITIES", "IMPLEMENTATION", "SOURCE", "bind_gate_prepare",
    "bind_prefill_scans", "recipes", "validate",
]
