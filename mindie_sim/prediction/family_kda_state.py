"""Explicit source selection for the KDA prefill FWD_H state phase.

[Role] Bind ``runtime.kernels.lowerings.kda`` to the retained vLLM-Ascend
  ``chunk_kda_fwd`` FWD_H stage without changing the KDA semantic boundary.
[Invariants] Selection is by implementation name, never model label or measured
  duration.  Only the prefill ``recurrent_scan`` member is replaced; decode,
  caller gate placement and QK normalization retain their existing behavior.
[Timeouts & Locks] Pure metadata validation; no I/O or locks.
[Side-Effect Policy] No device work, source reads or global mutation.
"""
from mindie_sim.schema.common import ModelingError
from .mechanism import PlannedRecipe
from .operators import kda_state

IMPLEMENTATION = kda_state.IMPLEMENTATION
CAPABILITIES = frozenset({"prefill_fwd_h"})
SOURCE = {
    "repository": "vllm-ascend",
    "revision": "ae923905948f360f4b7ab06df52d0789a0c9316e",
    "paths": [
        "csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd_common.h",
        "csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd.cpp",
        "csrc/attention/chunk_kda_fwd/op_host/chunk_kda_fwd_tiling.cpp",
        "csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp",
        "csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/block/block_scheduler_gdn_fwd_h.hpp",
        "csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp",
        "csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/epilogue/block/block_epilogue_gdn_fwdh_update.hpp",
        "csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/epilogue/block/block_epilogue_gdn_fwdh_regbase.hpp",
        "vllm_ascend/ops/kimi_kda.py",
    ],
    "sha256": {
        "csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd_common.h":
            "e98c17e528d673a9b6c2f0bbbfce83746bcccf0afaeb413ea0ac41a2809e74f8",
        "csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd.cpp":
            "0ea787730736f4ed92a35e3bafa6e54417c84716fc6aed381a339259abfa74d0",
        "csrc/attention/chunk_kda_fwd/op_host/chunk_kda_fwd_tiling.cpp":
            "72148ca10b26ba4f34919c11e973c3bbcd696cfbc9a938c3cc3bf724e41f813c",
        "csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp":
            "aa9695e2d9eba0dc0963d248199a86ab5cbd321c0f5836a9dd8ee8901944cf13",
        "csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/block/block_scheduler_gdn_fwd_h.hpp":
            "3e7730a090795623af026efea6a8484238d1fd02e301e286936f1b4bcd8ba741",
        "csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp":
            "8ec011c4ff4adaa0398ba77e2b5cbc331335c3f6c44d00921e9cdbcaa953dd02",
        "csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/epilogue/block/block_epilogue_gdn_fwdh_update.hpp":
            "67dbb475b2ae630bb53fbdb9bfe44af774c0be8b9da7c16074b5d1ef6d88fedb",
        "csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/epilogue/block/block_epilogue_gdn_fwdh_regbase.hpp":
            "c2221dbb88f92fcf04adde0a0799264bd53c6e1f74033ed3bee634fb9daac4cc",
        "vllm_ascend/ops/kimi_kda.py":
            "35365b3b353fc492b4dc5d75b88916431f6802c21bf51b61b4893d39d54f11f3",
    },
    "symbols": [
        "KdaForward::RunFwdH",
        "KdaForward::DispatchStage(KDA_STAGE_FWD_H)",
        "Catlass::Gemm::Kernel::GDNFwdHKernel::Process",
        "Catlass::Gemm::Block::BlockSchedulerGdnFwdH",
        "KimiGatedDeltaNetAttention._run_prefill",
    ],
    "lines": {
        "chunk_kda_fwd_common.h": "88-111,225-241",
        "chunk_kda_fwd.cpp": "55-64",
        "chunk_kda_fwd_tiling.cpp": "219-254,274-293",
        "gdn_fwd_h_kernel.hpp": "508-1067",
        "block_scheduler_gdn_fwd_h.hpp": "214-225,300-353",
        "block_epilogue_gdn_fwdh_vnew.hpp": "365-418",
        "block_epilogue_gdn_fwdh_update.hpp": "292-453",
        "block_epilogue_gdn_fwdh_regbase.hpp": "53-212",
        "kimi_kda.py": "468-519",
    },
    "scope": "source_selected_FWD_H_phase_not_full_call_not_capture_binary_identity",
    "source_state": "committed",
}


def recipes():
    return {kda_state.RECIPE: PlannedRecipe(kda_state.build)}


def validate(*, key_heads, value_heads, key_width, value_width, model):
    if key_heads != value_heads:
        raise ModelingError("kda_state_source_requires_equal_key_value_heads")
    if key_width != 128 or value_width != 128:
        raise ModelingError("kda_state_source_requires_dk_dv_128")
    if model.activation_dtype != "bf16":
        raise ModelingError("kda_state_source_requires_bf16_inputs")


def bind_prefill_scans(scans, *, lengths, heads, key_heads, dk, dv, chunk,
                       dtype, state_dtype, requests, pcp=1, speculative=False):
    """Replace the prefill scan members with one actual varlen FWD_H op."""
    scans = tuple(scans)
    if not scans:
        raise ModelingError("kda_state_prefill_scan_missing")
    if pcp != 1:
        raise ModelingError("kda_state_requires_pcp1")
    if speculative:
        raise ModelingError("kda_state_speculative_prefill_not_supported")
    params = {
        "lengths": tuple(lengths), "heads": heads, "key_heads": key_heads,
        "dk": dk, "dv": dv, "chunk": chunk, "dtype": dtype,
        "gate_dtype": "fp32", "state_dtype": state_dtype,
        "use_initial_state": True, "output_final_state": True,
        "state_v_first": True, "use_exp2": True, "save_new_value": True,
    }
    first = scans[0]
    deps = tuple(dict.fromkeys(dep for scan in scans for dep in scan.deps))
    return kda_state.make_op(
        params, op_id=first.op_id, deps=deps, requests=requests,
        stage=first.stage, layer=first.layer, phase=first.phase,
        stream=first.stream, notes=first.notes,
    )


__all__ = [
    "CAPABILITIES", "IMPLEMENTATION", "SOURCE", "bind_prefill_scans",
    "recipes", "validate",
]
