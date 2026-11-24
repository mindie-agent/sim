"""Source-selected Kimi AttnRes cross-layer prefix mechanism.

[Role] Build one native AttnResFwd point per source residual point and price the
  retained AIV-only kernel on the common engine.
[Invariants] The retained caller owns the state transition: a block boundary
  writes slot ``layer // B`` and clears the live prefix before attention; every
  FFN output stays pending until the next point or final point. Bank
  capacity and physical token stride stay distinct. Resident versus reload uses
  the retained tiling formula over declared UB, never a device or model name.
  No Cube GEMM or FP32-v user workspace is created. Selection is explicit and
  an unselected graph is untouched.
[Timeouts & Locks] Finite local arithmetic and per-core programs; no I/O, locks,
  device contexts or measurement reads.
[Side-Effect Policy] Returns immutable plans/records; source files are cited by
  hash, never read.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

from mindie_sim.backends.abi_registry import default_abi_registry
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import (
    KernelBinding,
    OpSpec,
    OwnerMemoryResidency,
    OwnerScheduleSpec,
    SerialPhaseSpec,
    TensorPrecisionABI,
)

from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.ownership import read_role, unit_group
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from mindie_sim.prediction.structure_groups import declare_group
from mindie_sim.primitives.loader import P12_FLAG_CYCLES
from .common import (
    ATTNRES_CALLER_SOURCES,
    ATTNRES_FP32_ALIGNED_SCRATCH_ROWS,
    ATTNRES_FP32_SCRATCH_ROWS,
    ATTNRES_SOFTMAX_OPS_PER_ROW,
    FP32_BYTES,
    attnres_spec_terms,
)


IMPLEMENTATION = "vllm_ascend_attn_res_prefix"
RECIPE = "attn_res_fwd"
NATIVE_KIND = "AttnResFwd"
GROUP_ROLE = "attn_res_prefix_point"

CALLER_SHA256 = "be9adc792774f919f443caf483eb2e04e4453617465381f47a482dd4736ec63f"
SOURCE = {
    "repository": "vllm-ascend",
    "caller_sha256": CALLER_SHA256,
    "caller": {
        "path": "vllm_ascend/models/kimi_k3.py",
        "lines": "650-830,1030-1150",
        "sha256": CALLER_SHA256,
    },
    "operator": {
        "common_sha256": "f8abf2be43c0e0a70a19df4d4d1cdab518dc939fee6612aabcaf1f6e090e53a2",
        "tiling_sha256": "3249e4f252af0c68187b8caa7a00f63592a42c231fefecc28b7b1e7609b60669",
        "resident_sha256": "bb5a40a14e3d5637ab9deff26b77e8e3a305500f9c582a3274d7bf0b23840d94",
        "reload_sha256": "e346cd0db6eeedcfffaf911a07f6ddf4e34475c69677f73cb9e279f39641f070",
        "path": "vllm_ascend/csrc/attention/attn_res_fwd",
    },
    "scope": "source_compatible_declaration; not_capture_binary_identity",
    "capture_flags_proven": False,
}

CAPABILITIES = frozenset({
    "A5", "decode", "aiv_only", "attn_res_prefix", "source_selected",
    "explicit_prefill_cache", "banked_cross_layer_state",
})


def capabilities():
    """Machine-readable source domain and dynamic attributes of AttnRes."""
    return {
        "implementation": IMPLEMENTATION,
        "selector": "attention_residual",
        "recipe": RECIPE,
        "native_kind": NATIVE_KIND,
        "source_boundary": "kernel",
        "soc_families": ("A5",),
        "auto_phases": ("decode",),
        "explicit_phases": ("decode", "prefill"),
        "dtypes": ("bf16",),
        "dynamic_attributes": (
            "tokens", "tokens_local", "valid_blocks", "bank_capacity",
            "block_token_stride", "fuse_add", "mix", "materialize",
            "output_norm", "optimize_prefill", "prefix_id", "bank_ids",
        ),
        "group_role": GROUP_ROLE,
        "source": SOURCE,
        "limitations": LIMITATIONS,
    }

LIMITATIONS = (
    "capture_library_and_scalar_flags_not_proven",
    "strict_descriptor_contract_not_derived_from_csv",
    "per_owner_load_compute_store_phases_are_serial",
    "cross_core_overlap_is_shared_rate_service_not_a_compiler_schedule",
    "pp1_source_caller_domain",
    "sequence_parallel_ag_rs_and_tokens_local_are_caller_supplied",
    "semantic_vector_lanes_not_isa_latency_or_special_function_throughput",
    "aux_materialization_is_explicit_not_inferred",
    "no_measured_duration_or_bandwidth_fit_is_used",
)

UB_OVERHEAD_BYTES = 32 * 1024           # TPipe/alignment/queue allowance
SYSTEM_WORKSPACE_BYTES = 16 * 1024 * 1024  # system reservation, not user tensor
ELEM_BF16_BLOCK = 16
ELEM_FP32_BLOCK = 8
SCALAR_LOCAL_ELEMS = 8
_BF16_BYTES = 2
_RELOAD_BF16_BUFFER_ROWS = 3
_FUSED_ADD_OPS_PER_ELEMENT = 4
_SCORE_OPS_PER_ELEMENT = 6
_WEIGHTED_MIX_OPS_PER_ELEMENT = 3
_PROJECTION_SETUP_OPS_PER_ELEMENT = 3
_OUTPUT_NORM_OPS_PER_ELEMENT = 7
_OUTPUT_NORM_SCALAR_OPS = 3
ATTNRES_TREE_GROUP_LANES = 64
ATTNRES_REGBASE_GROUP_ELEMS = 512
ATTNRES_CAST_BYTES_PER_ELEMENT = 6
ATTNRES_SFU_STRIP_OPS = 4
#: Integer AIV vector-strip reduction stall cycles per 64-lane FP32 strip
#: paired with calibrated ``P12_intracore_flag`` sync latency (zero duration fit).
ATTNRES_RESIDENT_STRIP_STALL_CYCLES = 3
ATTNRES_RELOAD_STRIP_STALL_CYCLES = 5
ATTNRES_RELOAD_PIPELINE_SLOTS = 2
ATTNRES_COOP_CORE_GAIN = 2
ATTNRES_COOP_LOW_OCCUPANCY_DIV = 4
ATTNRES_COOP_MIN_HIDDEN = 512
ATTNRES_COOP_MAX_ROWS = 128
ATTNRES_COOP_WHOLE_BF16_ROWS = 4
ATTNRES_COOP_FP32_ROWS = 3
ATTNRES_COOP_SCALAR_SLOTS = 2
ATTNRES_COOP_HIDDEN_ALIGN = 64
ATTNRES_COOP_SLOT_BYTES = 32
MS_PER_SECOND = 1000.0
NS_PER_MS = 1_000_000.0
NS_PER_SECOND = 1_000_000_000.0


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _align(value, multiple):
    return _ceil_div(value, multiple) * multiple


def _attnres_arch_supported(arch: str) -> bool:
    """Verify ``AttnResFwd`` availability via sealed ``ProbedAbiRegistry`` (fail-closed)."""
    norm = str(arch or "").strip().upper()
    if norm not in {"A2", "A3", "A5"}:
        return False
    return default_abi_registry().is_shipping_available("AttnResFwd", norm)


def source_eligible(model, arch, phase):
    """Shared source-domain gate for target graph and ordinary serving."""
    if not _attnres_arch_supported(arch) or phase != "decode":
        return False
    if model.activation_dtype != "bf16":
        return False
    blocks = {p.residual_block_size for p in model.layer_plans}
    if len(blocks) != 1 or next(iter(blocks), 0) < 1:
        return False
    return not any(p.engram or p.hyper_connections for p in model.layer_plans)


def source_image_match(runtime):
    """LEGACY_PURGED: container image_id routing gate prohibited by DEV-READONLY-PATH."""
    raise NotImplementedError("LEGACY_PURGED: wire ProbedAbiRegistry ArchId gating without container image_id")


def explicit_generic(kernels):
    recipes = {"attn_res_fwd", "residual_values_fp32", "rmsnorm",
               "attention_residual_scores", "attention_residual_softmax",
               "attention_residual_mix", "attention_residual_output_cast"}
    families = {"gemm", "rmsnorm", "softmax", "elementwise", "cast"}
    return (bool(recipes & set(kernels.get("recipes", ())))
            or bool(families & set(kernels.get("families", ())))
            or "attn_res_fwd" in kernels.get("programs", ())
            or "recurrent_vector" in kernels.get("programs", ()))


def selection(runtime, *, model=None, phase=None, machine=None):
    """Return explicit or source-backed options, or None.

    Explicit source binding wins on A5 (or synthetic arch-unspecified fixtures)
    and is rejected on non-A5 hardware (A2/A3) so arch35 kernels are never
    priced on A2/A3.  Auto selection requires the shared A5 decode domain.
    """
    bindings = runtime.kernels.get("lowerings", {})
    arch = str(
        (getattr(machine, "arch_family", "") if machine is not None else "")
        or dict(runtime.metadata).get("arch_family")
        or ""
    ).strip().upper()
    if "attention_residual" in bindings:
        if arch and arch not in {"GENERIC", "SYNTHETIC"} and not _attnres_arch_supported(arch):
            return None
        from mindie_sim.prediction.implementations import select

        selected = select(bindings, "attention_residual")
        if selected.name != IMPLEMENTATION:
            return None
        value = selected.options.get("optimize_prefill", False)
        if not isinstance(value, bool):
            raise ModelingError("attn_res_optimize_prefill_must_be_boolean")
        return {"optimize_prefill": value}
    if model is None or phase is None or explicit_generic(runtime.kernels):
        return None
    if not source_eligible(model, arch, phase):
        return None
    return {"optimize_prefill": False}


def bank_capacity(model):
    block = next((plan.residual_block_size for plan in model.layer_plans
                  if plan.residual_block_size), 0)
    if block < 1:
        raise ModelingError("attn_res_requires_residual_block_size")
    end = max((plan.index for plan in model.layer_plans), default=-1) + 1
    return _ceil_div(end, block)


def block_token_stride(model, capacity):
    """The source caller allocates a contiguous [T, capacity, H] bank."""
    return capacity * model.hidden_size


def validate_source_shapes(*, hidden, prefix_width, blocks_width, proj_width,
                          norm_width, addend_width, output_norm_width):
    """Real six-operand AttnResFwd shape contract for the retained source."""
    if prefix_width != hidden:
        raise ModelingError("attn_res_source_prefix_width_mismatch")
    if blocks_width < hidden or blocks_width % hidden:
        raise ModelingError("attn_res_source_blocks_width_mismatch")
    if proj_width != hidden or norm_width != hidden:
        raise ModelingError("attn_res_source_weight_width_mismatch")
    if addend_width not in (None, hidden):
        raise ModelingError("attn_res_source_addend_width_mismatch")
    if output_norm_width not in (None, hidden):
        raise ModelingError("attn_res_source_output_norm_width_mismatch")


def max_resident_rows(hidden, ub_bytes, fuse_add=True):
    """Exact retained CalcMaxResidentRows over a declared UB byte capacity."""
    if hidden < 1 or ub_bytes < 1:
        return 0
    work = (ATTNRES_FP32_SCRATCH_ROWS * hidden * FP32_BYTES
            + 2 * hidden * _BF16_BYTES + UB_OVERHEAD_BYTES
            + (hidden * _BF16_BYTES if fuse_add else 0))
    if work >= ub_bytes:
        return 0
    return (ub_bytes - work) // (hidden * _BF16_BYTES + FP32_BYTES)


def estimate_ub_bytes(block_count, hidden, resident, fuse_add=True):
    """Retained EstimateUbComputeBytes for the active buffer geometry."""
    h_bf16, h_fp32 = _align(hidden, ELEM_BF16_BLOCK), _align(hidden, ELEM_FP32_BLOCK)
    meta = _align(block_count, ELEM_FP32_BLOCK)
    total = 0
    if resident:
        total += 2 * h_bf16 * _BF16_BYTES + block_count * h_bf16 * _BF16_BYTES
    else:
        total += _RELOAD_BF16_BUFFER_ROWS * h_bf16 * _BF16_BYTES
    total += ATTNRES_FP32_ALIGNED_SCRATCH_ROWS * h_fp32 * FP32_BYTES
    total += 2 * meta * FP32_BYTES
    total += meta * ELEM_FP32_BLOCK * FP32_BYTES
    total += SCALAR_LOCAL_ELEMS * FP32_BYTES
    if fuse_add:
        total += h_bf16 * _BF16_BYTES
    return total + UB_OVERHEAD_BYTES


def _cache_output_norm(view, ub_bytes):
    if not (view["optimize_prefill"] and view["output_norm"]):
        return False
    if view["tokens_per_core"] <= 1:
        return False
    fit = estimate_ub_bytes(view["valid_blocks"] + 1, view["hidden"],
                            view["resident"], True)
    return fit + _align(view["hidden"], ELEM_BF16_BLOCK) * 2 <= ub_bytes


def _validate_bank(bank_ids, capacity, valid, write_slot, mix):
    if len(bank_ids) != capacity:
        raise ModelingError("attn_res_bank_ids_require_capacity_slots")
    if valid < 0 or valid > capacity:
        raise ModelingError("attn_res_valid_blocks_out_of_capacity")
    if write_slot is not None:
        if isinstance(write_slot, bool) or not isinstance(write_slot, int):
            raise ModelingError("attn_res_write_slot_invalid")
        if write_slot < 0 or write_slot >= capacity or write_slot < valid:
            raise ModelingError("attn_res_write_slot_requires_next_slot")
    if mix and valid:
        if any(bank_ids[index] is None for index in range(valid)):
            raise ModelingError("attn_res_mixture_requires_written_valid_slots")


def _vector_work(hidden, valid, *, mixing, fuse_add, output_norm):
    """Semantic arithmetic/cast lanes, not instruction or cycle estimates.

    Reductions count their logical additions. Exp/sqrt/div each count one
    operation; their compiled expansions and lane issue costs are not inferred.
    """
    # Two input casts, add, then BF16 round before scoring or bank storage.
    add = _FUSED_ADD_OPS_PER_ELEMENT * hidden if fuse_add else 0
    if mixing:
        rows = valid + 1
        # Cast, square/reduce/inverse RMS, normalize, dot/reduce; then each
        # BF16 value is cast again for its weighted multiply-add.
        score = rows * (_SCORE_OPS_PER_ELEMENT * hidden + 2)
        weighted = rows * _WEIGHTED_MIX_OPS_PER_ELEMENT * hidden
        # max, subtraction, exp, sum, reciprocal, normalization.
        softmax = ATTNRES_SOFTMAX_OPS_PER_ROW * rows - 1
        mixture = score + weighted + softmax
        # Two weight casts and proj*norm, once per owner.
        initialize = _PROJECTION_SETUP_OPS_PER_ELEMENT * hidden
    else:
        mixture, initialize = hidden, 0  # bypass BF16 -> FP32
    output_round = hidden
    # Rounded mixture cast + RMS + gamma cast + two multiplies + final cast.
    normalization = (_OUTPUT_NORM_OPS_PER_ELEMENT * hidden + _OUTPUT_NORM_SCALAR_OPS
                     if output_norm else 0)
    return {
        "per_token": add + mixture + output_round + normalization,
        "per_owner": initialize,
        "fused_add": add, "mixture": mixture, "output_round": output_round,
        "output_norm": normalization,
    }


def _coop_row_parallel_ub_bytes(rows: int, tile: int, regbase: bool = True) -> int:
    float_block = ATTNRES_COOP_SLOT_BYTES // FP32_BYTES
    meta_align = _align(rows, float_block)
    meta_elems = (
        _align(meta_align, ATTNRES_COOP_HIDDEN_ALIGN)
        if regbase else max(meta_align, ATTNRES_COOP_HIDDEN_ALIGN)
    )
    slots = max(meta_align, rows)
    return (
        tile * (ATTNRES_COOP_WHOLE_BF16_ROWS * _BF16_BYTES + ATTNRES_COOP_FP32_ROWS * FP32_BYTES)
        + (2 * meta_elems + meta_align * float_block + slots * float_block
           + ATTNRES_COOP_SCALAR_SLOTS * float_block) * FP32_BYTES
        + ATTNRES_COOP_SLOT_BYTES + UB_OVERHEAD_BYTES
    )


def select_attnres_dispatch_plan(
    tokens_local: int,
    valid_blocks: int,
    hidden: int,
    ub_bytes: int,
    aiv: int,
    *,
    mix: bool = True,
    mode: str = "token_local",
) -> dict[str, Any]:
    """Resolve physical owner tiling for token-local or upstream cooperative dispatch."""
    if tokens_local <= 0 or aiv <= 0:
        return {
            "tiling_strategy": "empty",
            "launched_owners": 0,
            "active_owners": 0,
            "tokens_per_core": 0,
            "token_group_size": 0,
            "core_ranges": (),
        }
    rows = valid_blocks + 1
    if (mode == "cooperative" and mix and valid_blocks > 0
            and rows <= ATTNRES_COOP_MAX_ROWS and hidden >= ATTNRES_COOP_MIN_HIDDEN
            and tokens_local <= aiv // ATTNRES_COOP_CORE_GAIN):
        whole_tile = _align(hidden, ATTNRES_COOP_HIDDEN_ALIGN)
        if _coop_row_parallel_ub_bytes(rows, whole_tile, regbase=True) <= ub_bytes:
            max_group = min(rows, aiv // tokens_local)
            group_waves = _ceil_div(rows, max_group)
            group = _ceil_div(rows, group_waves)
            row_tasks = tokens_local * rows
            global_cores = min(aiv, row_tasks)
            global_waves = _ceil_div(row_tasks, aiv)
            occupied = tokens_local > aiv // ATTNRES_COOP_LOW_OCCUPANCY_DIV
            if occupied or (group_waves <= global_waves and tokens_local * group < global_cores):
                used = tokens_local * group
                return {
                    "tiling_strategy": "coop_token_group",
                    "launched_owners": used,
                    "active_owners": used,
                    "tokens_per_core": 1,
                    "token_group_size": group,
                    "core_ranges": tuple((idx // group, 1) for idx in range(used)),
                }
            return {
                "tiling_strategy": "coop_global_row",
                "launched_owners": global_cores,
                "active_owners": global_cores,
                "tokens_per_core": 1,
                "token_group_size": 0,
                "core_ranges": tuple((idx % tokens_local, 1) for idx in range(global_cores)),
            }
    launched = min(tokens_local, aiv)
    per_core = _ceil_div(tokens_local, launched)
    ranges = tuple((start, min(per_core, tokens_local - start))
                   for start in range(0, tokens_local, per_core))
    active = len(ranges)
    return {
        "tiling_strategy": "token_local",
        "launched_owners": launched,
        "active_owners": active,
        "tokens_per_core": per_core,
        "token_group_size": 1,
        "core_ranges": ranges,
    }


def _attnres_precision_abi() -> TensorPrecisionABI:
    return TensorPrecisionABI(
        linear_method="none",
        input_dtype="bf16",
        weight_dtype="bf16",
        output_dtype="bf16",
        compute_dtype="fp32",
        accum_dtype="fp32",
        explicit_cast_on_entry="bf16->fp32",
        explicit_cast_on_exit="fp32->bf16",
    )


def _geometry_arch_family(geometry) -> str:
    if geometry is None:
        return ""
    dp_arch = getattr(getattr(geometry, "datapath", None), "compute_arch", None)
    ca = dp_arch or getattr(geometry, "compute_arch", None)
    arch_id = str(getattr(ca, "arch_id", "") or getattr(ca, "preset", "") or "").strip().lower()
    if "3510" in arch_id or arch_id == "a5":
        return "A5"
    if "910b" in arch_id or arch_id == "a2":
        return "A2"
    if "2201" in arch_id or arch_id == "a3":
        return "A3"
    return ""


def _attnres_kernel_binding(arch_family: str = "", *, resident: bool = True) -> KernelBinding:
    arch_norm = str(arch_family or "").strip().upper()
    shipping = (
        arch_norm in {"A2", "A3", "A5"}
        and default_abi_registry().is_shipping_available(NATIVE_KIND, arch_norm)
    )
    return KernelBinding(
        kernel_id=NATIVE_KIND,
        backend="CANN",
        arch_family=arch_norm or "unresolved",
        submodule="attention_residual",
        macro_module="residual",
        readiness="shipping" if shipping else "proposed_pr",
        source_ref=ATTNRES_CALLER_SOURCES[0],
        recipe_id=RECIPE,
        template_id="tk_resident" if resident else "tk_reload",
    )


def _build_owner_schedule(
    view: dict[str, Any],
    dispatch: dict[str, Any],
    cores: Sequence[dict[str, Any]],
    vector: dict[str, int],
    *,
    ub_bytes: int,
    aiv: int,
    total_read: float,
    total_write: float,
) -> OwnerScheduleSpec:
    tokens = int(view["tokens_local"])
    hidden = int(view["hidden"])
    valid = int(view["valid_blocks"])
    mixing = bool(view["mix"] and valid > 0)
    resident = bool(view["resident"])
    q = int(dispatch["tokens_per_core"])
    row_bytes = float(_BF16_BYTES * hidden)
    tree_groups = _ceil_div(hidden, ATTNRES_TREE_GROUP_LANES)
    max_read = max((float(c["read_bytes"]) for c in cores), default=0.0)
    max_write = max((float(c["write_bytes"]) for c in cores), default=0.0)
    phases: list[SerialPhaseSpec] = []
    if tokens > 0:
        if mixing:
            phases.append(SerialPhaseSpec(
                "weight_setup", "mte2", 2.0 * row_bytes, "bytes",
            ))
            phases.append(SerialPhaseSpec(
                "score_rms_reduce", "vec_reduce",
                float(q * (valid + 1) * (hidden - 1)), "op",
                dependent_chain_steps=tree_groups,
                overlap_with=("history_load",) if not resident else (),
            ))
            phases.append(SerialPhaseSpec(
                "softmax_gate", "vec_sfu",
                float(q * (ATTNRES_SOFTMAX_OPS_PER_ROW * (valid + 1) - 1)), "op",
            ))
            phases.append(SerialPhaseSpec(
                "resident_pass_mix" if resident else "reload_pass_mix", "vec_alu",
                float(q * (valid + 1) * _WEIGHTED_MIX_OPS_PER_ELEMENT * hidden), "op",
                overlap_with=("history_reload",) if not resident else (),
            ))
        else:
            phases.append(SerialPhaseSpec(
                "bypass_round", "cast", float(q * 2 * hidden), "op",
            ))
        if view["output_norm"]:
            phases.append(SerialPhaseSpec(
                "output_norm", "vec_reduce", float(q * (hidden - 1)), "op",
                dependent_chain_steps=tree_groups,
            ))
        phases.append(SerialPhaseSpec("output_store", "mte3", max_write, "bytes"))
    residency = OwnerMemoryResidency(
        ub_live_bytes=int(view["ub_bytes_per_owner"]),
        ub_capacity_bytes=int(ub_bytes),
        spill_reload_passes=(1 if resident else 2) if (tokens > 0 and mixing) else 0,
        owner_mte2_bytes=max_read,
        owner_mte3_bytes=max_write,
        shared_hbm_read_bytes=float(total_read),
        shared_hbm_write_bytes=float(total_write),
        l2_admission_mode="reuse",
    )
    return OwnerScheduleSpec(
        owner_kind="aiv",
        tiling_strategy=str(dispatch["tiling_strategy"]),
        parallel_axis="token_rows",
        work_tiles=tokens,
        launched_owners=int(dispatch["launched_owners"]),
        active_owners=int(dispatch["active_owners"]),
        total_owners=int(aiv),
        max_tiles_per_owner=q,
        residency=residency,
        serial_phases=tuple(phases),
        dispatch_envelope_source="P15_dispatch",
    )


def point_metrics(view, ub_bytes, aiv, *, arch_family: str = ""):
    """Independent traffic/vector arithmetic and owner schedule for one native point.

    ``T`` rows use ``launched=min(T,AIV)`` contiguous owners, ``q=ceil(T/launched)``
    rows per owner, and ``active=ceil(T/q)`` nonempty owners; owners past the last
    real row are absent. Resident history is read once (2 bytes per element);
    reload reads it twice (4 bytes).
    """
    tokens, hidden = view["tokens_local"], view["hidden"]
    valid = view["valid_blocks"]
    mixing = view["mix"] and valid > 0
    dispatch_mode = str(view.get("tiling_mode", "token_local") or "token_local")
    dispatch = select_attnres_dispatch_plan(
        tokens, valid, hidden, ub_bytes, aiv, mix=bool(view["mix"]), mode=dispatch_mode,
    )
    ranges = dispatch["core_ranges"]
    per_core = dispatch["tokens_per_core"]
    resident = max_resident_rows(hidden, ub_bytes, True)
    resident = valid + 1 <= resident
    ready = dict(view, resident=resident, used_cores=len(ranges),
                 launched_owners=dispatch["launched_owners"],
                 active_owners=dispatch["active_owners"],
                 tiling_strategy=dispatch["tiling_strategy"],
                 token_group_size=dispatch["token_group_size"],
                 tokens_per_core=per_core, core_ranges=ranges)
    ready["output_norm_cached"] = _cache_output_norm(ready, ub_bytes)
    footprint = estimate_ub_bytes(valid + 1, hidden, resident)
    if ready["output_norm_cached"]:
        footprint += _align(hidden, ELEM_BF16_BLOCK) * 2
    if footprint > ub_bytes:
        raise ModelingError("attn_res_source_ub_footprint_exceeds_capacity")
    ready["ub_bytes_per_owner"] = footprint
    byte2, byte4 = dtype_bytes("bf16"), dtype_bytes("float32")
    bank_read = ((byte2 if resident else byte4) * tokens * valid * hidden
                 if mixing else 0)
    score_weights = (byte2 * hidden + byte2 * hidden) * len(ranges) if mixing else 0
    gamma = (byte2 * hidden * (len(ranges) if ready["output_norm_cached"] else tokens)
             if view["output_norm"] else 0)
    prefix_read = byte2 * tokens * hidden
    addend_read = byte2 * tokens * hidden if view["fuse_add"] else 0
    writes = byte2 * tokens * hidden * (
        1 + int(view["fuse_add"]) + int(view["block_write_idx"] >= 0)
        + int(view["save_materialized"]))
    vector = _vector_work(hidden, valid, mixing=mixing,
                          fuse_add=view["fuse_add"], output_norm=view["output_norm"])
    ready["vector_work_basis"] = vector
    cores = []
    seen_token_starts: set[int] = set()
    for start, count in ranges:
        writes_output = start not in seen_token_starts
        seen_token_starts.add(start)
        read = (_BF16_BYTES * count * hidden
                + (_BF16_BYTES * count * hidden if view["fuse_add"] else 0)
                + ((_BF16_BYTES if resident else 2 * _BF16_BYTES) * count * valid * hidden
                   if mixing else 0)
                + (2 * _BF16_BYTES * hidden if mixing else 0)
                + (_BF16_BYTES * hidden if ready["output_norm_cached"]
                   else (_BF16_BYTES * count * hidden if view["output_norm"] else 0)))
        write = (
            _BF16_BYTES * count * hidden * (
                1 + int(view["fuse_add"]) + int(view["block_write_idx"] >= 0)
                + int(view["save_materialized"]))
            if writes_output else 0
        )
        cores.append({
            "token_start": start, "token_count": count,
            "writes_output": writes_output,
            "read_bytes": read, "write_bytes": write,
            "vector_ops": count * vector["per_token"] + vector["per_owner"],
        })
    total_read = prefix_read + addend_read + bank_read + score_weights + gamma
    resources = {
        "hbm": total_read + writes,
        "mte2": total_read,
        "mte3": writes,
        "vector": tokens * vector["per_token"] + len(ranges) * vector["per_owner"],
    }
    owner_schedule = _build_owner_schedule(
        ready, dispatch, cores, vector,
        ub_bytes=ub_bytes, aiv=aiv, total_read=total_read, total_write=writes,
    )
    return {
        "resources": {key: float(value) for key, value in resources.items() if value},
        "view": ready,
        "cores": tuple(cores),
        "bank_read": bank_read,
        "score_weight_read": score_weights,
        "output_gamma_read": gamma,
        "write_bytes": writes,
        "owner_schedule": owner_schedule,
        "kernel_binding": _attnres_kernel_binding(arch_family=arch_family, resident=resident),
        "precision_abi": _attnres_precision_abi(),
    }


class PointResult(NamedTuple):
    ops: tuple
    hidden_id: str
    raw_prefix_id: str
    materialized_id: str
    bank_ids: tuple


def _point_view(context, *, prefix_id, addend_id, bank_ids, valid_blocks,
                write_slot, output_norm, mix, materialize, tokens_local_value,
                capacity, stride, optimize_prefill):
    model = context["model"]
    if model.activation_dtype != "bf16":
        raise ModelingError("attn_res_source_requires_bf16")
    hidden = model.hidden_size
    if tokens_local_value < 0 or hidden < 1 or capacity < 1:
        raise ModelingError("attn_res_point_geometry_invalid")
    if stride < capacity * hidden:
        raise ModelingError("attn_res_token_stride_smaller_than_capacity_rows")
    if not prefix_id:
        raise ModelingError("attn_res_point_requires_prefix")
    source_prefix_width = context.get("source_prefix_width", hidden)
    validate_source_shapes(
        hidden=hidden, prefix_width=source_prefix_width,
        blocks_width=capacity * hidden, proj_width=hidden, norm_width=hidden,
        addend_width=hidden if addend_id is not None else None,
        output_norm_width=hidden if output_norm else None)
    bank_ids = tuple(bank_ids)
    _validate_bank(bank_ids, capacity, valid_blocks, write_slot, mix)
    view = {
        "tokens": context.get("tokens", tokens_local_value),
        "tokens_local": tokens_local_value,
        "hidden": hidden,
        "valid_blocks": valid_blocks,
        "bank_capacity": capacity,
        "block_token_stride": stride,
        "fuse_add": addend_id is not None,
        "block_write_idx": -1 if write_slot is None else write_slot,
        "mix": bool(mix),
        "output_norm": bool(output_norm),
        "save_materialized": bool(materialize),
        "optimize_prefill": bool(optimize_prefill),
        "phase": context.get("mode", "prefill"),
        "requested_optimize_prefill": bool(optimize_prefill),
        "system_workspace_bytes": SYSTEM_WORKSPACE_BYTES,
        "user_fp32_workspace_bytes": 0,
        "stored_dtype": model.activation_dtype,
        "working_dtype": "fp32",
        "rounding": "bf16_after_add_and_mix",
        "dtype": model.activation_dtype,
        "attn_res_source_abi": {
            "prefix_sum": (tokens_local_value, hidden),
            "block_residual": (tokens_local_value, capacity, hidden),
            "proj_weight": (1, hidden),
            "norm_weight": (hidden,),
            "addend": (tokens_local_value, hidden) if addend_id is not None else None,
            "output_norm": (hidden,) if output_norm else None,
            "returns": ("hidden_states", "prefix_out", "materialized"),
            "prefix_width": hidden,
            "dynamic_prefix_width_supported": False,
            "dtype": "bf16",
        },
    }
    return view, bank_ids


def attn_res_point(context, *, prefix_id, addend_id, bank_ids, valid_blocks,
                   write_slot, output_norm, mix, materialize, tokens_local,
                   bank_capacity, block_token_stride, optimize_prefill=False):
    """Build one native AttnResFwd call and its explicit bank transition.

    ``bank_ids`` is one entry per physical capacity slot. Only ``valid_blocks``
    older rows feed the mixture; the current prefix is the extra row. The
    returned raw-prefix/materialized ids alias input/output exactly when the
    source does (no addend, or no save_materialized).
    """
    view, banks = _point_view(
        context, prefix_id=prefix_id, addend_id=addend_id, bank_ids=bank_ids,
        valid_blocks=valid_blocks, write_slot=write_slot,
        output_norm=output_norm, mix=mix, materialize=materialize,
        tokens_local_value=tokens_local, capacity=bank_capacity,
        stride=block_token_stride, optimize_prefill=optimize_prefill)
    bank_rows = tuple(banks[index] for index in range(valid_blocks)) if mix else ()
    deps = tuple(dict.fromkeys(
        (prefix_id, *((addend_id,) if addend_id is not None else ()), *bank_rows)))
    op_id = context["prefix"] + ".attn_res_fwd"
    raw_id = op_id if addend_id is not None else prefix_id
    materialized_id = op_id
    op = OpSpec(
        op_id=op_id, family="attn_res_prefix",
        phase=context.get("mode", "prefill"), stage=context["stage"],
        layer=context["layer"], requests=tuple(context.get("requests", ())),
        semantic_shape=(tokens_local, context["model"].hidden_size),
        local_shape=(tokens_local, context["model"].hidden_size),
        padded_shape=(tokens_local, context["model"].hidden_size),
        launch_shape=(tokens_local, context["model"].hidden_size),
        deps=deps, recipe=RECIPE, source_boundary="kernel",
        dtype=context["model"].activation_dtype,
        layout="contiguous",
        coverage="complete_source_call",
        returns=("hidden_states", "prefix_out", "materialized"),
        metadata=tuple(sorted({
            "native_kind": NATIVE_KIND,
            "attn_res_view": view,
            "attn_res_bank_ids": banks,
            "attn_res_prefix_id": prefix_id,
            "attn_res_addend_id": addend_id,
            "attn_res_bank_rows": bank_rows,
            "attn_res_raw_prefix_id": raw_id,
            "attn_res_materialized_id": materialized_id,
            "attn_res_prefix_out_aliases_prefix": addend_id is None,
            "attn_res_materialized_aliases_hidden": not materialize,
            "attn_res_source": SOURCE,
            "attn_res_limitations": LIMITATIONS,
            "source_boundary_kind": "kernel",
            "submitted_kernels": 1,
            "implementation": {"native_kind": NATIVE_KIND,
                               "scope": "full_source_call"},
        }.items())),
        notes=(
            "source: retained AttnResFwd caller and AIV-only arch35 kernel",
            "one native point launch; no cube GEMM and no FP32-v user workspace",
            "serial per-owner load/compute/store phases, shared-rate contention",
        ),
    )
    op = replace(op, **attnres_spec_terms(op))
    ops = [op]
    declare_group(
        ops, key=op_id + ".group", role=GROUP_ROLE, family="attn_res_prefix",
        members=(op_id,), entries=(op_id,), anchor=op_id, view=view,
        implementation={"native_kind": NATIVE_KIND,
                        "conditions": {"scope": "full_source_call"}},
        source=SOURCE, block=context.get("structure_block"),
        scope_role="residual",
        coverage={"kind": "complete_subgraph", "limitations": list(LIMITATIONS)},
    )
    bank_outputs = list(banks)
    if write_slot is not None:
        bank_outputs[write_slot] = op_id
    return PointResult(tuple(ops), op_id, raw_id, materialized_id,
                       tuple(bank_outputs))


@dataclass
class AttnResPrefixState:
    """The caller's cross-layer prefix/pending-MLP state machine."""

    model: object
    runtime: object
    tokens: int
    tokens_local: int
    mode: str
    requests: tuple
    optimize_prefill: bool
    capacity: int
    stride: int
    block_size: int
    prefix_id: str
    bank_ids: tuple
    pending_mlp_id: str | None = None
    clear_before_attention: bool = False

    @classmethod
    def start(cls, *, model, runtime, ready, tokens,
              mode, requests, optimize_prefill=False):
        if runtime.parallel.pp < 1:
            raise ModelingError("attn_res_source_caller_requires_pp1")
        block = next((plan.residual_block_size for plan in model.layer_plans
                      if plan.residual_block_size), 0)
        if block < 1:
            raise ModelingError("attn_res_requires_residual_block_size")
        if any((plan.residual_block_size != block or plan.engram
                or plan.hyper_connections) for plan in model.layer_plans):
            raise ModelingError("attn_res_selected_semantics_unsupported_for_layer")
        capacity = bank_capacity(model)
        return cls(model, runtime, tokens, tokens, mode,
                   tuple(requests), bool(optimize_prefill and mode == "prefill"), capacity,
                   block_token_stride(model, capacity), block, ready,
                   (None,) * capacity)

    def _context(self, prefix, layer, stage, structure_block):
        return {
            "prefix": prefix, "stage": stage, "layer": layer.index,
            "requests": self.requests, "model": self.model,
            "mode": self.mode,
            "tokens": self.tokens, "structure_block": structure_block,
        }

    def prepare(self, *, prefix, layer, stage, materialize=False):
        block = self.block_size
        boundary = layer.index == 0 or layer.index % self.block_size == 0
        valid = _ceil_div(layer.index, block)
        slot = layer.index // block if boundary else None
        result = attn_res_point(
            self._context(prefix, layer, stage, "attention"),
            prefix_id=self.prefix_id, addend_id=self.pending_mlp_id,
            bank_ids=self.bank_ids, valid_blocks=valid, write_slot=slot,
            output_norm=True, mix=True, materialize=materialize,
            tokens_local=self.tokens_local, bank_capacity=self.capacity,
            block_token_stride=self.stride,
            optimize_prefill=self.optimize_prefill)
        self.prefix_id = result.raw_prefix_id
        self.bank_ids = result.bank_ids
        self.pending_mlp_id = None
        self.clear_before_attention = boundary
        return result

    def attention_added(self, *, prefix, layer, stage, attention_id, materialize=False):
        block = self.block_size
        boundary = self.clear_before_attention
        valid = _ceil_div(layer.index, block) + int(boundary)
        result = attn_res_point(
            self._context(prefix, layer, stage, "ffn"),
            prefix_id=attention_id if boundary else self.prefix_id,
            addend_id=None if boundary else attention_id,
            bank_ids=self.bank_ids, valid_blocks=valid, write_slot=None,
            output_norm=True, mix=True, materialize=materialize,
            tokens_local=self.tokens_local, bank_capacity=self.capacity,
            block_token_stride=self.stride,
            optimize_prefill=self.optimize_prefill)
        self.prefix_id = result.raw_prefix_id
        self.bank_ids = result.bank_ids
        self.clear_before_attention = False
        return result

    def defer_mlp(self, mlp_id):
        if self.pending_mlp_id is not None:
            raise ModelingError("attn_res_pending_mlp_must_be_absorbed_once")
        self.pending_mlp_id = mlp_id

    def final(self, *, prefix, last_layer, stage):
        return attn_res_point(
            self._context(prefix, last_layer, stage, "ffn"),
            prefix_id=self.prefix_id, addend_id=self.pending_mlp_id,
            bank_ids=self.bank_ids,
            valid_blocks=_ceil_div(last_layer.index + 1, self.block_size),
            write_slot=None, output_norm=False, mix=True, materialize=False,
            tokens_local=self.tokens_local, bank_capacity=self.capacity,
            block_token_stride=self.stride,
            optimize_prefill=self.optimize_prefill)

def _physical_metrics(op, geometry):
    if op.recipe != RECIPE or op.source_boundary != "kernel":
        raise ModelingError("attn_res_plan_requires_selected_kernel_source")
    if op.dtype != "bf16":
        raise ModelingError("attn_res_source_requires_bf16")
    metadata = dict(op.metadata)
    view = dict(metadata["attn_res_view"])
    ub = dict(geometry.buffers).get("ub")
    if not ub:
        raise ModelingError("attn_res_requires_declared_ub_buffer")
    aiv = geometry.resource_units.get("vector", geometry.compute_units)
    role = read_role(geometry.resource_units, 0)
    arch_family = _geometry_arch_family(geometry)
    metrics = point_metrics(view, ub, aiv, arch_family=arch_family)
    return metrics, role, aiv


def _primitive_p50_ns(table: Any, pid: str, metric: str) -> float | None:
    cell_fn = getattr(table, "cell", None)
    if not callable(cell_fn):
        return None
    band = cell_fn(pid).band(metric)
    val = getattr(band, "p50", None)
    if val is None or float(val) <= 0.0:
        return None
    return float(val)


def _primitive_p50_per_ms(table: Any, pid: str, metric: str) -> float | None:
    val = _primitive_p50_ns(table, pid, metric)
    return (val / MS_PER_SECOND) if val is not None else None


def _attnres_phase_timings(context: Any, view: dict[str, Any], role: str, aiv: int) -> dict[str, Any]:
    """Derive per-owner causal phase durations (ms) and DMA caps from bound hardware rates."""
    hidden = int(view["hidden"])
    valid = int(view["valid_blocks"])
    resident = bool(view["resident"])
    row_bytes = float(_BF16_BYTES * hidden)
    vec_row = context.resource_names.get("vector", "vector")
    vec_units = max(1, int(context.resource_units.get("vector", aiv)))
    vec_owner_ms = float(context.rates[vec_row]) / float(vec_units)
    tbl = getattr(context, "primitive_table", None)
    p07_raw = _primitive_p50_per_ms(tbl, "P07_vec_alu", "vec_alu_ops_per_s_per_aiv")
    p07_ms = min(p07_raw, vec_owner_ms) if p07_raw is not None else vec_owner_ms
    scale = (p07_raw / p07_ms) if (p07_raw is not None and p07_ms > 0.0) else 1.0
    p08_raw = _primitive_p50_per_ms(tbl, "P08_vec_sfu", "vec_sfu_ops_per_s_per_aiv")
    p08_ms = (p08_raw / scale) if p08_raw is not None else p07_ms
    p09_red_raw = _primitive_p50_per_ms(tbl, "P09_vec_reduce_cast", "vec_reduce_ops_per_s_per_aiv")
    p09_red_ms = (p09_red_raw / scale) if p09_red_raw is not None else p07_ms
    p09_cast_raw = _primitive_p50_per_ms(tbl, "P09_vec_reduce_cast", "cast_bytes_per_s_per_aiv")
    p09_cast_ms = (
        (p09_cast_raw / scale)
        if p09_cast_raw is not None
        else (p07_ms * float(ATTNRES_CAST_BYTES_PER_ELEMENT))
    )
    ub_read_row = context.resource_names.get("ub_read_bytes")
    ub_read_rate = context.rates.get(ub_read_row) if ub_read_row else None
    if ub_read_rate is not None and float(ub_read_rate) > 0.0:
        ub_copy_ms = float(ub_read_rate) / float(max(1, aiv))
    else:
        ub_copy_bps = context.service_rate("ub_copy")
        if ub_copy_bps is not None and float(ub_copy_bps) > 0.0:
            ub_copy_ms = float(ub_copy_bps) / (MS_PER_SECOND * float(max(1, aiv)))
        elif p09_cast_ms > 0.0:
            ub_copy_ms = p09_cast_ms
        else:
            raise ModelingError("attn_res_ub_copy_rate_missing")
    p12_flag_ns = _primitive_p50_ns(tbl, "P12_intracore_flag", "flag_ns")
    p12_mte2_v_ns = _primitive_p50_ns(tbl, "P12_intracore_flag", "flag_mte2_v_ns")
    clock_hz = getattr(getattr(context, "compute_arch", None), "clock_hz", None)
    if p12_flag_ns is not None:
        cycle_ns = p12_flag_ns / P12_FLAG_CYCLES
        flag_ns = p12_flag_ns
    elif clock_hz is not None and float(clock_hz) > 0.0:
        cycle_ns = NS_PER_SECOND / float(clock_hz)
        flag_ns = P12_FLAG_CYCLES * cycle_ns
    else:
        cycle_ns = 0.0
        flag_ns = 0.0
    mte2_v_ns = p12_mte2_v_ns if p12_mte2_v_ns is not None else flag_ns
    tree_groups = _ceil_div(hidden, ATTNRES_TREE_GROUP_LANES)
    regbase_groups = _ceil_div(hidden, ATTNRES_REGBASE_GROUP_ELEMS)
    stall_cycles = (
        ATTNRES_RESIDENT_STRIP_STALL_CYCLES if resident
        else ATTNRES_RELOAD_STRIP_STALL_CYCLES
    )
    has_calibrated_vec = p09_red_raw is not None or p07_raw is not None
    per_row_stall_ns = (
        float(tree_groups) * (float(stall_cycles) * cycle_ns + flag_ns)
        + (0.0 if resident else float(regbase_groups) * mte2_v_ns)
    )
    tree_delta_ms = ((per_row_stall_ns / NS_PER_MS) * scale) if has_calibrated_vec else 0.0

    alu_row = float(hidden) / p07_ms
    cast_row = float(ATTNRES_CAST_BYTES_PER_ELEMENT * hidden) / p09_cast_ms
    reduce_row = float(max(0, hidden - 1)) / p09_red_ms + tree_delta_ms
    sfu_row = float(ATTNRES_SFU_STRIP_OPS * ATTNRES_TREE_GROUP_LANES) / p08_ms
    score_row = max(alu_row + reduce_row, cast_row) + 2.0 * alu_row + reduce_row + sfu_row
    mix_row = max(2.0 * alu_row, cast_row)
    norm_row = max(alu_row + reduce_row, cast_row) + max(2.0 * alu_row, 2.0 * cast_row) + sfu_row
    softmax_ms = float(ATTNRES_SOFTMAX_OPS_PER_ROW * (valid + 1) - 1) / p08_ms if valid > 0 else 0.0

    read_caps: dict[str, float] = {}
    write_caps: dict[str, float] = {}
    read_row = context.resource_names.get(context.physical_role(role))
    cold_mte2_bps = context.service_rate(f"{role}_dram") or context.service_rate("mte2_aiv_dram")
    if read_row and cold_mte2_bps is not None and float(cold_mte2_bps) > 0.0:
        read_units = max(1, int(context.resource_units.get(context.physical_role(role), aiv)))
        read_caps[read_row] = float(cold_mte2_bps) / (MS_PER_SECOND * float(read_units))
    hbm_row = context.resource_names.get("hbm")
    hbm_r_ms = _primitive_p50_per_ms(tbl, "P01_datacopy", "hbm_read_bytes_per_s")
    hbm_w_ms = _primitive_p50_per_ms(tbl, "P01_datacopy", "hbm_write_bytes_per_s")
    mte3_row = context.resource_names.get("mte3")
    mte3_w_ms = _primitive_p50_per_ms(tbl, "P04_mte3_store", "mte3_store_bytes_per_s_per_aiv")
    if hbm_row and hbm_r_ms is not None:
        read_caps[hbm_row] = hbm_r_ms
    if hbm_row and hbm_w_ms is not None:
        write_caps[hbm_row] = hbm_w_ms
    if mte3_row and mte3_w_ms is not None:
        write_caps[mte3_row] = mte3_w_ms
    return {
        "setup_ms": 2.0 * cast_row + alu_row,
        "fuse_add_ms": 2.0 * cast_row + alu_row,
        "cast_row_ms": cast_row,
        "score_row_ms": score_row,
        "mix_row_ms": mix_row,
        "norm_row_ms": norm_row,
        "softmax_ms": softmax_ms,
        "ub_copy_row_ms": row_bytes / ub_copy_ms if ub_copy_ms > 0.0 else 0.0,
        "read_caps": read_caps,
        "write_caps": write_caps,
    }


@dataclass(frozen=True, slots=True)
class _AttnResPrefixProgram:
    """Picklable causal phase-program builder for native AttnResFwd."""

    view: dict[str, Any]
    cores: tuple[dict[str, Any], ...]
    role: str
    aiv: int

    def _build_reload_stages(
        self,
        context: Any,
        placement: dict[str, tuple[int, ...]],
        timings: dict[str, Any],
        *,
        prev: str,
        valid: int,
        row_bytes: float,
        score_ops: float,
        mix_ops: float,
        softmax_ops: float,
    ) -> tuple[list[Stage], str]:
        read_caps = timings["read_caps"]
        load_one = context.bind(Work({}, {"hbm": row_bytes, self.role: row_bytes}))
        score_one = context.bind(Work({}, {"vector": score_ops}))
        mix_one = context.bind(Work({}, {"vector": mix_ops}))
        s_lat = timings["score_row_ms"]
        m_lat = timings["mix_row_ms"]

        def _score_step(_i: int):
            return (
                Stage("load_row", load_one, previous=("load_row",),
                      placement=placement, caps=read_caps, coupled=True),
                Stage("score_row", score_one, latency=s_lat,
                      after=("load_row",), previous=("score_row",), placement=placement),
            )

        def _mix_step(_i: int):
            return (
                Stage("reload_row", load_one, previous=("reload_row",),
                      placement=placement, caps=read_caps, coupled=True),
                Stage("mix_row", mix_one, latency=m_lat,
                      after=("reload_row",), previous=("mix_row",), placement=placement),
            )

        stages = [
            Stage(
                "pass1_history", after=(prev,),
                program=PipelineProgram(valid, ATTNRES_RELOAD_PIPELINE_SLOTS, _score_step),
            ),
            Stage(
                "pass1_prefix_score", score_one, latency=s_lat,
                after=("pass1_history",), placement=placement,
            ),
            Stage(
                "softmax",
                context.bind(Work({}, {"vector": softmax_ops})),
                latency=timings["softmax_ms"], after=("pass1_prefix_score",), placement=placement,
            ),
            Stage(
                "pass2_history", after=("softmax",),
                program=PipelineProgram(valid, ATTNRES_RELOAD_PIPELINE_SLOTS, _mix_step),
            ),
            Stage(
                "pass2_prefix_mix", mix_one, latency=m_lat,
                after=("pass2_history",), placement=placement,
            ),
        ]
        return stages, "pass2_prefix_mix"

    def _build_token_stages(
        self,
        context: Any,
        placement: dict[str, tuple[int, ...]],
        timings: dict[str, Any],
        *,
        writes_output: bool = True,
    ) -> tuple[Stage, ...]:
        view = self.view
        hidden = int(view["hidden"])
        valid = int(view["valid_blocks"])
        mixing = bool(view["mix"] and valid > 0)
        resident = bool(view["resident"])
        fuse_add = bool(view["fuse_add"])
        output_norm = bool(view["output_norm"])
        gamma_cached = bool(view["output_norm_cached"])
        row_bytes = float(_BF16_BYTES * hidden)
        write_mult = (
            1 + int(fuse_add) + int(view["block_write_idx"] >= 0)
            + int(view["save_materialized"])
        )
        read_caps = timings["read_caps"]
        write_caps = timings["write_caps"]
        # In each mixing pass, the per-row value BF16->FP32 cast runs concurrently
        # with the 2H weighted-accumulate ALU ops (max(2A, C)); place the overlapped
        # cast credit on the score pass so mix_row is not double-serialized on vector
        # while total declared vector work stays strictly conserved.
        score_ops = float((_SCORE_OPS_PER_ELEMENT + 1) * hidden + 2)
        mix_ops = float((_WEIGHTED_MIX_OPS_PER_ELEMENT - 1) * hidden)
        softmax_ops = float(ATTNRES_SOFTMAX_OPS_PER_ROW * (valid + 1) - 1)
        norm_ops = float(_OUTPUT_NORM_OPS_PER_ELEMENT * hidden + _OUTPUT_NORM_SCALAR_OPS)
        add_ops = float(_FUSED_ADD_OPS_PER_ELEMENT * hidden)
        stages: list[Stage] = []
        prefix_bytes = row_bytes * float(1 + int(fuse_add))
        stages.append(Stage(
            "prefix_load",
            context.bind(Work({}, {"hbm": prefix_bytes, self.role: prefix_bytes})),
            placement=placement, caps=read_caps, coupled=True,
        ))
        prev = "prefix_load"
        if fuse_add:
            stages.append(Stage(
                "fused_add",
                context.bind(Work({}, {"vector": add_ops})),
                latency=timings["fuse_add_ms"], after=(prev,), placement=placement,
            ))
            prev = "fused_add"
        if mixing and resident:
            hist_bytes = float(valid) * row_bytes
            stages.append(Stage(
                "history_load",
                context.bind(Work({}, {"hbm": hist_bytes, self.role: hist_bytes})),
                after=(prev,), placement=placement, caps=read_caps, coupled=True,
            ))
            stages.append(Stage(
                "history_ub_copy",
                latency=float(valid) * timings["ub_copy_row_ms"],
                after=("history_load",), placement=placement,
            ))
            stages.append(Stage(
                "score_rows",
                context.bind(Work({}, {"vector": float(valid + 1) * score_ops})),
                latency=float(valid + 1) * timings["score_row_ms"],
                after=("history_ub_copy",), placement=placement,
            ))
            stages.append(Stage(
                "softmax",
                context.bind(Work({}, {"vector": softmax_ops})),
                latency=timings["softmax_ms"], after=("score_rows",), placement=placement,
            ))
            stages.append(Stage(
                "mix_rows",
                context.bind(Work({}, {"vector": float(valid + 1) * mix_ops})),
                latency=float(valid + 1) * timings["mix_row_ms"],
                after=("softmax",), placement=placement,
            ))
            prev = "mix_rows"
        elif mixing:
            reload_stages, prev = self._build_reload_stages(
                context, placement, timings,
                prev=prev, valid=valid, row_bytes=row_bytes,
                score_ops=score_ops, mix_ops=mix_ops, softmax_ops=softmax_ops,
            )
            stages.extend(reload_stages)
        else:
            stages.append(Stage(
                "bypass_cast",
                context.bind(Work({}, {"vector": float(hidden)})),
                latency=timings["cast_row_ms"], after=(prev,), placement=placement,
            ))
            prev = "bypass_cast"
        stages.append(Stage(
            "output_round",
            context.bind(Work({}, {"vector": float(hidden)})),
            latency=timings["cast_row_ms"], after=(prev,), placement=placement,
        ))
        prev = "output_round"
        if output_norm:
            if not gamma_cached:
                stages.append(Stage(
                    "output_gamma_load",
                    context.bind(Work({}, {"hbm": row_bytes, self.role: row_bytes})),
                    after=(prev,), placement=placement, caps=read_caps, coupled=True,
                ))
                prev = "output_gamma_load"
            stages.append(Stage(
                "output_norm",
                context.bind(Work({}, {"vector": norm_ops})),
                latency=timings["norm_row_ms"], after=(prev,), placement=placement,
            ))
            prev = "output_norm"
        if writes_output:
            write_bytes = float(write_mult) * row_bytes
            stages.append(Stage(
                "output_store",
                context.bind(Work({}, {"hbm": write_bytes, "mte3": write_bytes})),
                after=(prev,), placement=placement, caps=write_caps, coupled=True,
            ))
        return tuple(stages)

    def __call__(self, context: Any) -> ParallelProgram | KernelProgram:
        if not self.cores:
            return KernelProgram(())
        view = self.view
        hidden = int(view["hidden"])
        mixing = bool(view["mix"] and int(view["valid_blocks"]) > 0)
        gamma_cached = bool(view["output_norm"] and view["output_norm_cached"])
        row_bytes = float(_BF16_BYTES * hidden)
        setup_vec = float(_PROJECTION_SETUP_OPS_PER_ELEMENT * hidden)
        timings = _attnres_phase_timings(context, view, self.role, self.aiv)
        ub_pool_units = context.pool_units.get("ub", self.aiv)
        programs: list[KernelProgram] = []
        for owner, core in enumerate(self.cores):
            placement = context.owned(owner, self.aiv, (self.role, "vector", "mte3"))
            allocation = (context.allocate(
                "attn_res_ub", "ub", view["ub_bytes_per_owner"],
                unit_group(owner, self.aiv, ub_pool_units)),)
            stages: list[Stage] = [Stage("ub_acquire", acquire=allocation)]
            prev = "ub_acquire"
            setup_read = (2.0 * row_bytes if mixing else 0.0) + (row_bytes if gamma_cached else 0.0)
            if setup_read > 0.0:
                stages.append(Stage(
                    "weight_load",
                    context.bind(Work({}, {"hbm": setup_read, self.role: setup_read})),
                    after=(prev,), placement=placement, caps=timings["read_caps"], coupled=True,
                ))
                prev = "weight_load"
            if mixing:
                stages.append(Stage(
                    "weight_setup",
                    context.bind(Work({}, {"vector": setup_vec})),
                    latency=timings["setup_ms"], after=(prev,), placement=placement,
                ))
                prev = "weight_setup"
            token_stages = self._build_token_stages(
                context, placement, timings,
                writes_output=bool(core.get("writes_output", True)),
            )
            count = int(core["token_count"])
            stages.append(Stage(
                "tokens", after=(prev,),
                program=PipelineProgram(count, 1, lambda _idx, ts=token_stages: ts),
            ))
            stages.append(Stage("ub_release", after=("tokens",), free=allocation))
            programs.append(KernelProgram(tuple(stages)))
        return ParallelProgram(tuple(programs))


def _plan(metrics, role, aiv):
    resources = dict(metrics["resources"])
    resources[role] = resources.pop("mte2", 0)
    total = Work({}, resources)
    program = _AttnResPrefixProgram(
        view=dict(metrics["view"]),
        cores=tuple(metrics["cores"]),
        role=role,
        aiv=int(aiv),
    )
    return OperatorPlan(
        total,
        program,
        metadata={
            "native_kind": NATIVE_KIND,
            "tiling_strategy": metrics["view"].get("tiling_strategy", "token_local"),
            "launched_owners": metrics["view"].get("launched_owners", 0),
            "active_owners": metrics["view"].get("active_owners", 0),
            "resident": bool(metrics["view"].get("resident", True)),
            "owner_schedule": metrics["owner_schedule"].to_dict(),
            "kernel_binding": metrics["kernel_binding"].to_dict(),
            "precision_abi": metrics["precision_abi"].to_dict(),
        },
    )


def build(op, geometry):
    """Bind physical ownership at pricing, independently of semantic lowering."""
    return _plan(*_physical_metrics(op, geometry))


class AttnResRecipe(PlannedRecipe):
    def prepare(self, op, geometry):
        metrics, role, aiv = _physical_metrics(op, geometry)
        plan = _plan(metrics, role, aiv)
        metadata = dict(op.metadata)
        metadata.update(attn_res_view=metrics["view"],
                        attn_res_core_work=metrics["cores"],
                        attn_res_read_role=role)
        metadata["structure_subgraphs"] = tuple(
            {**group, "tensor_view": metrics["view"]}
            for group in metadata.get("structure_subgraphs", ()))
        bound = replace(
            op,
            metadata=tuple(metadata.items()),
            precision_abi=metrics["precision_abi"],
            kernel_binding=metrics["kernel_binding"],
            owner_schedule=metrics["owner_schedule"],
        )
        return plan.work.apply(bound), plan


def recipes():
    return {RECIPE: AttnResRecipe(build)}


__all__ = [
    "AttnResPrefixState", "CAPABILITIES", "IMPLEMENTATION", "LIMITATIONS", "NATIVE_KIND",
    "PointResult", "RECIPE", "SOURCE", "attn_res_point", "bank_capacity",
    "block_token_stride", "build", "capabilities", "estimate_ub_bytes",
    "max_resident_rows", "point_metrics", "recipes", "select_attnres_dispatch_plan",
    "selection",
]
