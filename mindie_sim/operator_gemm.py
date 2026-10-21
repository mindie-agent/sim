"""Explicit, uncalibrated BF16 GEMM execution plans.

This provider describes one declared tile program, not an optimal tiling or a
measured kernel.  All on-chip capacities and rates come from ``GemmProfile``;
machine compute, HBM and aggregate copy rates remain engine resources.  Missing
profiles and unsupported layouts/quantization fall back to the caller's normal
cost path.  There are no SKU-selected constants or implicit cache-hit credits.

Each output tile has cyclic core ownership.  Per-core command ordering and
finite L1, L0A/B and accumulator slots are DAG edges, so prologue, finite K-loop,
epilogue and tail-core imbalance emerge in the existing resource engine.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

from .a3_ops import OpSpec
from .a3_types import MachineSpec
from .operator_plan import (
    EstimateContext, ExecutionStage, KernelEstimate, KernelPlan,
)


_DENSE_FAMILIES = frozenset({
    "gemm", "qkv_projection", "o_projection", "gate_up_projection",
    "down_projection", "mla_q_a", "mla_q_b", "mla_kv_a", "mla_kv_b",
    "shared_expert_gate_up", "shared_expert_down", "logits",
    "vision_patch_embed", "vision_projector",
})
_BF16 = frozenset({"bf16", "bfloat16"})
_ASCEND_FAMILIES = frozenset({"A2", "A3", "A5"})


@dataclass(frozen=True)
class GemmProfile:
    """Declared per-core hierarchy and one tile program, in bytes and bytes/s.

    ``source`` must identify the declaration/measurement supporting the
    parameters.  In examples it should explicitly say ``synthetic_fixture`` or
    ``uncalibrated_assumption``.  This class never infers per-core copy rates
    from a package bandwidth.  MTE1 A and B commands share a serial stream in
    this plan; their per-command service times therefore add.
    """

    core_count: int
    tile_m: int
    tile_n: int
    tile_k: int
    l1_k: int
    l1_bytes: int
    l0a_bytes: int
    l0b_bytes: int
    l0c_bytes: int
    l2_bytes_per_s_per_core: float
    mte2_bytes_per_s_per_core: float
    mte1_a_bytes_per_s_per_core: float
    mte1_b_bytes_per_s_per_core: float
    mte3_bytes_per_s_per_core: float
    fix_bytes_per_s_per_core: float
    source: str
    l1_buffers: int = 2
    l0_buffers: int = 2
    c_buffers: int = 2
    max_stages: int = 100_000

    def __post_init__(self) -> None:
        integral = {
            "core_count", "tile_m", "tile_n", "tile_k", "l1_k",
            "l1_bytes", "l0a_bytes", "l0b_bytes", "l0c_bytes",
            "l1_buffers", "l0_buffers", "c_buffers", "max_stages",
        }
        for item in fields(self):
            if item.name == "source":
                if not isinstance(self.source, str) or not self.source.strip():
                    raise ValueError("gemm_profile_source_missing")
                continue
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"gemm_profile_invalid:{item.name}")
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"gemm_profile_invalid:{item.name}")
            if item.name in integral and not isinstance(value, int):
                raise ValueError(f"gemm_profile_integer_required:{item.name}")
        if self.l1_k % self.tile_k:
            raise ValueError("gemm_profile_l1_k_not_tile_k_multiple")
        if any(getattr(self, key) not in (1, 2) for key in (
            "l1_buffers", "l0_buffers", "c_buffers",
        )):
            raise ValueError("gemm_profile_buffers_must_be_one_or_two")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GemmProfile":
        if not isinstance(raw, Mapping):
            raise ValueError("gemm_profile_mapping_required")
        allowed = {item.name for item in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"gemm_profile_unknown_field:{sorted(unknown)[0]}")
        try:
            return cls(**dict(raw))
        except TypeError as exc:
            raise ValueError("gemm_profile_required_fields_missing") from exc


def _dependencies(*names: str | None) -> tuple[str, ...]:
    return tuple(dict.fromkeys(name for name in names if name))


def _reject(reason: str) -> KernelEstimate:
    return KernelEstimate(unsupported_reason=reason)


class GemmProvider:
    """Opt-in dense BF16 planner implementing the shared estimate protocol."""

    def __init__(self, profile: GemmProfile | None = None) -> None:
        self.profile = profile

    def estimate(
        self, op: OpSpec, machine: MachineSpec, context: EstimateContext,
    ) -> KernelEstimate:
        p = self.profile
        if p is None:
            return _reject("gemm_profile_missing")
        if op.family not in _DENSE_FAMILIES:
            return _reject("gemm_family_unsupported")
        # Recognized Ascend BF16 programs issue whole 16-lane matrix
        # fragments.  Unrecognized arithmetic fixtures can use normalized
        # blocks, but are never evidence that such blocks are legal Ascend
        # instructions.  This shape guard is not kernel certification.
        ascend_fragments = machine.arch_family.upper() in _ASCEND_FAMILIES
        if ascend_fragments and any(v % 16 for v in (p.tile_m, p.tile_n, p.tile_k)):
            return _reject("gemm_bf16_fragment_alignment_required")
        if str(context.quantization or "").strip().lower() not in {
            "", "none", "unquantized",
        }:
            return _reject("gemm_quantization_unsupported")
        dtypes = (
            op.dtype or context.compute_dtype, context.compute_dtype,
            context.weight_dtype or op.dtype or context.compute_dtype,
            context.activation_dtype or op.dtype or context.compute_dtype,
        )
        if any(str(value).lower() not in _BF16 for value in dtypes):
            return _reject("gemm_requires_bf16_dense")
        if op.layout.lower() not in {"contiguous", "row_major", "nd"}:
            return _reject("gemm_layout_unsupported")
        meta = dict(op.metadata)
        for key in ("groups", "group_count", "batch", "batch_count"):
            if key in meta and meta[key] != 1:
                return _reject("gemm_grouped_or_batched_unsupported")
        if any(meta.get(key) for key in (
            "ragged", "grouped", "trans_a", "trans_b", "transpose_a",
            "transpose_b", "quantized", "quantization", "quant_method", "quant",
            "quant_type", "weight_quant", "activation_quant", "packing",
            "weight_scale", "activation_scale", "bias", "fused_activation",
        )):
            return _reject("gemm_features_unsupported")
        for key in ("weight_dtype", "activation_dtype", "output_dtype"):
            if key in meta and str(meta[key]).lower() not in _BF16:
                return _reject("gemm_requires_bf16_dense")
        shape = op.launch_shape
        if (len(shape) != 3 or any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0
            for v in shape
        )):
            return _reject("gemm_shape_requires_positive_m_n_k")
        if not machine.dies or op.stage < 0:
            return _reject("gemm_machine_missing")
        die = machine.dies[op.stage % len(machine.dies)]
        if die.aic_count and die.aic_count != p.core_count:
            return _reject("gemm_profile_core_count_mismatch")
        for key in ("l1_bytes", "l0a_bytes", "l0b_bytes", "l0c_bytes"):
            declared = getattr(die, key)
            if declared and getattr(p, key) > declared:
                return _reject(f"gemm_profile_exceeds_machine:{key}")
        peak = machine.peak_flops_for(context.compute_dtype)
        if not all(math.isfinite(v) and v > 0 for v in (
            peak, die.hbm_bw_bytes_per_s, die.mte_bw_bytes_per_s,
        )):
            return _reject("gemm_machine_rates_missing")
        required = {
            "l1_bytes": p.l1_buffers * (p.tile_m + p.tile_n) * p.l1_k * 2,
            "l0a_bytes": p.l0_buffers * p.tile_m * p.tile_k * 2,
            "l0b_bytes": p.l0_buffers * p.tile_n * p.tile_k * 2,
            "l0c_bytes": p.c_buffers * p.tile_m * p.tile_n * 4,
        }
        for key, used in required.items():
            if used > getattr(p, key):
                return _reject(f"gemm_buffer_capacity_exceeded:{key}")
        m, n, k = shape
        tiles_m = (m + p.tile_m - 1) // p.tile_m
        tiles_n = (n + p.tile_n - 1) // p.tile_n
        tile_count = tiles_m * tiles_n
        macros = (k + p.l1_k - 1) // p.l1_k
        micros = (k + p.tile_k - 1) // p.tile_k
        stage_count = tile_count * (macros + 2 * micros + 1)
        if stage_count > p.max_stages:
            return _reject("gemm_plan_stage_limit_exceeded")
        stages: list[ExecutionStage] = []
        states = [{
            "mte2": None, "mte1": None, "mmad": None, "fix_store": None,
            "l1_slots": [None] * p.l1_buffers,
            "l0_slots": [None] * p.l0_buffers,
            "c_slots": [None] * p.c_buffers,
            "macro_index": 0, "micro_index": 0, "tile_index": 0,
        } for _ in range(min(p.core_count, tile_count))]
        total_input = total_mte1 = total_output = total_flops = 0
        owners: list[int] = []
        local_peak = peak / p.core_count
        for tile in range(tile_count):
            core = tile % p.core_count
            owners.append(core)
            state = states[core]
            row, col = divmod(tile, tiles_n)
            valid_m = min(p.tile_m, m - row * p.tile_m)
            valid_n = min(p.tile_n, n - col * p.tile_n)
            prefix = f"core{core}.tile{tile}"
            c_slot = state["tile_index"] % p.c_buffers
            c_free = state["c_slots"][c_slot]
            for macro_start in range(0, k, p.l1_k):
                valid_macro_k = min(p.l1_k, k - macro_start)
                l1_slot = state["macro_index"] % p.l1_buffers
                load = f"{prefix}.k{macro_start}.mte2"
                input_bytes = (valid_m + valid_n) * valid_macro_k * 2
                stages.append(ExecutionStage(
                    name=load,
                    demands=(("hbm", float(input_bytes)), ("mte2", float(input_bytes))),
                    deps=_dependencies(state["mte2"], state["l1_slots"][l1_slot]),
                    minimum_duration_ms=1000.0 * max(
                        input_bytes / p.l2_bytes_per_s_per_core,
                        input_bytes / p.mte2_bytes_per_s_per_core,
                    ),
                    rate_limits=(("mte2", min(
                        p.l2_bytes_per_s_per_core,
                        p.mte2_bytes_per_s_per_core,
                    ) / 1000.0),),
                    notes=(f"core={core}", f"l1_slot={l1_slot}", "hbm_cache_credit=none"),
                ))
                state["mte2"] = load
                total_input += input_bytes
                for sub_start in range(0, valid_macro_k, p.tile_k):
                    valid_k = min(p.tile_k, valid_macro_k - sub_start)
                    l0_slot = state["micro_index"] % p.l0_buffers
                    key = f"{prefix}.k{macro_start + sub_start}"
                    transfer = f"{key}.mte1"
                    a_bytes = valid_m * valid_k * 2
                    b_bytes = valid_n * valid_k * 2
                    stages.append(ExecutionStage(
                        name=transfer, demands=(("mte1", float(a_bytes + b_bytes)),),
                        deps=_dependencies(load, state["mte1"], state["l0_slots"][l0_slot]),
                        minimum_duration_ms=1000.0 * (
                            a_bytes / p.mte1_a_bytes_per_s_per_core
                            + b_bytes / p.mte1_b_bytes_per_s_per_core
                        ),
                        rate_limits=(("mte1", (a_bytes + b_bytes) / (
                            1000.0 * (a_bytes / p.mte1_a_bytes_per_s_per_core
                                      + b_bytes / p.mte1_b_bytes_per_s_per_core)
                        )),),
                        notes=(f"core={core}", f"l0_slot={l0_slot}",
                               "masked_padding_zero_fill_in_mte1_service"),
                    ))
                    state["mte1"] = transfer
                    total_mte1 += a_bytes + b_bytes
                    compute = f"{key}.mmad"
                    flops = 2 * p.tile_m * p.tile_n * p.tile_k
                    stages.append(ExecutionStage(
                        name=compute, demands=(("cube", float(flops)),),
                        deps=_dependencies(transfer, state["mmad"], c_free),
                        minimum_duration_ms=flops / local_peak * 1000.0,
                        rate_limits=(("cube", local_peak / 1000.0),),
                        notes=(f"core={core}", f"c_slot={c_slot}",
                               "issued_full_microtile_with_masked_tail"),
                    ))
                    state["mmad"] = compute
                    state["l0_slots"][l0_slot] = compute
                    state["micro_index"] += 1
                    total_flops += flops
                # MTE1's last consumption releases this L1 macro slot.  The
                # MMAD itself retains only its own L0 slot, not the L1 input.
                state["l1_slots"][l1_slot] = state["mte1"]
                state["macro_index"] += 1
            output_bytes = valid_m * valid_n * 2
            store = f"{prefix}.fix_store"
            stages.append(ExecutionStage(
                name=store,
                demands=(("hbm", float(output_bytes)), ("mte3", float(output_bytes))),
                deps=_dependencies(state["mmad"], state["fix_store"]),
                minimum_duration_ms=1000.0 * max(
                    output_bytes / p.fix_bytes_per_s_per_core,
                    output_bytes / p.l2_bytes_per_s_per_core,
                    output_bytes / p.mte3_bytes_per_s_per_core,
                ),
                rate_limits=(("mte3", min(
                    p.fix_bytes_per_s_per_core,
                    p.l2_bytes_per_s_per_core,
                    p.mte3_bytes_per_s_per_core,
                ) / 1000.0),),
                notes=(f"core={core}", "fp32_accumulator_to_bf16_output",
                       "fix_and_store_joint_service_not_two_additive_transfers"),
            ))
            total_output += output_bytes
            state["fix_store"] = store
            state["c_slots"][c_slot] = store
            state["tile_index"] += 1
        certificate = {
            "tile": {"m": p.tile_m, "n": p.tile_n, "k": p.tile_k, "l1_k": p.l1_k},
            "fragment_shape_guard": [16, 16, 16] if ascend_fragments else None,
            "buffers": {"l1": p.l1_buffers, "l0_ab": p.l0_buffers, "l0_c": p.c_buffers},
            "required_bytes_per_core": required,
            "capacity_bytes_per_core": {key: getattr(p, key) for key in required},
            "shape": {"m": m, "n": n, "k": k},
            "output_tiles": tile_count,
            "active_cores": min(tile_count, p.core_count),
            "tile_owners": owners,
            "tiles_per_core": [
                (tile_count - 1 - core) // p.core_count + 1
                for core in range(min(tile_count, p.core_count))
            ],
            "logical_flops": 2 * m * n * k,
            "issued_flops": total_flops,
            "traffic_bytes": {
                "hbm_read": total_input, "hbm_write": total_output,
                "l2_to_l1": total_input, "l1_to_l0": total_mte1,
                "l0c_to_output": total_output,
            },
        }
        provenance = {
            "source_class": "uncalibrated_analytic",
            "profile_source": p.source,
            "profile": asdict(p),
            "machine_source": die.source,
            "compute_peak_flops_per_s": peak,
            "hbm_bw_bytes_per_s": die.hbm_bw_bytes_per_s,
            "aggregate_mte_bw_bytes_per_s": die.mte_bw_bytes_per_s,
            "timing_scope": context.timing_scope,
            "assumptions": [
                "one_explicit_dense_bf16_program_not_optimal_tiling",
                ("bf16_16_lane_fragment_shape_guard_not_kernel_certification"
                 if ascend_fragments else
                 "unrecognized_architecture_arithmetic_fixture_no_ascend_fragment_legality_claim"),
                "cyclic_output_tile_ownership",
                "every_output_tile_reloads_inputs_no_hbm_cache_credit",
                "mte1_waits_for_entire_mte2_macro_panel_before_consumption",
                "full_microtile_arithmetic_masked_input_output_tails",
                "masked_zero_fill_cost_included_in_declared_mte1_service",
                "per_core_pipeline_rates_are_fixed_resource_rate_limits",
                "l2_folded_into_local_copy_rate_no_shared_l2_cache_contention_model",
                "fix_conversion_and_output_copy_are_one_joint_service",
                "no_host_submission_or_dispatch_charge_in_plan",
            ],
        }
        return KernelEstimate(plan=KernelPlan(
            backend="gemm_finite_pipeline", version="1", stages=tuple(stages),
            provenance=provenance, buffer_certificate=certificate,
            includes_dispatch=False,
        ))


__all__ = ["GemmProfile", "GemmProvider"]
