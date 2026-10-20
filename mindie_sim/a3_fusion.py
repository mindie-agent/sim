"""Executable semantic-DAG finite fusion candidates.

[Role]
  Transform an unfused ``IterationWork`` op DAG into legal implementation
  candidates.  Attention becomes a finite online-softmax KV pipeline,
  RMSNorm+GEMM becomes row/weight/output tile stages, and the remaining
  rules emit real dependent resource phases instead of a scalar serial cost.

[Invariants]
  * A transformation is applied only when structural/dtype/layout/ownership
    conditions hold; no model-name constants or global speedup.
  * Fused stages conserve the original useful FLOPs and necessary traffic.  A
    removed intermediate is justified by an explicit resident buffer with a
    last-consumer release; extra recurrence/reread work and bytes are charged.
  * Multiple consumers block fusion.  Escaping normalized values or score
    tensors prevent deletion.  MLA absorption stays a consideration, never an
    assumed equivalent.
  * Every fused task carries explicit stage extents, resource work, buffer
    lifetimes, finite slot counts and per-pool admission.  A unified scratch
    number is not called a physical L1/UB/L0 proof.
  * Downstream dependencies are remapped to the fused producer and external
    predecessors of every group member are retained.

[Timeouts & Locks]
  Pure DAG rewrite bounded by op count and finite tile enumeration.

[Side-Effect Policy]
  No I/O, device or global mutation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from .a3_fusion_attention import plan_attention_pipeline
from .a3_fusion_common import (
    MemoryBudget, PipelinePlan, assign_stage_cores,
)
from .a3_fusion_multiworker import (
    plan_attention_multiworker, plan_norm_multiworker,
)
from .a3_fusion_placement import place_full_work_fusions
from .a3_fusion_norm import plan_norm_gemm_pipeline
from .a3_ops import IterationWork, OpSpec
from .a3_types import dtype_bytes


@dataclass(frozen=True)
class FusionResult:
    work: IterationWork
    applied_rules: tuple[str, ...]
    considered_rules: tuple[dict[str, Any], ...]
    removed_intermediates: tuple[str, ...]
    retained_intermediates: tuple[str, ...]
    changes: dict[str, Any]
    witness: dict[str, Any]
    legal: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied_rules": list(self.applied_rules),
            "considered_rules": [dict(item) for item in self.considered_rules],
            "removed_intermediates": list(self.removed_intermediates),
            "retained_intermediates": list(self.retained_intermediates),
            "changes": dict(self.changes),
            "witness": dict(self.witness),
            "legal": self.legal,
        }


_METRICS = ("flops", "hbm_bytes", "mte2_bytes", "mte3_bytes", "vector_ops")


def _consumer_count(ops: Sequence[OpSpec]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for op in ops:
        for dep in op.deps:
            counts[dep] = counts.get(dep, 0) + 1
    return counts


def _group_attention(ops: Sequence[OpSpec]) -> list[tuple[OpSpec, OpSpec, OpSpec]]:
    groups = []
    for qk in ops:
        if qk.family != "attention_qk":
            continue
        qk_consumers = [op for op in ops if qk.op_id in op.deps]
        if len(qk_consumers) != 1:
            continue
        soft = qk_consumers[0]
        if soft.family != "attention_softmax":
            continue
        pv_consumers = [op for op in ops if op.deps == (soft.op_id,)]
        if len(pv_consumers) != 1 or pv_consumers[0].family != "attention_pv":
            continue
        groups.append((qk, soft, pv_consumers[0]))
    return groups


def _same_tensor_contract(a: OpSpec, b: OpSpec) -> bool:
    return a.layout == b.layout and a.dtype == b.dtype


def _totals(group: Sequence[OpSpec]) -> dict[str, float]:
    return {
        key: float(sum(float(getattr(item, key, 0.0) or 0.0) for item in group))
        for key in _METRICS + ("link_bytes",)
    }


def _deps_union(group: Sequence[OpSpec]) -> tuple[str, ...]:
    internal = {item.op_id for item in group}
    external: list[str] = []
    seen: set[str] = set()
    for item in group:
        for dep in item.deps:
            if dep not in internal and dep not in seen:
                seen.add(dep)
                external.append(dep)
    return tuple(external)


def _fuse_group(
    group: Sequence[OpSpec],
    *,
    rule: str,
    family: str,
    pipeline: PipelinePlan | None = None,
    stages: Sequence[Mapping[str, Any]] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    serial_ms: float = 0.0,
) -> OpSpec:
    leader = group[0]
    tail = group[-1]
    original = _totals(group)
    if pipeline is not None:
        fused_totals = pipeline.totals()
        fused_totals.setdefault("link_bytes", original["link_bytes"])
        stage_payload = tuple(dict(stage) for stage in pipeline.stages)
        peak = float(sum(pipeline.required_by_pool.values()))
        buffer_plan = tuple(dict(item) for item in pipeline.buffers)
        pipeline_meta = pipeline.to_metadata()
    else:
        stage_payload = tuple(dict(stage) for stage in (stages or ()))
        sums = {
            key: float(sum(float(stage.get(key, 0.0) or 0.0) for stage in stage_payload))
            for key in _METRICS
        }
        fused_totals = {
            **original,
            **{key: sums[key] for key in _METRICS},
        }
        buffer_plan = ()
        peak = float(extra_metadata.get("tile_scratch_bytes", 0.0)) if extra_metadata else 0.0
        pipeline_meta = {
            "fusion_stages": stage_payload,
            "buffer_plan": (),
            "peak_live_bytes": int(round(peak)),
            "required_by_pool": {},
            "stage_count": len(stage_payload),
            "buffer_count": 0,
        }
    tile_m = max(1, int(leader.launch_shape[0])) if leader.launch_shape else 1
    tile_n = max(1, int(leader.launch_shape[1])) if len(leader.launch_shape) > 1 else 1
    stage_count = len(stage_payload)
    metadata = {
        **dict(leader.metadata),
        "fusion_rule": rule,
        "fused_ops": ",".join(item.op_id for item in group),
        "removed_intermediates": ",".join(item.op_id for item in group[1:]),
        "launch_count": 1,
        "stage_count": stage_count,
        "physical_transfer_bytes": float(sum(
            float(value or 0.0)
            for stage in stage_payload
            for key, value in dict(stage.get("resources", {})).items()
            if str(key) == "mte1"
        )),
        "pipeline_stages": max(1, min(stage_count, 2)),
        "startup_launches": 1 if stage_count > 1 else 0,
        "drain_launches": 1 if stage_count > 1 else 0,
        "serial_ms": float(serial_ms),
        "tile_scratch_bytes": int(round(peak)),
        "peak_live_bytes": int(round(peak)),
        "double_buffer": bool(
            dict(pipeline_meta.get("pipeline_notes", {})).get("finite_kv_slots", 0) > 1
            or dict(pipeline_meta.get("pipeline_notes", {})).get("finite_group_slots", 0) > 1
            or dict(extra_metadata or {}).get("double_buffer", False)
        ),
        "tile_count": int(dict(extra_metadata or {}).get("tile_count", stage_count)),
        "tile_shape": list(
            dict(extra_metadata or {}).get("tile_shape", (min(64, tile_m), min(64, tile_n)))
        ),
        "tail_shape": list(
            dict(extra_metadata or {}).get(
                "tail_shape",
                (tile_m % 64 or (64 if tile_m else 0),
                 tile_n % 64 or (64 if tile_n else 0)),
            )
        ),
        "fusion_stages": stage_payload,
        "buffer_plan": buffer_plan,
        "peak_live_bytes_reconstructable": True,
        "required_by_pool": dict(pipeline_meta.get("required_by_pool", {})),
        "pipeline_kind": pipeline_meta.get("pipeline_kind", rule),
        "work_conservation": {
            "original": {key: original[key] for key in _METRICS},
            "fused": {key: fused_totals[key] for key in _METRICS},
            "stage_sums": {
                key: float(sum(stage.get(key, 0.0) or 0.0 for stage in stage_payload))
                for key in _METRICS
            },
            "flops_preserved": (
                fused_totals["flops"] + 1e-9 >= original["flops"]
            ),
            "necessary_bytes_not_reduced_without_buffer": True,
        },
        **dict(pipeline_meta.get("pipeline_notes", {})),
        **dict(extra_metadata or {}),
    }
    return replace(
        leader,
        op_id=f"fused:{leader.op_id}",
        family=family,
        deps=_deps_union(group),
        flops=fused_totals["flops"],
        hbm_bytes=fused_totals["hbm_bytes"],
        mte2_bytes=fused_totals["mte2_bytes"],
        mte3_bytes=fused_totals["mte3_bytes"],
        vector_ops=fused_totals["vector_ops"],
        link_bytes=fused_totals["link_bytes"],
        recipe=f"fused:{rule}",
        coverage="fused",
        metadata=tuple(sorted(metadata.items())),
        notes=tuple(dict.fromkeys(leader.notes + (
            f"fused_by:{rule}", "finite_dependent_stages", "intermediate_kept_on_chip",
        ))),
    )


def _apply_rule(
    *,
    group: Sequence[OpSpec],
    rule: str,
    family: str,
    pipeline: PipelinePlan | None = None,
    stages: Sequence[Mapping[str, Any]] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    serial_ms: float = 0.0,
) -> tuple[OpSpec, dict[str, float], dict[str, Any]]:
    fused = _fuse_group(
        group, rule=rule, family=family, pipeline=pipeline, stages=stages,
        extra_metadata=extra_metadata, serial_ms=serial_ms,
    )
    original = _totals(group)
    fused_totals = _totals((fused,))
    changes = {
        key: fused_totals[key] - original[key] for key in _METRICS
    }
    changes["launch"] = -(len(group) - 1)
    changes["serial_ms"] = -serial_ms
    changes["peak_scratch_bytes"] = float(
        dict(fused.metadata).get("peak_live_bytes", 0.0) or 0.0
    )
    return fused, changes, {
        "original": original,
        "fused": fused_totals,
    }


def _budget(
    scratch_capacity_bytes: int | None,
    pool_capacities: Mapping[str, int] | None,
) -> MemoryBudget:
    return MemoryBudget.from_declared(scratch_capacity_bytes, pool_capacities)


def apply_fusion(
    work: IterationWork,
    *,
    enabled: bool = True,
    model_features: Mapping[str, Any] | None = None,
    scratch_capacity_bytes: int | None = None,
    cube_flops_per_s: float | None = None,
    vector_ops_per_s: float | None = None,
    pool_capacities: Mapping[str, int] | None = None,
    core_count: int | None = None,
    per_core_rates: Mapping[str, float] | None = None,
    core_base: int | None = None,
    worker_count: int = 1,
    require_physical: bool = False,
) -> FusionResult:
    """Apply conservative finite fusion rules to a work DAG.

    ``scratch_capacity_bytes`` remains the existing product-facing unified
    capacity.  ``pool_capacities`` optionally declares physical pool budgets
    (``l1``, ``ub``, ``l0a``/``l0b``/``l0c``); when absent the witness records
    that admission is unified-capacity-only and not physical-pool proof.
    ``core_count`` and ``per_core_rates`` add one-tile-one-core assignments;
    ``require_physical=True`` rejects a non-positive physical L0 path.
    """
    ops = list(work.ops)
    budget = _budget(scratch_capacity_bytes, pool_capacities)
    budget_details = budget.details({})
    physical = budget.declares_positive(("l1", "ub", "l0a", "l0b", "l0c"))

    def memory_details(required: Mapping[str, float]) -> dict[str, Any]:
        details = budget.details(required)
        if budget.pools and not physical:
            details = dict(details)
            details["scope"] = (
                "declared_pools_but_missing_positive_physical_l0_path"
            )
            details["physical_feasible"] = False
        return details
    if not enabled or not ops:
        return FusionResult(
            work=work, applied_rules=(), considered_rules=(),
            removed_intermediates=(), retained_intermediates=tuple(op.op_id for op in ops),
            changes={"flops": 0.0, "hbm_bytes": 0.0, "mte2_bytes": 0.0,
                     "vector_ops": 0.0, "launch": 0},
            witness={"nodes": [op.op_id for op in ops], "edges": [],
                     "scratch_pool": budget_details}, legal=True,
        )
    consumers = _consumer_count(ops)
    skipped: set[str] = set()
    remap: dict[str, str] = {}
    member_to_fused: dict[str, str] = {}
    replacements: dict[str, OpSpec] = {}
    applied: list[str] = []
    considered: list[dict[str, Any]] = []
    removed: list[str] = []
    changes = {"flops": 0.0, "hbm_bytes": 0.0, "mte2_bytes": 0.0,
               "mte3_bytes": 0.0, "vector_ops": 0.0, "launch": 0,
               "peak_scratch_bytes": 0.0, "serial_ms": 0.0}

    def accept(group: Sequence[OpSpec], fused: OpSpec, rule: str,
               rule_changes: Mapping[str, float], metadata: Mapping[str, Any]) -> None:
        replacements[fused.op_id] = fused
        for item in group:
            skipped.add(item.op_id)
            remap[item.op_id] = fused.op_id
            member_to_fused[item.op_id] = fused.op_id
        removed.extend(
            item for item in str(
                dict(fused.metadata).get("removed_intermediates", "")
            ).split(",") if item
        )
        applied.append(rule)
        for key in ("flops", "hbm_bytes", "mte2_bytes", "mte3_bytes",
                    "vector_ops", "launch", "serial_ms"):
            changes[key] += float(rule_changes.get(key, 0.0))
        changes["peak_scratch_bytes"] = max(
            changes["peak_scratch_bytes"],
            float(dict(fused.metadata).get("peak_live_bytes", 0.0) or 0.0),
        )
        considered.append({
            "rule": rule, "eligible": True,
            "reason": metadata.get("reason", "finite_stage_pipeline"),
            "fused_op": fused.op_id,
            **{key: value for key, value in metadata.items() if key != "reason"},
        })

    # 1) Attention: finite online-softmax KV tiles with query/head groups.
    for group in _group_attention(ops):
        if any(item.op_id in skipped for item in group):
            continue
        qk, soft, pv = group
        if consumers.get(soft.op_id, 0) != 1 or pv.op_id in skipped:
            continue
        if not (_same_tensor_contract(qk, pv)
                and qk.launch_shape[:2] == pv.launch_shape[:2]
                and len(qk.launch_shape) >= 4 and len(pv.launch_shape) >= 4
                and soft.deps == (qk.op_id,) and pv.deps == (soft.op_id,)):
            considered.append({
                "rule": "attention_qk_softmax_pv", "eligible": False,
                "reason": "extent_layout_dtype_or_dependency_mismatch",
            })
            continue
        active_workers = max(1, int(worker_count))
        if core_count is not None:
            active_workers = min(active_workers, max(1, int(core_count)))
        if active_workers > 1:
            pipeline, rejection = plan_attention_multiworker(
                qk, soft, pv, budget=budget, physical=physical,
                workers=active_workers,
            )
        else:
            pipeline, rejection = plan_attention_pipeline(
                qk, soft, pv, budget=budget, physical=physical,
            )
        if pipeline is None:
            considered.append({
                "rule": "attention_qk_softmax_pv", "eligible": False,
                **rejection,
            })
            continue
        core_contract = assign_stage_cores(
            pipeline, core_count=core_count, per_core_rates=per_core_rates,
            core_base=core_base,
        )
        physical_feasible = bool(
            physical and pipeline.notes.get("physical_path")
            and memory_details(pipeline.required_by_pool)["fits"]
        )
        if require_physical and not physical_feasible:
            considered.append({
                "rule": "attention_qk_softmax_pv", "eligible": False,
                "reason": "physical_pool_path_unavailable",
                "budget": memory_details(pipeline.required_by_pool),
                "physical_mode": physical,
            })
            continue
        metadata = {
            "reason": "finite_online_softmax_tiles",
            "tile_count": int(pipeline.notes.get("tiling", {}).get(
                "q_tiles", 1
            )) * int(pipeline.notes.get("tiling", {}).get("kv_tiles", 1))
            * int(pipeline.notes.get("tiling", {}).get("head_tiles", 1)),
            "tile_shape": (
                pipeline.notes.get("tiling", {}).get("q_tile"),
                pipeline.notes.get("tiling", {}).get("kv_tile"),
                pipeline.notes.get("tiling", {}).get("head_tile"),
            ),
            "tail_shape": [
                pipeline.notes.get("tiling", {}).get("query_tail"),
                pipeline.notes.get("tiling", {}).get("kv_tail"),
                pipeline.notes.get("tiling", {}).get("head_tail"),
            ],
            "tile_source": "qk_softmax_pv",
            "live_score_buffer_bytes": (
                pipeline.notes.get("tiling", {}).get("q_tile", 1)
                * pipeline.notes.get("tiling", {}).get("kv_tile", 1)
                * pipeline.notes.get("tiling", {}).get("head_tile", 1)
                * 4.0
            ),
            "stage_count": len(pipeline.stages),
            "required_by_pool": dict(pipeline.required_by_pool),
            "budget": memory_details(pipeline.required_by_pool),
            "memory_scope": memory_details(pipeline.required_by_pool)["scope"],
            "physical_mode": bool(physical),
            "physical_feasible": physical_feasible,
            "physical_path": pipeline.notes.get("physical_path"),
            "worker_count": int(
                pipeline.notes.get("active_workers", 1) or 1
            ),
            "worker_partition": pipeline.notes.get(
                "partition", "single_worker"
            ),
            "core_contract": core_contract,
            "pool_scope_note": (
                "unified scratch is not physical L1/UB/L0 proof; declared "
                "pool_capacities are consumed when supplied"
            ),
        }
        fused, rule_changes, conservation = _apply_rule(
            group=group, rule="attention_qk_softmax_pv",
            family="attention_fused", pipeline=pipeline,
            extra_metadata=metadata,
        )
        metadata["conservation"] = conservation
        accept(group, fused, "attention_qk_softmax_pv", rule_changes, metadata)

    # 2) RMSNorm + single-consumer GEMM with full-K row tiling.
    for norm in ops:
        if norm.family != "rmsnorm" or norm.op_id in skipped:
            continue
        users = [op for op in ops if norm.op_id in op.deps]
        if len(users) != 1 or users[0].family not in {
            "qkv_projection", "gate_up_projection", "mla_q_a", "mla_kv_a",
        }:
            considered.append({
                "rule": "rmsnorm_gemm_epilogue", "eligible": False,
                "reason": "multiple_or_non_gemm_consumers",
            })
            continue
        if users[0].deps != (norm.op_id,):
            continue
        gemm = users[0]
        if gemm.op_id in skipped:
            continue
        norm_elements = 1
        for value in norm.launch_shape:
            norm_elements *= max(1, int(value))
        expected_elements = (
            int(gemm.launch_shape[0]) * int(gemm.launch_shape[2])
            if len(gemm.launch_shape) >= 3 else 0
        )
        quantized_weight = str((model_features or {}).get("weight_dtype", "")).lower()
        if quantized_weight and not quantized_weight.startswith("bf16") and not quantized_weight.startswith("fp16"):
            considered.append({
                "rule": "rmsnorm_gemm_epilogue", "eligible": False,
                "reason": "no_explicit_quant_scale_epilogue_for_weight_format",
                "weight_dtype": quantized_weight,
            })
            continue
        if not (_same_tensor_contract(norm, gemm)
                and norm.launch_shape and gemm.launch_shape
                and norm_elements == expected_elements
                and gemm.deps == (norm.op_id,)):
            considered.append({
                "rule": "rmsnorm_gemm_epilogue", "eligible": False,
                "reason": "extent_layout_dtype_or_dependency_mismatch",
            })
            continue
        active_workers = max(1, int(worker_count))
        if core_count is not None:
            active_workers = min(active_workers, max(1, int(core_count)))
        if active_workers > 1:
            pipeline, rejection = plan_norm_multiworker(
                norm, gemm, budget=budget, physical=physical,
                workers=active_workers,
            )
        else:
            pipeline, rejection = plan_norm_gemm_pipeline(
                norm, gemm, budget=budget, physical=physical,
            )
        if pipeline is None:
            considered.append({
                "rule": "rmsnorm_gemm_epilogue", "eligible": False,
                **rejection,
            })
            continue
        core_contract = assign_stage_cores(
            pipeline, core_count=core_count, per_core_rates=per_core_rates,
            core_base=core_base,
        )
        physical_feasible = bool(
            physical and pipeline.notes.get("physical_path")
            and memory_details(pipeline.required_by_pool)["fits"]
        )
        if require_physical and not physical_feasible:
            considered.append({
                "rule": "rmsnorm_gemm_epilogue", "eligible": False,
                "reason": "physical_pool_path_unavailable",
                "budget": memory_details(pipeline.required_by_pool),
                "physical_mode": physical,
            })
            continue
        tiling = dict(pipeline.notes.get("tiling", {}))
        metadata = {
            "reason": "full_k_row_tiles_with_bf16_boundary",
            "tile_count": int(tiling.get("row_tiles", 1))
            * int(tiling.get("n_tiles", 1)) * int(tiling.get("k_tiles", 1)),
            "tile_shape": [tiling.get("row_tile"), tiling.get("n_tile"),
                           tiling.get("k_tile")],
            "tail_shape": [tiling.get("row_tail"), tiling.get("n_tail"),
                           tiling.get("k_tail")],
            "double_buffer": (
                int(tiling.get("row_slots", 1) or 1) > 1
                or int(tiling.get("weight_slots", 1) or 1) > 1
            ),
            "stage_count": len(pipeline.stages),
            "required_by_pool": dict(pipeline.required_by_pool),
            "budget": memory_details(pipeline.required_by_pool),
            "memory_scope": memory_details(pipeline.required_by_pool)["scope"],
            "physical_mode": bool(physical),
            "physical_feasible": physical_feasible,
            "physical_path": pipeline.notes.get("physical_path"),
            "worker_count": int(
                pipeline.notes.get("active_workers", 1) or 1
            ),
            "worker_partition": pipeline.notes.get(
                "partition", "single_worker"
            ),
            "core_contract": core_contract,
            "pool_scope_note": (
                "unified scratch is not physical L1/UB/L0 proof; declared "
                "pool_capacities are consumed when supplied"
            ),
        }
        fused, rule_changes, conservation = _apply_rule(
            group=(gemm, norm), rule="rmsnorm_gemm_epilogue",
            family="fused_norm_gemm", pipeline=pipeline,
            extra_metadata=metadata,
        )
        metadata["conservation"] = conservation
        accept(group=(norm, gemm), fused=fused, rule="rmsnorm_gemm_epilogue",
               rule_changes=rule_changes, metadata=metadata)

    # 3) MLA absorption eligibility (never assumed equivalent).
    mla_q = next((op for op in ops if op.family == "mla_q_b"), None)
    mla_kv = next((op for op in ops if op.family == "mla_kv_b"), None)
    if mla_q is not None and mla_kv is not None:
        qdim = int(mla_q.launch_shape[-1]) if mla_q.launch_shape else 0
        kvdim = int(mla_kv.launch_shape[-1]) if mla_kv.launch_shape else 0
        eligible = (
            qdim == kvdim and mla_q.dtype == mla_kv.dtype
            and mla_q.layout == mla_kv.layout and qdim > 0
        )
        considered.append({
            "rule": "mla_projection_absorption", "eligible": eligible,
            "reason": "projected_dims_and_precision_match" if eligible
            else "projected_dims_or_precision_differ",
            "q_dim": qdim, "kv_dim": kvdim,
        })

    # Rebuild the DAG with remapped dependencies, inserting each fused task at
    # the position of its first group member and then topologically ordering.
    final: list[OpSpec] = []
    inserted: set[str] = set()
    for op in ops:
        if op.op_id in skipped:
            fused_id = member_to_fused.get(op.op_id)
            if fused_id is not None and fused_id not in inserted:
                fused = replacements[fused_id]
                final.append(replace(
                    fused,
                    deps=tuple(remap.get(dep, dep) for dep in fused.deps),
                ))
                inserted.add(fused_id)
            continue
        deps = tuple(remap.get(dep, dep) for dep in op.deps)
        final.append(replace(op, deps=deps) if deps != op.deps else op)
    edges = [
        (dep, op.op_id) for op in final for dep in op.deps
        if dep in {item.op_id for item in final}
    ]
    by_id = {op.op_id: op for op in final}
    original_index = {op.op_id: index for index, op in enumerate(final)}
    indegree = {op.op_id: 0 for op in final}
    children: dict[str, list[str]] = {op.op_id: [] for op in final}
    for op in final:
        for dep in op.deps:
            if dep in by_id:
                indegree[op.op_id] += 1
                children[dep].append(op.op_id)
    ready = sorted(
        (key for key, degree in indegree.items() if degree == 0),
        key=lambda key: original_index[key],
    )
    ordered_final: list[OpSpec] = []
    while ready:
        current = ready.pop(0)
        ordered_final.append(by_id[current])
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort(key=lambda key: original_index[key])
    if len(ordered_final) != len(final):
        return FusionResult(
            work=work, applied_rules=(), considered_rules=tuple(considered),
            removed_intermediates=(), retained_intermediates=tuple(op.op_id for op in work.ops),
            changes={}, witness={"valid_contraction": False, "reason": "cycle"},
            legal=False,
        )
    final = ordered_final
    final, placement = place_full_work_fusions(
        final, core_count=core_count, core_base=core_base,
        fixed_core_base=core_base is not None,
    )
    # Placement may add predecessor edges for reused worker bundles; the
    # witness must expose exactly the rewritten DAG.
    node_ids_after = {op.op_id for op in final}
    edges = [
        (dep, op.op_id) for op in final for dep in op.deps
        if dep in node_ids_after
    ]
    result_work = replace(work, ops=tuple(final), notes=tuple(dict.fromkeys(
        work.notes + ("fusion_candidate",)
    )))
    fused_tasks = {
        op.op_id: dict(op.metadata) for op in final if op.coverage == "fused"
    }
    for entry in considered:
        fused_id = entry.get("fused_op")
        if fused_id in fused_tasks:
            updated = fused_tasks[fused_id]
            entry["core_contract"] = updated.get("core_contract")
            entry["full_work_placement"] = updated.get("full_work_placement")
    per_task_pools = {
        task_id: dict(dict(metadata).get("required_by_pool", {}))
        for task_id, metadata in fused_tasks.items()
    }
    aggregate_pool_requirements: dict[str, float] = {}
    for required in per_task_pools.values():
        for pool, amount in required.items():
            aggregate_pool_requirements[pool] = max(
                aggregate_pool_requirements.get(pool, 0.0), float(amount)
            )
    scratch_pool = memory_details(aggregate_pool_requirements)
    scratch_pool["witness_pool_requirement_is_max_per_task"] = True
    scratch_pool["per_fused_task_required_by_pool"] = per_task_pools
    witness = {
        "scratch_capacity_bytes": scratch_capacity_bytes,
        "pool_capacities": dict(pool_capacities or {}),
        "scratch_pool": scratch_pool,
        "memory_scope": scratch_pool["scope"],
        "physical_mode": bool(physical),
        "physical_feasible_by_task": {
            task_id: bool(dict(metadata).get("physical_feasible", False))
            for task_id, metadata in fused_tasks.items()
        },
        "core_contract": {
            "core_count": core_count,
            "core_base": core_base,
            "worker_count": int(worker_count),
            "per_core_rates": dict(per_core_rates or {}),
            "scope_note": (
                "one emitted tile per core; independent tiles may share the die; "
                "no multi-core partition is claimed without explicit reduction"
            ),
        },
        "pool_scope_note": (
            "unified scratch is a bounded per-kernel admission count, not "
            "physical L1/UB/L0 proof; per-pool capacities are required for that "
            "claim and are consumed when supplied"
        ),
        "per_fused_task_required_by_pool": per_task_pools,
        "full_work_placement": placement,
        "model_features": dict(model_features or {}),
        "nodes": [op.op_id for op in final],
        "edges": [{"from": src, "to": dst} for src, dst in edges],
        "fused_tasks": fused_tasks,
        "stage_contract": "add_costed_ops: fusion_stages with integer deps",
    }
    return FusionResult(
        work=result_work,
        applied_rules=tuple(dict.fromkeys(applied)),
        considered_rules=tuple(considered),
        removed_intermediates=tuple(dict.fromkeys(removed)),
        retained_intermediates=tuple(
            op.op_id for op in final if op.family != "attention_fused"
        ),
        changes={key: round(value, 9) for key, value in changes.items()},
        witness=witness,
        legal=True,
    )


__all__ = ["FusionResult", "apply_fusion"]
