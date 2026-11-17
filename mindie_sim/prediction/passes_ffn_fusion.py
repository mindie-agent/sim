"""Physical FFN/MoE gated-activation, on-chip GMM+SwiGLU+Quant, and tail passes.

[Role]
  Rewrite semantic FFN and MoE operator DAGs for gated-activation + dynamic
  quantization fusion, on-chip GroupedMatmul+SwiGLU/SiTU+Quant epilogue fusion,
  and 3-way MoE shared+routed+residual tail combine.

[Invariants]
  * Preserves 100% of Cube FLOPs, scale epilogues, and Vector OPs; eliminates
    only genuinely fused intermediate HBM round-trips and kernel launches.
  * Matches structural operator families, recipes, and dependency edges, never
    model-name strings.

[Timeouts & Locks]
  Linear O(|V| + |E|) passes over one active layer graph; no I/O or locks.

[Side-Effect Policy]
  Pure functional transforms returning new immutable ``OpSpec`` sequences.
"""
from __future__ import annotations

from dataclasses import replace
import math

from .implementations import (
    append_audit, attach_quantized_ports, rewire_replacements,
)
from .passes_mhc_recurrent import merge_meta, users_map

NO_PORTS: dict = {}


def activation_quant_fusion(ops, source_dtype):
    """Fuse ``swiglu``/``situ``/``geglu`` + ``dynamic_quantize`` into one vector op."""
    del source_dtype
    by_id, users = {op.op_id: op for op in ops}, users_map(ops)
    replacements, removed, aliases = {}, set(), {}
    for quant in ops:
        if quant.plan_required or quant.family != "quantize" or len(quant.deps) != 1:
            continue
        act = by_id.get(quant.deps[0])
        if (act is None or act.plan_required or act.family not in {"swiglu", "situ", "geglu"}
                or act.stage != quant.stage or act.stream != quant.stream):
            continue
        if users[act.op_id] != [quant.op_id]:
            continue
        read = float(act.mte2_bytes)
        write = float(quant.mte3_bytes)
        meta = dict(merge_meta(act, quant, extra=(
            ("fused_from", (act.family, "quantize")),
            ("primary_output", "payload"),
        )))
        append_audit(
            meta, kind="transform_boundary", op_id=quant.op_id,
            boundary={
                "kind": "activation_quant",
                "replaces_work": True, "replaces_allocation": True,
                "replaced_occurrences": (act.op_id, quant.op_id),
                "replacement_occurrence": quant.op_id,
                "work": {"mte2_bytes": read, "mte3_bytes": write,
                         "hbm_bytes": read + write,
                         "vector_ops": float(act.vector_ops) + float(quant.vector_ops),
                         "flops": 0.0},
            },
            layer=quant.layer, stage=quant.stage,
        )
        replacements[quant.op_id] = replace(
            quant,
            deps=act.deps,
            family="quantize",
            recipe=f"{act.family}_quantize",
            vector_ops=float(act.vector_ops) + float(quant.vector_ops),
            mte2_bytes=read,
            mte3_bytes=write,
            hbm_bytes=read + write,
            metadata=tuple(meta.items()),
        )
        removed.add(act.op_id)
        aliases[act.op_id] = quant.op_id
    if not replacements:
        return list(ops)
    full_aliases = {old: old for old in replacements}
    full_aliases.update(aliases)
    rewired = rewire_replacements(
        ops, replacements=replacements, aliases=full_aliases,
        removed=removed, ports=NO_PORTS,
    )
    return attach_quantized_ports(rewired, replacements)


def gmm_swiglu_quant_fusion(ops, source_dtype):
    """Keep the ``grouped_gemm`` activation payload on chip for the fused quant epilogue.

    The gate/up matrix phase stops writing its ``[M, 2*I]`` payload to HBM and the
    activation+quantize phase stops reading it back, while every Cube FLOP,
    scale epilogue and Vector OP is preserved.  A single submitted native kernel
    additionally requires a source-verified ``declared_native_fusion`` plan
    binding; this pass never fabricates that binding, so it models the traffic
    elimination without claiming an unverified launch reduction.
    """
    ops = activation_quant_fusion(ops, source_dtype)
    by_id, users = {op.op_id: op for op in ops}, users_map(ops)
    replacements: dict = {}
    for tail in ops:
        if (tail.plan_required or not str(tail.recipe).endswith("_quantize")
                or tail.family != "quantize" or len(tail.deps) != 1):
            continue
        gate = by_id.get(tail.deps[0])
        if (gate is None or gate.plan_required or gate.family != "grouped_gemm"
                or gate.stage != tail.stage or gate.stream != tail.stream):
            continue
        if users[gate.op_id] != [tail.op_id]:
            continue
        if gate.op_id in replacements or tail.op_id in replacements:
            continue
        replacements[gate.op_id] = replace(
            gate,
            mte3_bytes=0.0,
            hbm_bytes=max(0.0, float(gate.hbm_bytes) - float(gate.mte3_bytes)),
            metadata=merge_meta(gate, extra=(
                ("onchip_handoff_to", tail.op_id),
                ("fusion_boundary", {
                    "implementation": "declared_onchip_handoff",
                    "native": "GroupedMatmulSwigluQuantOnChip",
                    "submitted_kernels": 2,
                    "semantic_phase_count": 2,
                    "timing_basis": "resource_DAG_not_sum_of_kernel_durations",
                    "launch_reduction_claimed": False,
                }),
            )),
        )
        replacements[tail.op_id] = replace(
            tail,
            mte2_bytes=0.0,
            hbm_bytes=float(tail.mte3_bytes),
            metadata=merge_meta(tail, extra=(
                ("onchip_handoff_from", gate.op_id),
                ("fused_from", ("grouped_gemm", tail.recipe)),
            )),
        )
    if not replacements:
        return list(ops)
    return [replacements.get(op.op_id, op) for op in ops]


def moe_tail_residual_fusion(ops, source_dtype):
    """Fold MoE ``shared + routed`` elementwise combine into post-MoE ``residual_add`` (PR #21999)."""
    del source_dtype
    by_id, users = {op.op_id: op for op in ops}, users_map(ops)
    replacements, removed, aliases = {}, set(), {}
    for res in ops:
        if res.recipe != "residual_add" or len(res.deps) != 2:
            continue
        comb = by_id.get(res.deps[0])
        if (comb is None or comb.plan_required or comb.family not in {"elementwise", "add"}
                or comb.recipe in {"residual_add"} or users[comb.op_id] != [res.op_id]
                or comb.stage != res.stage or comb.stream != res.stream
                or math.prod(comb.local_shape) != math.prod(res.local_shape)):
            continue
        # 3-way add (shared + routed + saved_residual): eliminates intermediate [T, H] write+read
        fused_read = max(0.0, float(comb.mte2_bytes) + float(res.mte2_bytes) - float(comb.mte3_bytes))
        fused_write = float(res.mte3_bytes)
        replacements[res.op_id] = replace(
            res,
            deps=tuple(dict.fromkeys((*comb.deps, res.deps[1]))),
            vector_ops=float(comb.vector_ops) + float(res.vector_ops),
            mte2_bytes=fused_read,
            mte3_bytes=fused_write,
            hbm_bytes=fused_read + fused_write,
            metadata=merge_meta(comb, res, extra=(("fused_moe_tail_add", True),)),
        )
        removed.add(comb.op_id)
        aliases[comb.op_id] = res.op_id
    if not replacements:
        return list(ops)
    full_aliases = {old: old for old in replacements}
    full_aliases.update(aliases)
    return rewire_replacements(
        ops, replacements=replacements, aliases=full_aliases,
        removed=removed, ports=NO_PORTS,
    )
