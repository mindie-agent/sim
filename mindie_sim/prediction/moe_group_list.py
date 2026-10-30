"""Explicit count-to-prefix conversion at the fused INT8 expert caller.

[Role] Preserve the separate group-list producer required by the native ABI.
[Invariants] A prefix scan consumes all local expert descriptors, including
  zero-row experts. Only the fused gate/up consumes this converted list; the
  down projection retains its original count-list contract. No routing values
  are inferred from probability moments or observed durations.
[Timeouts & Locks] One bounded graph traversal per expert call; no I/O or locks.
[Side-Effect Policy] Returns immutable operators. Native AI_CPU service may be
  declared independently; the uncalibrated reference prices memory only.
"""
from dataclasses import replace

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from .mechanism import OperatorPlan, PlannedRecipe, Work
from .program import KernelProgram, Stage

IMPLEMENTATION = "vllm_ascend_group_list_prefix"
RECIPE = "expert_group_list_cumsum"
SOURCE = {
    "repository": "vllm-ascend",
    "revision": "8c30e4f491f73cba737a1dacc522894e437cdf3d",
    "caller": "vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py:408-415",
    "conversion": "vllm_ascend/ops/fused_moe/moe_utils.py:205-214",
    "operation": "group_list_type=1 -> type=0; group_list.cumsum(dim=0)",
    "native_binary_identity": "not_established",
}


def build(op, geometry):
    work = Work(resources={"hbm": op.hbm_bytes})
    return OperatorPlan(work, lambda context: KernelProgram((
        Stage("count_to_prefix_memory", context.bind(work)),)))


def recipes():
    return {RECIPE: PlannedRecipe(build)}


def lower(ops, bindings):
    from .implementations import select
    selected = select(bindings, "group_list_prefix")
    if selected is None:
        return tuple(ops)
    fusion = select(bindings, "native_fusion")
    if fusion is None or "grouped_gate_up_activation_quant" not in fusion.options.get("roles", ()):
        raise ModelingError("group_list_prefix_requires_fused_gate_up_quant_consumer")
    by_id = {op.op_id: op for op in ops}
    result = []
    for op in ops:
        meta = dict(op.metadata)
        if op.family != "grouped_gemm" or op.recipe not in {
                "grouped_gate_up", "balanced_grouped_gate_up"}:
            result.append(op)
            continue
        if meta.get("activation_format") != "int8" or meta.get("weight_format") != "int8":
            raise ModelingError("group_list_prefix_requires_selected_int8_fused_gate_up")
        # The count producer precedes optional packing/activation quantization.
        pending, visited, producers = list(op.deps), set(), []
        while pending:
            key = pending.pop()
            if key in visited or key not in by_id:
                continue
            visited.add(key)
            source = by_id[key]
            if source.layer != op.layer:
                continue
            sm = dict(source.metadata)
            if source.recipe == "route_pack" and sm.get("num_experts"):
                producers.append(source)
            else:
                pending.extend(source.deps)
        if len(producers) != 1:
            raise ModelingError("group_list_prefix_requires_one_local_count_producer")
        producer = producers[0]
        experts = dict(producer.metadata)["num_experts"]
        if meta.get("matrix_weight_bank_groups", experts) != experts:
            raise ModelingError("group_list_prefix_requires_local_count_list")
        name = producer.op_id + ".group_list_prefix"
        prefix = OpSpec(
            op_id=name, family="prefix_sum", phase=op.phase, stage=op.stage,
            layer=op.layer, requests=op.requests, deps=(producer.op_id,),
            semantic_shape=(experts,), local_shape=(experts,),
            padded_shape=(experts,), launch_shape=(1,), dtype="int64", layout="flat",
            recipe=RECIPE, hbm_bytes=experts * 16, mte2_bytes=0, mte3_bytes=0,
            source_boundary="kernel", stream=op.stream,
            metadata=(("source", SOURCE), ("native_kind", "Cumsum"),
                      ("source_execution_domain", "aicpu"),
                      ("logical_integer_adds", max(0, experts - 1)),
                      ("unpriced_service_work", {"aicpu_integer_adds": max(0, experts - 1)}),
                      ("input_group_list_type", 1), ("output_group_list_type", 0),
                      ("group_count", experts),
                      *tuple((k, v) for k, v in op.metadata if k in {
                          "graph_captured", "graph_segment", "structure_block", "structure_role"})),
        )
        result.extend((prefix, replace(op, deps=(*op.deps, name), metadata=tuple({
            **meta, "group_list_producer": name, "group_list_type": 0,
            "group_list_dtype": "int64", "group_list_conversion": SOURCE,
        }.items()))))
    return tuple(result)
