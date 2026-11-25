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
from typing import Literal

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.program import KernelProgram, Stage

TARGET_BACKEND: Literal["ascend"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="175ef4d49c2072f9233619636f0309f4f88b4e2c",
        relative_path="vllm_ascend/ops/fused_moe/moe_mlp.py",
        source_sha256="6b5f4f6d3259cb14e0613a6f7b1c2592331f143a14cf7240dd9d49a0a57dd53b",
    ),
    SourceRef(
        repository="Ascend/op-plugin",
        revision="48189acf95882e23fae56ea96ceeb89358cb42da",
        relative_path="op_plugin/ops/opapi/GroupedMatmulKernelNpuOpApi.cpp",
        source_sha256="a58dd51addfa3511735fcf1f46355a4597f62920cc5d24edc335b59610128fdc",
    ),
)

IMPLEMENTATION = "vllm_ascend_group_list_prefix"
RECIPE = "expert_group_list_cumsum"
_CUMSUM_DTYPE = "int64"
_CUMSUM_ELEMENT_BYTES = int(dtype_bytes(_CUMSUM_DTYPE))
_ARCH35_IDS = frozenset({"npu_arch_3510", "dav_3510"})
SOURCE = {
    "repository": "vllm-ascend",
    "revision": "8c30e4f491f73cba737a1dacc522894e437cdf3d",
    "caller": "vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py:408-415",
    "conversion": "vllm_ascend/ops/fused_moe/moe_utils.py:205-214",
    "operation": "group_list_type=1 -> type=0; group_list.cumsum(dim=0)",
    "native_binary_identity": "not_established",
}


def _is_arch35(geometry) -> bool:
    dp_arch = getattr(getattr(geometry, "datapath", None), "compute_arch", None)
    ca = dp_arch or getattr(geometry, "compute_arch", None)
    arch_id = str(getattr(ca, "arch_id", "") or getattr(ca, "preset", "") or "").strip().lower()
    return arch_id in _ARCH35_IDS


def build(op, geometry):
    if _is_arch35(geometry):
        experts = int(op.local_shape[0]) if op.local_shape else int(dict(op.metadata).get("group_count", 0))
        count_bytes = float(max(0, experts) * _CUMSUM_ELEMENT_BYTES)
        adds = float(max(0, experts - 1))
        res_units = getattr(geometry, "resource_units", {}) or {}
        read_role = "mte2_aiv" if "mte2_aiv" in res_units else "mte2"
        die_aiv = max(1, int(res_units.get("vector", 1)))
        owners = getattr(geometry, "stream_owners", None)
        owner = (
            int(owners.aiv[0])
            if (owners is not None and owners.isolation == "disjoint" and owners.aiv)
            else 0
        )
        work = Work(resources={
            "hbm": 2.0 * count_bytes,
            read_role: count_bytes,
            "mte3": count_bytes,
            "vector": adds,
        })

        def program(context):
            placement = context.owned(owner, die_aiv, (read_role, "vector", "mte3"))
            return KernelProgram((
                Stage("count_to_prefix_simd", context.bind(work), placement=placement),
            ))

        return OperatorPlan(
            work, program, metadata={"source_execution_domain": "aicore_simd"},
        )
    work = Work(resources={"hbm": op.hbm_bytes})
    return OperatorPlan(
        work,
        lambda context: KernelProgram((Stage("count_to_prefix_memory", context.bind(work)),)),
        metadata={"source_execution_domain": "aicpu"},
    )


def recipes():
    return {RECIPE: PlannedRecipe(build)}


def lower(ops, bindings):
    from mindie_sim.prediction.implementations import select
    selected = select(bindings, "group_list_prefix")
    if selected is None:
        return tuple(ops)
    fusion = select(bindings, "native_fusion")
    if fusion is None or "grouped_gate_up_activation_quant" not in fusion.options.get("roles", ()):
        raise ModelingError("group_list_prefix_requires_fused_gate_up_quant_consumer")
    by_id = {op.op_id: op for op in ops}
    result = []
    pos_by_id: dict[str, int] = {}
    materialized_producers: set[str] = set()
    for op in ops:
        meta = dict(op.metadata)
        if op.family != "grouped_gemm" or op.recipe not in {
                "grouped_gate_up", "balanced_grouped_gate_up"}:
            pos_by_id[op.op_id] = len(result)
            result.append(op)
            continue
        if meta.get("activation_format") != "int8" or meta.get("weight_format") != "int8":
            raise ModelingError("group_list_prefix_requires_selected_int8_fused_gate_up")
        # Resolve the receiver-local count producer: under EP > 1 the dispatch
        # collective establishes receiver-local expert counts; under EP == 1 the
        # local route_pack/moe_init_routing_v3 producer owns them directly.
        pending, visited, pack_producers, dispatch_producers = list(op.deps), set(), [], []
        while pending:
            key = pending.pop()
            if key in visited or key not in by_id:
                continue
            visited.add(key)
            source = by_id[key]
            if source.layer != op.layer:
                continue
            sm = dict(source.metadata)
            if (
                source.recipe == "expert_dispatch"
                or source.family in {"alltoall", "alltoallv", "mc2_protocol"}
            ):
                dispatch_producers.append(source)
                pending.extend(source.deps)
            elif (source.recipe in {"route_pack", "moe_init_routing_v3"}
                    and sm.get("num_experts")):
                pack_producers.append(source)
            else:
                pending.extend(source.deps)
        if len(pack_producers) != 1:
            raise ModelingError("group_list_prefix_requires_one_local_count_producer")
        if len(dispatch_producers) > 1:
            raise ModelingError("group_list_prefix_requires_one_local_count_producer")
        pack_producer = pack_producers[0]
        producer = dispatch_producers[0] if dispatch_producers else pack_producer
        global_experts = int(dict(pack_producer.metadata)["num_experts"])
        raw_local = meta.get("matrix_weight_bank_groups", global_experts)
        if isinstance(raw_local, bool) or not isinstance(raw_local, int):
            raise ModelingError("group_list_prefix_requires_local_count_list")
        experts = int(raw_local)
        if experts < 1 or experts > global_experts:
            raise ModelingError("group_list_prefix_requires_local_count_list")
        if not dispatch_producers and experts != global_experts:
            raise ModelingError("group_list_prefix_requires_receiver_local_count_producer")
        count_bytes = experts * _CUMSUM_ELEMENT_BYTES
        if (
            "expert_counts" not in producer.returns
            and producer.op_id not in materialized_producers
            and producer.op_id in pos_by_id
        ):
            idx = pos_by_id[producer.op_id]
            current = result[idx]
            result[idx] = replace(
                current,
                mte3_bytes=float(current.mte3_bytes) + float(count_bytes),
                hbm_bytes=float(current.hbm_bytes) + float(count_bytes),
                returns=(*current.returns, "expert_counts"),
            )
            materialized_producers.add(producer.op_id)
        name = producer.op_id + ".group_list_prefix"
        adds = max(0, experts - 1)
        count_and_prefix_bytes = 2 * count_bytes
        prefix = OpSpec(
            op_id=name, family="prefix_sum", phase=op.phase, stage=op.stage,
            layer=op.layer, requests=op.requests, deps=(producer.op_id,),
            semantic_shape=(experts,), local_shape=(experts,),
            padded_shape=(experts,), launch_shape=(1,), dtype=_CUMSUM_DTYPE, layout="flat",
            recipe=RECIPE,
            vector_ops=float(adds),
            hbm_bytes=count_and_prefix_bytes + _CUMSUM_ELEMENT_BYTES,
            mte2_bytes=float(count_bytes), mte3_bytes=float(count_bytes),
            source_boundary="kernel", stream=op.stream,
            metadata=(("source", SOURCE), ("native_kind", "Cumsum"),
                      ("source_execution_domain", "aicore_simd"),
                      ("logical_integer_adds", adds),
                      ("unpriced_service_work", {"aicpu_integer_adds": adds}),
                      ("input_group_list_type", 1), ("output_group_list_type", 0),
                      ("axis_dtype", _CUMSUM_DTYPE), ("axis_value", 0),
                      ("group_count", experts),
                      *tuple((k, v) for k, v in op.metadata if k in {
                          "graph_captured", "graph_segment", "structure_block", "structure_role"})),
        )
        pos_by_id[name] = len(result)
        pos_by_id[op.op_id] = len(result) + 1
        result.extend((prefix, replace(op, deps=(*op.deps, name), metadata=tuple({
            **meta, "group_list_producer": name, "group_list_type": 0,
            "group_list_dtype": _CUMSUM_DTYPE, "group_list_conversion": SOURCE,
        }.items()))))
    return tuple(result)
