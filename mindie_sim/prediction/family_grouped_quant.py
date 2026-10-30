"""Explicit INT8 grouped accumulator -> dequant/SwiGLU/INT8 caller.

[Role] Retain the nonfused grouped caller's INT32 intermediate and scale edges.
[Invariants] Grouped GEMM and activation are separate invocations. Integer
  expert rows select weight scales; input row scales retain their producer.
  No BF16 intermediate or matrix epilogue survives this selected INT32 branch.
  The default consumer is analytical; source arch35 tiling is an explicit
  option with allocated capacity distinct from valid integer expert rows.
[Timeouts & Locks] O(experts + physical AIVs) metadata/program size; no I/O.
[Side-Effect Policy] Immutable graph transforms and common Engine plans only.
"""
from dataclasses import replace
from math import ceil

from mindie_sim.schema.common import ModelingError
from .mechanism import OperatorPlan, PlannedRecipe, Work
from .program import KernelProgram, ParallelProgram, Stage
from .transforms import _fusion_metadata

SELECTOR = "grouped_int8_caller"
IMPLEMENTATION = "ascend_grouped_int32_dequant_swiglu"
RECIPE = "grouped_int32_dequant_swiglu"
SOURCE = {
    "repository": "vllm-ascend", "revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
    "caller": "vllm_ascend/ops/fused_moe/moe_mlp.py:326-361",
    "caller_sha256": "6b5f4f6d3259cb14e0613a6f7b1c2592331f143a14cf7240dd9d49a0a57dd53b",
    "kernel": "csrc/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant.h:210-277,374-495,555-589",
    "kernel_sha256": "2e516dc7332449057ea8379eebd61a0b6f2a2917907b7f88d07b6025d3cf72b2",
    "scope": "INT8_no_bias_nonfused_SwiGLU_caller; captured_binary_not_identified",
}


def recipes():
    return {RECIPE: PlannedRecipe(build)}


def tensor_contract(op):
    meta = op["metadata"]
    config = meta.get("grouped_dequant")
    if op["recipe"] != RECIPE or not config:
        return None
    rows, width = op["local_shape"]
    groups = config["expert_rows"]
    if sum(groups) != rows or config["width"] != width:
        return None
    # Source host tiling sees allocated tensor extents; group_index alone
    # states which rows carry valid data. Work and consumer scales stay valid.
    allocated = config.get("input_capacity_rows", rows)
    empty = ((), None, None)
    return ("native_tensor_abi", "DequantSwigluQuant", (
        ((allocated, width * 2), "int32", "ND"),
        ((len(groups), width * 2), "fp32", "ND"), ((allocated,), "fp32", "ND"),
        empty, empty, empty, ((len(groups),), "int64", "ND")), (
        ((allocated, width), "int8", "ND"), ((allocated,), "fp32", "ND")))


def _one(items, reason):
    if len(items) != 1:
        raise ModelingError("grouped_int32_" + reason)
    return items[0]


def _row_scale(gate, by_id, rows):
    explicit = dict(gate.metadata).get("activation_scale_producer")
    if explicit is None:
        return _one([by_id[d] for d in gate.deps if d in by_id
                     and "activation_scales" in by_id[d].returns
                     and dict(by_id[d].metadata).get("activation_scale_format") == "fp32"
                     and tuple(dict(by_id[d].metadata).get("activation_scale_shape", ())) == (rows,)],
                    "requires_one_input_row_scale_producer")
    if explicit not in gate.deps or explicit not in by_id:
        raise ModelingError("grouped_int32_input_row_scale_producer_not_ready")
    scale = by_id[explicit]
    meta = dict(scale.metadata)
    # A collective's local_shape describes sent rows. Its completion supplies
    # the receiver rows already declared by the gate, including an idle sender.
    received = (scale.recipe == "expert_scale_dispatch" and meta.get("payload_kind") == "row_scale"
                and meta.get("payload_dtype") == "fp32" and meta.get("row_bytes") == 4)
    local = ("activation_scales" in scale.returns and meta.get("activation_scale_format") == "fp32"
             and tuple(meta.get("activation_scale_shape", ())) == (rows,))
    if not (received or local):
        raise ModelingError("grouped_int32_input_row_scale_contract_mismatch")
    return scale


def _rewrite(ops, gate, options):
    by_id = {op.op_id: op for op in ops}
    users = {op.op_id: set() for op in ops}
    for op in ops:
        for dep in op.deps:
            if dep in users:
                users[dep].add(op.op_id)
    meta = dict(gate.metadata)
    rows, columns, hidden = gate.local_shape
    groups = tuple(meta.get("matrix_group_rows", ()))
    if (not groups or any(type(n) is not int or n < 0 for n in groups)
            or sum(groups) != rows or columns % 2):
        raise ModelingError("grouped_int32_requires_integral_expert_rows_and_even_columns")
    if (meta.get("activation_format") != "int8" or meta.get("weight_format") != "int8"
            or meta.get("activation_scale_format") != "fp32"
            or tuple(meta.get("activation_scale_shape", ())) != (rows,)):
        raise ModelingError("grouped_int32_requires_int8_and_fp32_row_scales")
    if gate.plan_required:
        raise ModelingError("grouped_int32_cannot_rewrite_selected_source_matrix")
    width = columns // 2
    scale = _row_scale(gate, by_id, rows)
    activation = _one([by_id[key] for key in users[gate.op_id]
                       if by_id[key].family == "swiglu"], "requires_one_swiglu_consumer")
    quant = _one([by_id[key] for key in users[activation.op_id]
                  if by_id[key].family == "quantize"], "requires_one_output_quantizer")
    down = _one([by_id[key] for key in users[quant.op_id]
                 if by_id[key].family == "grouped_gemm"], "requires_one_down_matrix")
    if (users[gate.op_id] != {activation.op_id} or users[activation.op_id] != {quant.op_id}
            or users[quant.op_id] != {down.op_id} or activation.deps != (gate.op_id,)
            or quant.deps != (activation.op_id,) or down.deps != (quant.op_id,)):
        raise ModelingError("grouped_int32_requires_exclusive_activation_chain")
    qm, dm = dict(quant.metadata), dict(down.metadata)
    if (tuple(activation.local_shape) != (rows, width)
            or qm.get("output_format") != "int8" or qm.get("activation_scale_format") != "fp32"
            or dm.get("activation_format") != "int8"
            or tuple(dm.get("matrix_group_rows", ())) != groups
            or tuple(dm.get("activation_scale_shape", ())) != (rows,)):
        raise ModelingError("grouped_int32_activation_and_down_contract_mismatch")
    # The selected GMM call supplies no scale argument and writes INT32.
    # Scales remain external weight storage and move to the activation consumer.
    active = sum(n > 0 for n in groups)
    weight = active * columns * hidden
    read, write = rows * hidden + weight, rows * columns * 4
    gm = {**meta, "output_format": "int32", "packed_weight_bytes": weight,
          "weight_bytes": weight, "weight_payload_bytes": weight,
          "weight_representation": "int8_payload_without_matrix_scales",
          "activation_scale_in_mte2": False, "activation_scale_read_by_producer": False,
          "matrix_output_contract": "int32_quant_matmul_without_epilogue",
          "column_metadata_policy": "no_scale_argument_in_selected_int32_gmm",
          "grouped_int32_caller": SOURCE}
    for key in ("matrix_epilogue", "activation_scale_bytes", "activation_scale_shape", "activation_scale_format"):
        gm.pop(key, None)
    new_gate = replace(gate, hbm_bytes=read + write, mte2_bytes=read, mte3_bytes=write,
                       vector_ops=0, metadata=tuple(gm.items()))
    # Preserve both semantic memberships so fused graph reporting cannot lose
    # the quantizer when its resource producer becomes the activation kernel.
    am = dict(activation.metadata)
    for key in ("structure_subgraph_memberships", "structure_subgraph_entries"):
        am[key] = tuple(dict.fromkeys((*am.get(key, ()), *qm.get(key, ()))))
    am["structure_subgraphs"] = tuple({
        **group, "tensor_view": {**group["tensor_view"], "intermediate_dtype": "int32"},
        "implementation": {"native_kinds": [], "selected_caller": IMPLEMENTATION,
                           "kernel_sequence": ("GroupedMatmul", "DequantSwigluQuant"),
                           "conditions": {"separate_invocations": True,
                                          "down_projection_external": True}},
        "source": SOURCE,
    } if gate.op_id in group["op_ids"] else group
        for group in am.get("structure_subgraphs", ()))
    am.update({"input_format": "int32", "output_format": "int8",
               "activation_scale_format": "fp32", "activation_scale_shape": (rows,),
               "activation_scale_bytes": rows * 4, "grouped_dequant": {
                   "expert_rows": groups, "width": width, "swiglu_mode": 0,
                   "bias": False, "smooth_scale": False, **options},
               "native_kind": "DequantSwigluQuant", "source": SOURCE,
               "invocation_class": "vector", "input_row_scale_producer": scale.op_id,
               "group_index_dtype": "int64", "group_index_values": groups,
               "group_index_semantics": "rows_per_expert",
               "semantic_recipe": activation.recipe,
               "fused_origin_scopes": _fusion_metadata(activation, quant)["fused_origin_scopes"]})
    consumer = Work().apply(replace(
        activation, deps=(gate.op_id, scale.op_id), recipe=RECIPE,
        family="dequant_swiglu_quant", dtype="int8", source_boundary="kernel",
        returns=("activation_scales",), metadata=tuple(am.items())))
    new_down = replace(down, deps=(consumer.op_id,), metadata=tuple({
        **dm, "activation_payload_producer": consumer.op_id,
        "activation_scale_producer": consumer.op_id}.items()))
    replacements = {gate.op_id: new_gate, activation.op_id: consumer, down.op_id: new_down}
    return tuple(replacements.get(op.op_id, op) for op in ops if op.op_id != quant.op_id)


def lower(ops, bindings):
    from .implementations import select
    selected = select(bindings, SELECTOR)
    if selected is None:
        return tuple(ops)
    options = selected.options
    if options and (options.get("dequant_program") != "arch35_full_load"
                    or type(options.get("input_capacity_rows")) is not int):
        raise ModelingError("grouped_int32_source_dequant_requires_arch35_full_load_and_capacity")
    current = tuple(ops)
    for gate in ops:
        # Recipes carry semantic family roles; names and model labels do not.
        if (gate.family == "grouped_gemm" and gate.recipe in {"grouped_gate_up", "balanced_grouped_gate_up"}
                and "grouped_int32_caller" not in dict(gate.metadata)):
            current = _rewrite(current, gate, options)
    return current


def vector_elements(rows, width):
    # INT32 cast + two scale multiplies, normal SiLU+gate (five operations),
    # abs/max/scale/divide/three casts. Logical elements, not cycle counts.
    return rows * (3 * 2 * width + 5 * width + width + width - 1 + 1 + 8 + width + 3 * width)


def build(op, geometry):
    config = dict(op.metadata)["grouped_dequant"]
    if config.get("dequant_program") == "arch35_full_load":
        from .grouped_dequant_arch35 import build as source_build
        return source_build(op, geometry)
    groups, width = tuple(config["expert_rows"]), config["width"]
    if sum(groups) != op.local_shape[0] or width != op.local_shape[1]:
        raise ModelingError("grouped_dequant_rows_disagree_with_output")
    units = geometry.resource_units.get("vector", geometry.compute_units)
    if not {"mte2_aiv", "mte3", "vector"} <= geometry.resource_units.keys():
        raise ModelingError("grouped_dequant_requires_aiv_read_and_write_pipes")
    # Source group traversal repartitions each positive group over available
    # owners. Phase aggregation preserves per-owner work, not captured tiling,
    # scalar metadata latency or within-owner overlap.
    loads, computes, writes = [0] * units, [0] * units, [0] * units
    for owner in range(units):
        loads[owner] = len(groups) * 8
        for n in groups:
            if not n:
                continue
            chunk = ceil(n / units)
            count = max(0, min(chunk, n - owner * chunk))
            if count:
                loads[owner] += count * (8 * width + 4) + 8 * width
                computes[owner] += vector_elements(count, width)
                writes[owner] += count * (width + 4)
    work = Work(resources={"hbm": sum(loads) + sum(writes), "mte2_aiv": sum(loads),
                           "mte3": sum(writes), "vector": sum(computes)})
    def program(context):
        def owner_program(owner):
            placement = context.owned(owner, units, ("mte2_aiv", "mte3", "vector"))
            return KernelProgram((
                Stage("load", context.bind(Work(resources={"hbm": loads[owner], "mte2_aiv": loads[owner]})),
                      placement=placement),
                Stage("dequant_swiglu_quant", context.bind(Work(resources={"vector": computes[owner]})),
                      after=("load",), placement=placement),
                Stage("write", context.bind(Work(resources={"hbm": writes[owner], "mte3": writes[owner]})),
                      after=("dequant_swiglu_quant",), placement=placement)))
        return ParallelProgram(tuple(owner_program(i) for i in range(units)))
    return OperatorPlan(work, program, metadata={"grouped_dequant_schedule": {
        "kind": "analytical_per_owner_group_phases", "physical_aiv": units,
        "group_count": len(groups), "active_experts": sum(n > 0 for n in groups),
        "unmodeled": ("native_host_tiling", "scalar_group_descriptor_issue_latency",
                       "finite_ub_tiling_and_pipeline_overlap", "compiler_instruction_expansion")}})
