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
from typing import Literal

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, Stage
from mindie_sim.prediction.transforms import _fusion_metadata
from .lowering import GROUP_COUNT_BYTES

TARGET_BACKEND: Literal["ascend"] = "ascend"
SELECTOR = "grouped_int8_caller"
IMPLEMENTATION = "ascend_grouped_int32_dequant_swiglu"
RECIPE = "grouped_int32_dequant_swiglu"
_INT32_BYTES = int(dtype_bytes("int32"))
_FP32_BYTES = int(dtype_bytes("fp32"))
_DEQUANT_OPS_PER_INPUT = 3  # Cast and two scale multiplies.
_SWIGLU_OPS_PER_OUTPUT = 5
_ROW_SCALE_DIVISION_OPS = 8
_QUANT_CASTS_PER_OUTPUT = 3
_HOST_BLOCK_BYTES = 32
_HOST_BLOCK_FP32_ELEMS = _HOST_BLOCK_BYTES // _FP32_BYTES
_HOST_SWIGLU_FACTOR = 2
_HOST_INPUT_WIDTH_ALIGN = _HOST_BLOCK_BYTES * _HOST_SWIGLU_FACTOR
_HOST_UB_RESERVE_BYTES = 1024
_HOST_DOUBLE_BUFFER = 2
_HOST_STATIC_FP32_PLANES = 5
_HOST_DYNAMIC_FIXED_RESERVE_BYTES = (
    _HOST_UB_RESERVE_BYTES
    + _HOST_BLOCK_BYTES
    + _HOST_DOUBLE_BUFFER * _HOST_BLOCK_BYTES
    + _FP32_BYTES
    + _HOST_BLOCK_FP32_ELEMS * _HOST_BLOCK_FP32_ELEMS * _FP32_BYTES
)
_HOST_DYNAMIC_BASE_BYTES_PER_COL = (
    _HOST_STATIC_FP32_PLANES * _FP32_BYTES
    + _HOST_DOUBLE_BUFFER * _HOST_SWIGLU_FACTOR * _FP32_BYTES
    + 1
)
_HOST_PERFORMANCE_CORE_CAP = 36
_HOST_PERFORMANCE_H_SMALL = 2048
_HOST_PERFORMANCE_H_LARGE = 4096
_HOST_PERFORMANCE_UB_FACTOR_NUM = _HOST_PERFORMANCE_H_LARGE * _FP32_BYTES
_HOST_CUT_GROUP_MIN = 32
_HOST_CUT_GROUP_MAX_AVG_ROWS = 16
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-ascend",
        revision="175ef4d49c2072f9233619636f0309f4f88b4e2c",
        relative_path="vllm_ascend/ops/fused_moe/moe_mlp.py",
        source_sha256="6b5f4f6d3259cb14e0613a6f7b1c2592331f143a14cf7240dd9d49a0a57dd53b",
    ),
    SourceRef(
        repository="vllm-ascend",
        revision="175ef4d49c2072f9233619636f0309f4f88b4e2c",
        relative_path="csrc/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant.h",
        source_sha256="2e516dc7332449057ea8379eebd61a0b6f2a2917907b7f88d07b6025d3cf72b2",
    ),
)
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
                and meta.get("payload_dtype") == "fp32" and meta.get("row_bytes") == _FP32_BYTES)
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
    if options.get("input_capacity_rows") == "valid_rows":
        # A variable-size caller explicitly allocates its received expert rows.
        # Fixed captured allocations remain integers and keep their own tiling.
        options = {**options, "input_capacity_rows": max(1, rows),
                   "input_capacity_policy": "valid_rows"}
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
    read, write = rows * hidden + weight, rows * columns * _INT32_BYTES
    gm = {**meta, "output_format": "int32", "packed_weight_bytes": weight,
          "weight_bytes": weight, "weight_payload_bytes": weight,
          "weight_representation": "int8_payload_without_matrix_scales",
          "activation_scale_in_mte2": False, "activation_scale_read_by_producer": False,
          "matrix_output_contract": "int32_quant_matmul_without_epilogue",
          "column_metadata_policy": "no_scale_argument_in_selected_int32_gmm",
          "grouped_int32_caller": SOURCE}
    for key in ("matrix_epilogue", "activation_scale_bytes", "activation_scale_shape", "activation_scale_format"):
        gm.pop(key, None)
    new_gate_abi = None
    if gate.precision_abi is not None:
        new_gate_abi = replace(
            gate.precision_abi,
            output_dtype="int32",
            accum_dtype="int32",
            weight_scale_dtype="",
            weight_scale_bytes=0,
            act_scale_dtype="",
            act_scale_pack="",
            act_group_size=0,
        )
        gm["precision_abi_signature"] = new_gate_abi.abi_signature
    new_gate = replace(gate, hbm_bytes=read + write, mte2_bytes=read, mte3_bytes=write,
                       vector_ops=0, precision_abi=new_gate_abi, metadata=tuple(gm.items()))
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
               "activation_scale_bytes": rows * _FP32_BYTES, "grouped_dequant": {
                   "expert_rows": groups, "width": width, "swiglu_mode": 0,
                   "bias": False, "smooth_scale": False, **options},
               "native_kind": "DequantSwigluQuant", "source": SOURCE,
               "invocation_class": "vector", "input_row_scale_producer": scale.op_id,
               "group_index_dtype": "int64", "group_index_values": groups,
               "group_index_semantics": "rows_per_expert",
               "semantic_recipe": activation.recipe,
               "fused_origin_scopes": _fusion_metadata(activation, quant)["fused_origin_scopes"]})
    new_act_abi = None
    if activation.precision_abi is not None:
        new_act_abi = replace(
            activation.precision_abi,
            input_dtype="int32",
            output_dtype="int8",
            act_scale_dtype="fp32",
            activation_quant_mode="dynamic_per_token",
        )
        am["precision_abi_signature"] = new_act_abi.abi_signature
    consumer = Work().apply(replace(
        activation, deps=(gate.op_id, scale.op_id), recipe=RECIPE,
        family="dequant_swiglu_quant", dtype="int8", source_boundary="kernel",
        precision_abi=new_act_abi, returns=("activation_scales",), metadata=tuple(am.items())))
    new_down = replace(down, deps=(consumer.op_id,), metadata=tuple({
        **dm, "activation_payload_producer": consumer.op_id,
        "activation_scale_producer": consumer.op_id}.items()))
    replacements = {gate.op_id: new_gate, activation.op_id: consumer, down.op_id: new_down}
    return tuple(replacements.get(op.op_id, op) for op in ops if op.op_id != quant.op_id)


def lower(ops, bindings):
    from mindie_sim.prediction.implementations import select
    selected = select(bindings, SELECTOR)
    if selected is None:
        return tuple(ops)
    options = selected.options
    if options and (options.get("dequant_program") != "arch35_full_load"
                    or not (type(options.get("input_capacity_rows")) is int
                            or options.get("input_capacity_rows") == "valid_rows")):
        raise ModelingError("grouped_int32_source_dequant_requires_arch35_full_load_and_capacity")
    current = tuple(ops)
    for gate in ops:
        # Recipes carry semantic family roles; names and model labels do not.
        if (gate.family == "grouped_gemm" and gate.recipe in {"grouped_gate_up", "balanced_grouped_gate_up"}
                and "grouped_int32_caller" not in dict(gate.metadata)):
            current = _rewrite(current, gate, options)
    return current


def vector_elements(rows, width):
    dequant_and_gate = (2 * _DEQUANT_OPS_PER_INPUT + _SWIGLU_OPS_PER_OUTPUT) * width
    abs_and_max = width + (width - 1)
    scale_and_quantize = 1 + _ROW_SCALE_DIVISION_OPS + width + _QUANT_CASTS_PER_OUTPUT * width
    return rows * (dequant_and_gate + abs_and_max + scale_and_quantize)


def _host_used_aiv_cores(groups, width: int, config: dict, geometry, available_units: int) -> tuple[int, bool]:
    """Enforce vllm-ascend DequantSwigluQuant host tiling alignment, UB fit, and block count."""
    in_dim_y = _HOST_SWIGLU_FACTOR * width
    if in_dim_y % _HOST_INPUT_WIDTH_ALIGN:
        raise ModelingError(f"grouped_dequant_input_width_must_be_multiple_of_64:{in_dim_y}")
    ub_size = int(geometry.buffers.get("ub", 0) or 0)
    total_rows = int(config.get("input_capacity_rows", sum(groups)))
    if ub_size <= 0:
        return min(available_units, max(1, total_rows)), False
    has_bias = bool(config.get("bias", False))
    swiglu_mode = int(config.get("swiglu_mode", 0) or 0)
    is_dynamic = int(config.get("quant_mode", 1)) == 1
    bias_col = _HOST_SWIGLU_FACTOR * _FP32_BYTES if has_bias else 0
    swiglu_col = (1 + _INT32_BYTES) if swiglu_mode else 0
    quant_offset_col = 0 if is_dynamic else _FP32_BYTES
    denom_max = _HOST_DYNAMIC_BASE_BYTES_PER_COL + bias_col + swiglu_col + quant_offset_col
    max_out = max(0, (ub_size - _HOST_DYNAMIC_FIXED_RESERVE_BYTES) // denom_max) // _HOST_BLOCK_BYTES * _HOST_BLOCK_BYTES
    max_in = _HOST_SWIGLU_FACTOR * max_out
    if in_dim_y > max_in:
        raise ModelingError(f"grouped_dequant_width_exceeds_ub_limit:{in_dim_y}:{max_in}")
    if total_rows <= 0:
        return min(available_units, 1), False
    bias_x = in_dim_y * _FP32_BYTES if has_bias else 0
    swiglu_x = width * (1 + _INT32_BYTES) if swiglu_mode else 0
    quant_off_x = 0 if is_dynamic else width * _FP32_BYTES
    num_x = (
        ub_size - _HOST_UB_RESERVE_BYTES - width * _FP32_BYTES
        - _HOST_BLOCK_BYTES - in_dim_y * _FP32_BYTES - bias_x - quant_off_x
    )
    den_x = (
        _HOST_DOUBLE_BUFFER * (in_dim_y + _HOST_BLOCK_FP32_ELEMS) * _FP32_BYTES
        + width + _FP32_BYTES + in_dim_y * _FP32_BYTES + swiglu_x
        + _HOST_BLOCK_FP32_ELEMS * _FP32_BYTES
    )
    ub_factor_x = min(max(1, num_x // den_x), total_rows)
    if swiglu_mode == 0 and not has_bias and in_dim_y in {_HOST_PERFORMANCE_H_SMALL, _HOST_PERFORMANCE_H_LARGE}:
        ub_factor_x = _HOST_PERFORMANCE_UB_FACTOR_NUM // in_dim_y
    used_cores = min(available_units, _HOST_PERFORMANCE_CORE_CAP, ceil(total_rows / max(1, ub_factor_x)))
    cut_group = (
        len(groups) >= _HOST_CUT_GROUP_MIN
        and total_rows // len(groups) <= _HOST_CUT_GROUP_MAX_AVG_ROWS
        and not has_bias
        and not config.get("smooth_scale", False)
    )
    if cut_group:
        used_cores = min(available_units, total_rows)
    return max(1, used_cores), cut_group


def build(op, geometry):
    config = dict(op.metadata)["grouped_dequant"]
    if config.get("dequant_program") == "arch35_full_load":
        from .grouped_dequant import build as source_build
        return source_build(op, geometry)
    groups, width = tuple(config["expert_rows"]), config["width"]
    if sum(groups) != op.local_shape[0] or width != op.local_shape[1]:
        raise ModelingError("grouped_dequant_rows_disagree_with_output")
    available_units = geometry.resource_units.get("vector", geometry.compute_units)
    if not {"mte2_aiv", "mte3", "vector"} <= geometry.resource_units.keys():
        raise ModelingError("grouped_dequant_requires_aiv_read_and_write_pipes")
    used_units, cut_group = _host_used_aiv_cores(groups, width, config, geometry, available_units)
    # Source group traversal repartitions each positive group across the host
    # tiler's launched AIV block count (maxPreCore_), or assigns whole groups
    # cyclically by group index in cut-group mode.
    loads, computes, writes = [0] * used_units, [0] * used_units, [0] * used_units
    for owner in range(used_units):
        loads[owner] = len(groups) * GROUP_COUNT_BYTES
    if cut_group:
        for group_idx, n in enumerate(groups):
            if not n:
                continue
            owner = group_idx % used_units
            loads[owner] += (n * (_HOST_SWIGLU_FACTOR * width * _INT32_BYTES + _FP32_BYTES)
                             + _HOST_SWIGLU_FACTOR * width * _FP32_BYTES)
            computes[owner] += vector_elements(n, width)
            writes[owner] += n * (width + _FP32_BYTES)
    else:
        for owner in range(used_units):
            for n in groups:
                if not n:
                    continue
                chunk = ceil(n / used_units)
                count = max(0, min(chunk, n - owner * chunk))
                if count:
                    loads[owner] += (count * (_HOST_SWIGLU_FACTOR * width * _INT32_BYTES + _FP32_BYTES)
                                     + _HOST_SWIGLU_FACTOR * width * _FP32_BYTES)
                    computes[owner] += vector_elements(count, width)
                    writes[owner] += count * (width + _FP32_BYTES)
    work = Work(resources={"hbm": sum(loads) + sum(writes), "mte2_aiv": sum(loads),
                           "mte3": sum(writes), "vector": sum(computes)})
    def program(context):
        def owner_program(owner):
            placement = context.owned(owner, available_units, ("mte2_aiv", "mte3", "vector"))
            return KernelProgram((
                Stage("load", context.bind(Work(resources={"hbm": loads[owner], "mte2_aiv": loads[owner]})),
                      placement=placement),
                Stage("dequant_swiglu_quant", context.bind(Work(resources={"vector": computes[owner]})),
                      after=("load",), placement=placement),
                Stage("write", context.bind(Work(resources={"hbm": writes[owner], "mte3": writes[owner]})),
                      after=("dequant_swiglu_quant",), placement=placement)))
        return ParallelProgram(tuple(owner_program(i) for i in range(used_units)))
    return OperatorPlan(work, program, metadata={"grouped_dequant_schedule": {
        "kind": "analytical_per_owner_group_phases", "physical_aiv": used_units,
        "available_aiv": available_units,
        "group_count": len(groups), "active_experts": sum(n > 0 for n in groups),
        "unmodeled": ("scalar_group_descriptor_issue_latency",
                       "finite_ub_tiling_and_pipeline_overlap", "compiler_instruction_expansion")}})
