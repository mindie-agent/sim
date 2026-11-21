"""xLLM fused grouped matmul with SwiGLU and quantization.

[Role] Bind fused xLLM grouped matmul, SwiGLU and quantization.
[Invariants] Group rows, workspace windows and quantized outputs conserve the declared work.
[Timeouts & Locks] Bounded local computation. No subprocesses or external locks.
[Side-Effect Policy] Creates local records only. No I/O or device access.
"""

from __future__ import annotations

from typing import Any

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec

from mindie_sim.prediction.mechanism import OperatorPlan, Work
from mindie_sim.prediction.program import KernelProgram, Stage
from .contracts import (
    FP8_GROUPED_OWNER_RECIPE,
    FUSED_GROUPED_RECIPE,
    XLLM_CUSTOM_OPS,
    XLLM_VENDOR,
    _shape,
)
from .grouped_epilogue import register_epilogue_work

_MX_SCALE_GROUP_SIZE = 32
_WORKSPACE_CAP_BYTES = 64 * 1024 * 1024
_ACCUMULATOR_BYTES = int(dtype_bytes("int32"))
_ROW_SCALE_BYTES = int(dtype_bytes("fp32"))
_GROUP_COUNT_BYTES = int(dtype_bytes("int64"))
_SWIGLU_OPS_PER_ELEMENT = 5
_QUANT_OPS_PER_ELEMENT = 4


def _matmul_dims(op: OpSpec, meta: dict[str, Any]) -> tuple[int, int, int]:
    shape = _shape(op)
    if len(shape) == 3:
        rows, output_width, input_width = shape
    elif len(shape) == 2:
        rows, output_width = shape
        input_width = meta.get("input_width", meta.get("k"))
        if type(input_width) is not int or input_width < 1:
            raise ModelingError("xllm_grouped_matmul_requires_declared_K")
    else:
        raise ModelingError("xllm_grouped_matmul_requires_MNK_geometry")
    groups = tuple(meta.get("matrix_group_rows") or ())
    if groups and (any(type(value) is not int or value < 0 for value in groups)
                   or sum(groups) != rows):
        raise ModelingError("xllm_grouped_matmul_group_rows_disagree_with_M")
    if rows < 1 or output_width < 2 or input_width < 1 or output_width % 2:
        raise ModelingError("xllm_grouped_matmul_requires_positive_even_gate_up_width")
    return rows, output_width, input_width


def _fused_quant_contract(op: OpSpec, meta: dict[str, Any], dtype: str):
    """Resolve the source INT8 grouped contract or the legal FP8 owner."""
    declared = str(
        meta.get("activation_format") or meta.get("input_format")
        or meta.get("x_format") or dtype).lower()
    output_format = str(meta.get("output_format", "int8")).lower()
    if (dtype in {"fp8_e4m3", "fp8_e5m2"} or declared in {"fp8", "fp8_e4m3", "fp8_e5m2"}
            or output_format in {"fp8", "fp8_e4m3", "fp8_e5m2"}
            or meta.get("activation_scale_group_size") == _MX_SCALE_GROUP_SIZE
            or meta.get("scale_group_size") == _MX_SCALE_GROUP_SIZE):
        raise ModelingError(
            "xllm_fused_grouped_matmul_fp8_requires_owner_recipe:"
            + FP8_GROUPED_OWNER_RECIPE)
    if dtype != "int8" or declared != "int8":
        raise ModelingError(
            f"xllm_fused_grouped_matmul_requires_int8_activation:{dtype}:{declared}")
    weight_format = str(meta.get("weight_format") or meta.get("weight_dtype", "int8")).lower()
    if weight_format != "int8":
        raise ModelingError(
            f"xllm_fused_grouped_matmul_requires_int8_weight:{weight_format}")
    if output_format != "int8":
        raise ModelingError(
            f"xllm_fused_grouped_matmul_requires_int8_output:{output_format}")
    weight_scale_format = str(
        meta.get("weight_scale_format") or meta.get("weight_scale_dtype", "fp32")).lower()
    if weight_scale_format not in {"fp32", "float32", "bf16", "bfloat16", "fp16", "float16"}:
        raise ModelingError(
            "xllm_fused_grouped_matmul_unsupported_weight_scale:"
            + weight_scale_format)
    x_scale_format = str(
        meta.get("x_scale_format") or meta.get("activation_scale_format", "fp32")).lower()
    if x_scale_format not in {"fp32", "float32"}:
        raise ModelingError(
            "xllm_fused_grouped_matmul_requires_fp32_activation_scale:"
            + x_scale_format)
    y_scale_format = str(
        meta.get("output_scale_format") or meta.get("y_scale_format", "fp32")).lower()
    if y_scale_format not in {"fp32", "float32"}:
        raise ModelingError(
            "xllm_fused_grouped_matmul_requires_fp32_output_scale:"
            + y_scale_format)
    weight_scale_bytes = dtype_bytes(weight_scale_format)
    return output_format, weight_scale_format, weight_scale_bytes


def _fused_config(op: OpSpec, meta: dict[str, Any]) -> dict[str, Any]:
    rows, gate_up_width, input_width = _matmul_dims(op, meta)
    groups = tuple(meta.get("matrix_group_rows") or ())
    if not groups:
        raise ModelingError("xllm_fused_grouped_matmul_requires_matrix_group_rows")
    if any(type(value) is not int or value < 0 for value in groups) or sum(groups) != rows:
        raise ModelingError("xllm_fused_grouped_matmul_group_rows_disagree_with_M")
    output_width = gate_up_width // 2
    output_format, weight_scale_format, weight_scale_bytes = _fused_quant_contract(
        op, meta, str(op.dtype))
    active_groups = sum(1 for value in groups if value > 0)
    activation_bytes = rows * input_width
    weight_bytes = active_groups * gate_up_width * input_width
    weight_scale_bytes_total = active_groups * gate_up_width * weight_scale_bytes
    x_scale_bytes = rows * _ROW_SCALE_BYTES
    group_list_bytes = len(groups) * _GROUP_COUNT_BYTES
    output_bytes = rows * output_width
    output_scale_bytes = rows * _ROW_SCALE_BYTES
    workspace_bytes = rows * gate_up_width * _ACCUMULATOR_BYTES
    limit_rows = meta.get("workspace_limit_rows")
    if limit_rows is None:
        limit_rows = (_WORKSPACE_CAP_BYTES // 2 // _ACCUMULATOR_BYTES) // gate_up_width
    if type(limit_rows) is not int or limit_rows < 1:
        raise ModelingError("xllm_fused_grouped_invalid_workspace_limit_rows")
    split_workspace = rows > 2 * limit_rows
    workspace_live_bytes = min(rows, 2 * limit_rows) * gate_up_width * _ACCUMULATOR_BYTES
    read_bytes = (activation_bytes + weight_bytes + weight_scale_bytes_total
                  + x_scale_bytes + group_list_bytes)
    write_bytes = output_bytes + output_scale_bytes
    return {
        "rows": rows, "gate_up_width": gate_up_width,
        "output_width": output_width, "k": input_width,
        "group_rows": groups, "group_count": len(groups),
        "active_groups": active_groups,
        "matrix_flops": 2 * rows * gate_up_width * input_width,
        "activation_bytes": activation_bytes,
        "weight_bytes": weight_bytes,
        "weight_scale_bytes": weight_scale_bytes_total,
        "weight_scale_format": weight_scale_format,
        "x_scale_bytes": x_scale_bytes,
        "group_list_bytes": group_list_bytes,
        "output_bytes": output_bytes,
        "output_scale_bytes": output_scale_bytes,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "workspace_bytes": workspace_bytes,
        "workspace_live_bytes": workspace_live_bytes,
        "workspace_limit_rows": limit_rows,
        "workspace_pool": str(meta.get("workspace_pool", "gm")),
        "workspace_grant": meta.get("workspace_grant"),
        "split_workspace": split_workspace,
        "output_format": output_format,
        "quantization": "int8_per_channel_float_scale",
    }


def _fused_epilogue_work(config: dict[str, Any], geometry) -> tuple[Work, dict[str, Any]]:
    """Price epilogue instructions apart from the enclosing call's transfers."""
    aivs = int(geometry.resource_units.get("vector", geometry.compute_units))
    ub = int(geometry.buffers.get("ub", 0))
    if "reg_alu_lanes" in geometry.resource_units and ub > 0 and aivs > 0:
        work, blocks, tile_rows = register_epilogue_work(
            config["group_rows"], config["output_width"], config["rows"], aivs, ub)
        return work, {"basis": "arch35_dequant_swiglu_quant_capacity",
                      "aiv": blocks, "tile_rows": tile_rows}
    # Geometry without register-lane service executes the epilogue's work on
    # the common vector pipe.
    elements = config["rows"] * config["output_width"]
    return Work(resources={"vector": float(2 * elements + _SWIGLU_OPS_PER_ELEMENT * elements
                                           + _QUANT_OPS_PER_ELEMENT * elements
                                           + 2 * config["rows"])}), {
        "basis": "common_vector_equivalent_without_a5_register_lanes",
        "elements": elements}


def _build_fused_grouped(op: OpSpec, geometry):
    config = dict(dict(op.metadata).get("xllm_fused_config") or {})
    if not config:
        raise ModelingError("xllm_fused_grouped_requires_prepared_config")
    rows = int(config["rows"])
    limit = int(config["workspace_limit_rows"])
    split = bool(config["split_workspace"])
    window_rows = ([min(limit, rows - start) for start in range(0, rows, limit)]
                   if split else [rows])
    read_work = Work(resources={
        "hbm": config["read_bytes"], "mte2": config["read_bytes"]})
    write_work = Work(resources={
        "hbm": config["write_bytes"], "mte3": config["write_bytes"]})
    epilogue, basis = _fused_epilogue_work(config, geometry)
    total = Work()
    windows = []
    for index, count in enumerate(window_rows):
        share = float(count) / float(rows)
        read = read_work.scaled(share)
        workspace_bytes = float(count * config["gate_up_width"] * _ACCUMULATOR_BYTES)
        matrix = Work(matrix={"int8": 2.0 * count * config["gate_up_width"]
                              * config["k"]},
                      resources={"hbm": workspace_bytes, "mte3": workspace_bytes})
        space = Work(resources={"hbm": workspace_bytes, "mte2": workspace_bytes})
        epi = epilogue.scaled(share)
        write = write_work.scaled(share)
        total += read + matrix + space + epi + write
        windows.append((index, read, matrix, space, epi, write))

    def program(context):
        pool_name = str(config.get("workspace_pool", "gm"))
        if pool_name not in context.pool_names:
            pool_name = next((name for name in context.pool_names
                              if "workspace" in name or name in {"gm", "hbm"}),
                             pool_name)
        pool = context.pool_names.get(pool_name)
        allocations = {}
        if config.get("workspace_grant") is not None:
            pool = None
        if pool is not None and split:
            size = float(min(limit, rows) * config["gate_up_width"] * _ACCUMULATOR_BYTES)
            allocations = {
                slot: (context.allocate(f"xllm_ws_{slot}", pool_name, size, (0,)),)
                for slot in (0, 1)}
        elif pool is not None:
            size = float(rows * config["gate_up_width"] * _ACCUMULATOR_BYTES)
            allocations = {0: (context.allocate("xllm_ws_0", pool_name, size, (0,)),)}
        stages = []
        for index, read, matrix, space, epi, write in windows:
            slot = index % 2
            alloc = allocations.get(slot, ())
            stages.extend((
                Stage(f"read_{index}", context.bind(read)),
                Stage(f"matrix_{index}", context.bind(matrix),
                      after=((f"read_{index}",)
                             + ((f"workspace_read_{index - 2}",)
                                if index >= 2 else ())),
                      acquire=alloc),
                Stage(f"workspace_read_{index}", context.bind(space),
                      after=(f"matrix_{index}",), free=alloc),
                Stage(f"dequant_swiglu_quant_{index}", context.bind(epi),
                      after=(f"workspace_read_{index}",)),
                Stage(f"write_{index}", context.bind(write),
                      after=(f"dequant_swiglu_quant_{index}",)),
            ))
        return KernelProgram(tuple(stages))

    return OperatorPlan(total, program, metadata={
        "xllm_grouped_plan": config,
        "epilogue_work_basis": basis,
        "intermediate_hbm_bytes": int(config["workspace_bytes"]),
        "workspace_live_bytes": int(config["workspace_live_bytes"]),
        "workspace_window_rows": tuple(window_rows),
        "workspace_windows": len(window_rows),
        "workspace_requires_memory_grant": True,
        "workspace_grant_reused": config.get("workspace_grant") is not None,
        "workspace_allocation_pool": str(config.get("workspace_pool", "gm")),
        "split_workspace": split,
        "source": XLLM_VENDOR + ":moe_grouped_matmul_swiglu_quant",
    })


def fused_region_contract(members, outputs, selection, options=None):
    """Consume a declared three-member gate/up+SwiGLU+quant region.

    T11 owns the region replacement.  This function supplies the one typed
    contract that both the region path and direct complete-call lowering use;
    it does not rewrite a graph or claim ownership of member occurrences.
    """
    members = tuple(members)
    options = dict(options or {})
    if set(options) - {"op_name", "soc_family"}:
        raise ModelingError("xllm_fused_region_has_unexpected_options")
    matrices = [op for op in members
                if op.family in {"grouped_gemm"} or op.recipe in {
                    "grouped_gate_up", "balanced_grouped_gate_up"}]
    activations = [op for op in members if op.family == "swiglu"]
    quantizers = [op for op in members if op.family == "quantize"]
    if (len(members) != 3 or len(matrices) != 1 or len(activations) != 1
            or len(quantizers) != 1):
        raise ModelingError("xllm_fused_region_requires_gate_up_swiglu_quant")
    matrix, activation, quant = matrices[0], activations[0], quantizers[0]
    if (activation.deps != (matrix.op_id,) or quant.deps != (activation.op_id,)
            or set(outputs) != {quant.op_id}
            or any(op.plan_required or "implementation_binding" in dict(op.metadata)
                   for op in members)):
        raise ModelingError(
            "xllm_fused_region_requires_ordinary_exclusive_member_chain")
    meta = dict(matrix.metadata)
    config = _fused_config(matrix, meta)
    return {
        "config": config,
        "members": tuple(op.op_id for op in members),
        "external_outputs": tuple(outputs),
        "implementation": selection.name,
        "native_kind": XLLM_CUSTOM_OPS[FUSED_GROUPED_RECIPE]["native_kind"],
        "source": {
            "repository": "xllm",
            "op": FUSED_GROUPED_RECIPE,
            "path": XLLM_CUSTOM_OPS[FUSED_GROUPED_RECIPE]["source_path"],
            "scope": "declared_three_member_region; replacement_owned_by_T11",
        },
    }
