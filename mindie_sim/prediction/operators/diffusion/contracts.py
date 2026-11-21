"""Diffusion value contracts and physical stages.

[Role] Assemble diffusion tensor contracts and physical plans.
[Invariants] Each stage retains its value dependencies, transfer work and workspace lifetime.
[Timeouts & Locks] Bounded local computation. No subprocesses or external locks.
[Side-Effect Policy] Creates local records only. No I/O or device access.
"""

from __future__ import annotations

from math import prod
from typing import Any

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec

from mindie_sim.prediction.mechanism import OperatorPlan, Work
from mindie_sim.prediction.program import KernelProgram, Stage


def _shape(op: OpSpec) -> tuple[int, ...]:
    shape = tuple(int(value) for value in op.local_shape)
    if not shape or any(value < 0 for value in shape):
        raise ModelingError("mindie_sd_custom_op_requires_nonnegative_local_shape")
    return shape


def _tokens_width(op: OpSpec) -> tuple[int, int]:
    shape = _shape(op)
    return shape[0], prod(shape[1:]) if len(shape) > 1 else 1


def _input_dtype(op: OpSpec, meta: dict[str, Any]) -> str:
    dtype = op.dtype or meta.get("input_format") or meta.get("compute_format")
    if not dtype:
        raise ModelingError("mindie_sd_custom_op_requires_declared_input_dtype")
    return str(dtype)


def _ratio(meta: dict[str, Any], option, name: str) -> float:
    value = option if option is not None else meta.get(name)
    if value is None:
        raise ModelingError(f"mindie_sd_custom_op_requires_declared_{name}")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelingError(f"mindie_sd_custom_op_invalid_{name}") from exc
    return value


def _positive_int(meta: dict[str, Any], option, name: str) -> int:
    value = option if option is not None else meta.get(name)
    if type(value) is not int or value <= 0:
        raise ModelingError(f"mindie_sd_custom_op_requires_declared_{name}")
    return value


def _as_shape(value, name):
    if (isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list))
            or not value or any(type(item) is not int or item < 0 for item in value)):
        raise ModelingError(f"mindie_sd_{name}_shape_invalid")
    return tuple(value)


def _bytes(shape, dtype):
    return float(prod(shape) * dtype_bytes(dtype))


def _role(name, shape, dtype, *, materialized=True, alias=None, kind="value"):
    return {
        "name": name, "shape": tuple(shape), "dtype": str(dtype),
        "bytes": _bytes(shape, dtype), "materialized": bool(materialized),
        "alias": alias, "kind": kind,
    }


def _contract(reads, writes, compute, *, attributes, basis, limitations=(), values=None):
    mte2 = sum(item["bytes"] for item in reads)
    mte3 = sum(item["bytes"] for item in writes
                if item.get("materialized", True))
    vector = sum(float(item.get("vector_ops", 0)) for item in compute)
    matrix = {}
    for item in compute:
        for dtype, flops in item.get("matrix_flops", ()):
            matrix[dtype] = matrix.get(dtype, 0) + float(flops)
    bound = dict(values or {})
    for item in reads:
        bound[item["name"]] = {
            "role": "read", "shape": item.get("shape"), "dtype": item.get("dtype"),
            "bytes": item.get("bytes"), "kind": item.get("kind"),
            "producer": item.get("producer"),
        }
    for item in writes:
        bound[item["name"]] = {
            "role": "write", "shape": item.get("shape"), "dtype": item.get("dtype"),
            "bytes": item.get("bytes"), "kind": item.get("kind"),
            "source": item.get("source"),
            "materialized": item.get("materialized", True),
            "alias": item.get("alias"),
        }
    for item in compute:
        for output in item.get("outputs", (item["name"],)):
            bound.setdefault(output, {"role": "internal", "producer": item["name"]})
    public_outputs = tuple(item["name"] for item in writes
                           if item.get("materialized", True)
                           and item.get("kind") in {"value", "legacy"})
    return {
        "reads": tuple(reads), "writes": tuple(writes),
        "compute": tuple(compute), "values": bound,
        "public_outputs": public_outputs,
        "attributes": dict(attributes),
        "mte2_bytes": mte2, "mte3_bytes": mte3, "hbm_bytes": mte2 + mte3,
        "vector_ops": vector,
        "matrix_flops": tuple(sorted(matrix.items())),
        "logical_flops": sum(matrix.values()),
        "basis": basis, "limitations": tuple(limitations),
    }


def _read(name, shape, dtype, *, kind="value", producer=None):
    item = _role(name, shape, dtype, kind=kind)
    item["producer"] = producer
    return item


def _contract_work(contract):
    reads = sum(item["bytes"] for item in contract["reads"])
    writes = sum(item["bytes"] for item in contract["writes"]
                 if item.get("materialized", True))
    matrix = {}
    for item in contract["compute"]:
        for dtype, flops in item.get("matrix_flops", ()):
            matrix[dtype] = matrix.get(dtype, 0) + float(flops)
    return Work(matrix, {"hbm": reads + writes, "mte2": reads,
                         "mte3": writes,
                         "vector": float(contract["vector_ops"])})


def _build_contract_plan(op: OpSpec, geometry):
    contract = dict(dict(op.metadata).get("mindie_sd_contract") or {})
    if not contract:
        contract = _contract(
            [{"name": "declared_read", "bytes": float(op.mte2_bytes),
              "shape": tuple(), "dtype": op.dtype, "materialized": True,
              "alias": None, "kind": "declared", "producer": None}],
            [{"name": "declared_write", "bytes": float(op.mte3_bytes),
              "shape": tuple(), "dtype": op.dtype, "materialized": True,
              "alias": None, "kind": "declared", "source": "declared_compute"}],
            [{"name": "declared_compute", "kind": "declared",
              "vector_ops": float(op.vector_ops),
              "matrix_flops": op.matrix_flops or ((op.dtype, op.flops),),
              "inputs": (), "outputs": ("declared_compute",)}],
            attributes={}, basis="declared_op_fields")
    total = _contract_work(contract)
    node_specs = {}
    role_node = {}
    write_names = {item["name"] for item in contract["writes"]
                   if item.get("materialized", True)}
    compute_names = {item["name"] for item in contract["compute"]}
    for item in contract["reads"]:
        name = "read_" + item["name"]
        if name in node_specs:
            raise ModelingError("mindie_sd_contract_duplicate_stage:" + name)
        node_specs[name] = {"kind": "read", "item": item}
        role_node[item["name"]] = name
    for item in contract["compute"]:
        name = "compute_" + item["name"]
        if name in node_specs:
            raise ModelingError("mindie_sd_contract_duplicate_stage:" + name)
        node_specs[name] = {"kind": "compute", "item": item}
        for output in item.get("outputs", (name,)):
            if output in role_node:
                raise ModelingError("mindie_sd_contract_duplicate_value:" + output)
            role_node[output] = name
    for item in contract["writes"]:
        if not item.get("materialized", True):
            continue
        name = "write_" + item["name"]
        if name in node_specs:
            raise ModelingError("mindie_sd_contract_duplicate_stage:" + name)
        node_specs[name] = {"kind": "write", "item": item}

    def resolve(value):
        if value in role_node:
            return role_node[value]
        if value in write_names:
            return "write_" + value
        if value in compute_names:
            return "compute_" + value
        raise ModelingError("mindie_sd_contract_missing_producer:" + value)

    def dependencies(node):
        spec = node_specs[node]
        item = spec["item"]
        if spec["kind"] == "read":
            producer = item.get("producer")
            return () if producer is None else (resolve(producer),)
        if spec["kind"] == "compute":
            return tuple(resolve(value) for value in item.get("inputs", ()))
        source = item.get("source")
        if source is None:
            raise ModelingError(
                "mindie_sd_contract_write_missing_source:" + item["name"])
        return (resolve(source),)

    order, visited, visiting = [], set(), set()

    def visit(node):
        if node in visited:
            return
        if node in visiting:
            raise ModelingError("mindie_sd_contract_cycle:" + node)
        visiting.add(node)
        for dep in dependencies(node):
            visit(dep)
        visiting.remove(node)
        visited.add(node)
        order.append(node)

    for node in node_specs:
        visit(node)
    workspace_writes = {
        item["name"]: item for item in contract["writes"]
        if item.get("kind") == "workspace" and item.get("materialized", True)}
    workspace_consumers = {}
    for item in contract["reads"]:
        producer = item.get("producer")
        if producer in workspace_writes:
            workspace_consumers["read_" + item["name"]] = producer
    stages = []
    for node in order:
        spec = node_specs[node]
        item = spec["item"]
        if spec["kind"] == "read":
            work = Work(resources={"hbm": item["bytes"], "mte2": item["bytes"]})
        elif spec["kind"] == "compute":
            work = Work(dict(item.get("matrix_flops", ())),
                        {"vector": float(item.get("vector_ops", 0))})
        else:
            work = Work(resources={"hbm": item["bytes"], "mte3": item["bytes"]})
        acquire_name = None
        free_name = None
        if spec["kind"] == "write" and item["name"] in workspace_writes:
            acquire_name = item["name"]
        if spec["kind"] == "read" and node in workspace_consumers:
            free_name = workspace_consumers[node]
        stages.append((node, work, dependencies(node), acquire_name, free_name))

    workspace_pool_name = str(dict(op.metadata).get("workspace_pool", "gm"))
    workspace_grant = dict(op.metadata).get("workspace_grant")

    def program(context):
        grant = dict(getattr(context.op, "metadata", ()) or {}).get(
            "workspace_grant", workspace_grant)
        allocations = {}
        pool_name = workspace_pool_name
        if pool_name not in context.pool_names:
            pool_name = next((name for name in context.pool_names
                              if "workspace" in name or name in {"gm", "hbm"}),
                             pool_name)
        for write_name, item in workspace_writes.items():
            if grant is not None:
                continue
            if pool_name in context.pool_names:
                allocations[write_name] = (context.allocate(
                    "mindie_sd_workspace:" + write_name, pool_name,
                    item["bytes"], (0,)),)
        return KernelProgram(tuple(
            Stage(name, context.bind(work), after=after,
                  acquire=allocations.get(acquire_name, ()),
                  free=allocations.get(free_name, ()))
            for name, work, after, acquire_name, free_name in stages))

    return OperatorPlan(total, program, metadata={
        "mindie_sd_contract": contract,
        "shared_plan": "per_value_read_compute_write_dag",
        "workspace_values": tuple(workspace_writes),
        "workspace_bytes": int(sum(item["bytes"] for item in workspace_writes.values())),
        "workspace_allocation": ("gm" if workspace_writes else "none"),
        "workspace_allocation_pool": workspace_pool_name,
        "workspace_requires_memory_grant": bool(workspace_writes),
        "workspace_grant_reused": workspace_grant is not None,
    })
