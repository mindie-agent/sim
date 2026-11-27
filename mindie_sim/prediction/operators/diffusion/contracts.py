"""Diffusion value contracts and physical stages.

[Role] Assemble diffusion tensor contracts and physical plans.
[Invariants] Each stage retains its value dependencies, transfer work and workspace lifetime.
[Timeouts & Locks] Bounded local computation. No subprocesses or external locks.
[Side-Effect Policy] Creates local records only. No I/O or device access.
"""

from __future__ import annotations

from math import prod
from typing import Any, Literal

from mindie_sim.backends.abi_registry import default_abi_registry
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.physical_mechanisms import DAV_2201_PRESET, DAV_3510_PRESET

from mindie_sim.prediction.mechanism import OperatorPlan, Work
from mindie_sim.prediction.program import KernelProgram, Stage

from .registry import MINDIE_SD_CUSTOM_OPS

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"

SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="Ascend/MindIE-SD",
        revision="9fe772dfb90696e21b5e5d4e92ca07dffab84cf4",
        relative_path="csrc/plugin/register_ops.cpp",
        source_sha256="861fc97a28905b4e9c2ea39f8cd5641725b7b21f355ca8bb237446b5ae86c08a",
    ),
)

_ARCH_TO_SOC = {
    "A2": "Ascend910B",
    "A3": "Ascend910_93",
    "A5": "Ascend950",
}


def _geometry_soc_arch(geometry, op: OpSpec | None = None) -> tuple[str | None, str | None]:
    """Resolve physical Ascend (soc_arch, soc_family) from Geometry or Device."""
    inner = getattr(geometry, "geometry", geometry)
    compute_arch = getattr(geometry, "compute_arch", None) or getattr(inner, "compute_arch", None)
    if compute_arch is None:
        return None, None
    arch_id = str(
        getattr(compute_arch, "arch_id", "") or getattr(compute_arch, "preset", "") or ""
    ).strip().lower()
    if arch_id in {DAV_3510_PRESET, "arch35", "a5"} or "3510" in arch_id:
        return "A5", _ARCH_TO_SOC["A5"]
    if arch_id in {DAV_2201_PRESET, "arch22", "a2", "a3"} or "2201" in arch_id:
        sku = str(getattr(compute_arch, "sku", "") or getattr(inner, "sku", "") or "").lower()
        op_soc = str(
            dict(dict(op.metadata).get("native_implementation") or {}).get("soc_family") or ""
        ).lower() if op is not None else ""
        if "910b" in sku or "910b" in op_soc or arch_id == "a2":
            return "A2", _ARCH_TO_SOC["A2"]
        return "A3", _ARCH_TO_SOC["A3"]
    return None, None


def _expand_unfused_eagle_ffn_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Insert intermediate GM up-projection materialization for A2/A3 unfused EagleFFN."""
    attrs = dict(contract.get("attributes") or {})
    w1 = attrs.get("weight1_shape")
    hidden_shape = attrs.get("hidden_shape")
    hidden_dtype = attrs.get("hidden_dtype")
    if not w1 or not hidden_shape or not hidden_dtype:
        return contract
    w1_layout = attrs.get("weight1_layout", "linear_HxK")
    h_total = int(w1[0] if w1_layout == "linear_HxK" else w1[1])
    up_shape = (int(hidden_shape[0]), h_total)
    up_bytes = _bytes(up_shape, str(hidden_dtype))
    reads = list(contract["reads"])
    writes = list(contract["writes"])
    compute = []
    for item in contract["compute"]:
        if item["name"] == "activation":
            compute.append({**item, "inputs": ("up_read",)})
        else:
            compute.append(dict(item))
    writes.insert(0, {
        "name": "up_store", "shape": up_shape, "dtype": str(hidden_dtype),
        "bytes": up_bytes, "materialized": True, "alias": None,
        "kind": "workspace", "source": "linear1",
    })
    reads.append({
        "name": "up_read", "shape": up_shape, "dtype": str(hidden_dtype),
        "bytes": up_bytes, "materialized": True, "alias": None,
        "kind": "workspace", "producer": "up_store",
    })
    return _contract(
        reads, writes, compute,
        attributes={**attrs, "mindie_sd_unfused_fallback": True},
        basis="unfused_linear_activation_linear_chain",
        limitations=contract.get("limitations", ()),
    )


def _adapt_contract_for_soc(
    op_key: str, contract: dict[str, Any], soc_family: str | None, unfused_ffn: bool,
) -> dict[str, Any]:
    """Align SoC-agnostic contracts to the physical target SoC at prepare time."""
    if not contract or soc_family is None:
        return contract
    if unfused_ffn and op_key == "eagle_ffn":
        return _expand_unfused_eagle_ffn_contract(contract)
    if op_key == "adalayernorm" and soc_family == _ARCH_TO_SOC["A5"]:
        write_names = {item["name"] for item in contract.get("writes", ())}
        if "mean" in write_names and "rstd" in write_names:
            return contract
        y_write = next((item for item in contract.get("writes", ()) if item["name"] == "y"), None)
        if y_write is None or not y_write.get("shape"):
            return contract
        mean_shape = tuple(y_write["shape"][:-1]) + (1,)
        writes = list(contract["writes"])
        for stat_name in ("mean", "rstd"):
            if stat_name not in write_names:
                writes.append({**_role(stat_name, mean_shape, y_write["dtype"], kind="auxiliary"),
                               "source": "norm"})
        return _contract(
            contract["reads"], writes, contract["compute"],
            attributes={**dict(contract.get("attributes") or {}), "adaln_v2_stats_materialized": True},
            basis=contract.get("basis", "three_dimensional_adaln_modulation_and_optional_affine"),
            limitations=contract.get("limitations", ()),
        )
    return contract


def _abi_metadata_for_geometry(op: OpSpec, geometry) -> tuple[dict[str, Any], str | None, bool]:
    """Verify Device-derived SoC legality and record honest sealed ABI availability."""
    soc_arch, soc_family = _geometry_soc_arch(geometry, op)
    if soc_arch is None or soc_family is None:
        return {}, None, False
    op_key = str(op.recipe or "").removeprefix("mindie_sd_")
    spec = MINDIE_SD_CUSTOM_OPS.get(op_key)
    if spec is None:
        return {}, soc_family, False
    unfused_ffn = False
    if soc_family not in spec["supported_socs"]:
        declared_soc = dict(dict(op.metadata).get("native_implementation") or {}).get("soc_family")
        if (op_key == "eagle_ffn" and declared_soc is None
                and soc_family in spec.get("unfused_fallback_socs", ())):
            unfused_ffn = True
        else:
            raise ModelingError(f"mindie_sd_custom_op_soc_unsupported:{op_key}:{soc_family}")
    abi_id = spec["native_kind"]
    available = False if unfused_ffn else default_abi_registry().is_shipping_available(abi_id, soc_arch)
    return {
        "abi_id": abi_id,
        "abi_soc_arch": soc_arch,
        "abi_available_on_device": available,
        **({"mindie_sd_unfused_fallback": True} if unfused_ffn else {}),
    }, soc_family, unfused_ffn


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
    abi_meta, soc_family, unfused_ffn = _abi_metadata_for_geometry(op, geometry)
    op_key = str(op.recipe or "").removeprefix("mindie_sd_")
    contract = dict(dict(op.metadata).get("mindie_sd_contract") or {})
    if contract:
        contract = _adapt_contract_for_soc(op_key, contract, soc_family, unfused_ffn)
    else:
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
        **abi_meta,
    })
