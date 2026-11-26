"""Local-TP rank group resolution and projection sharding rewrites.

[Role] Validate Local-TP groups and rewrite projection collectives across
  local tensor-parallel domains inside the unified sharding subsystem.
[Invariants] Conserves rank-group membership and exact communication volumes.
[Timeouts & Locks] Pure graph rewrite; no I/O or locks.
[Side-Effect Policy] Pure transformation."""
from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

from mindie_sim.catalog.ops import (
    W4A16_DEQUANT_OPS_PER_WEIGHT,
    W4A16_WEIGHT_FORMATS,
    comm_op,
)
from mindie_sim.catalog.projection_placement import (
    ROLE_ATTENTION_INPUT, ROLE_DENSE_MLP_DOWN, ROLE_DENSE_MLP_UP,
    ROLE_DRAFT_LOGITS, ROLE_LINEAR_INPUT, ROLE_LOGITS,
    ROLE_SHARED_EXPERT_DOWN, ROLE_SHARED_EXPERT_UP,
    _padded_local, role_from_category, role_width,
)
from mindie_sim.catalog.weight_terms import vocab_parallel_rows
from mindie_sim.schema.common import ModelingError, align_up, dtype_bytes
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.runtime_features import LOCAL_TP_AXIS_NAMES


LOCAL_TP_AXIS = "local_tp"


def _enabled_widths(parallel: Any) -> tuple[int, ...]:
    return tuple(sorted({
        int(getattr(parallel, name, 1) or 1)
        for name in LOCAL_TP_AXIS_NAMES
    } - {1}))


def _lm_head_tp(parallel: Any) -> int:
    return max(1, int(getattr(parallel, "lm_head_tp", 1) or 1))


def _non_lm_head_widths(parallel: Any) -> tuple[int, ...]:
    return tuple(sorted({
        int(getattr(parallel, name, 1) or 1)
        for name in LOCAL_TP_AXIS_NAMES
        if name != "lm_head_tp"
    } - {1}))


def _model_is_moe(model: Any) -> bool:
    """True when the model architecture is MoE (including dense prefix slices)."""
    if model is None:
        return True
    if int(getattr(model, "num_experts", 0) or 0) > 0:
        return True
    if int(getattr(model, "moe_intermediate_size", 0) or 0) > 0:
        return True
    if any(getattr(lp, "ffn", "") == "moe" for lp in (getattr(model, "layer_plans", ()) or ())):
        return True
    facts = dict(getattr(model, "catalog_facts", ()) or ())
    return int(facts.get("moe_layers", 0) or 0) > 0


def _rank_order(machine: Any) -> tuple[int, ...]:
    return tuple(sorted(int(device.rank) for device in machine.dies))


def _device_map(machine: Any) -> dict[int, Any]:
    return {int(device.rank): device for device in machine.dies}


def _stage_rank_buckets(
    machine: Any, parallel: Any, *, stage: int, stride: int = 1,
) -> dict[tuple[int, int, int, int], list[int]]:
    from mindie_sim.prediction.topology import Topology

    topology = Topology(machine, parallel)
    buckets: dict[tuple[int, int, int, int], list[int]] = {}
    ext_dp = max(1, int(getattr(parallel, "external_dp", 1) or 1))
    internal_replicas = max(1, int(parallel.dp))
    devices = _device_map(machine)
    tp = max(1, int(parallel.tp))
    lane_stride = max(1, int(stride))
    for ext in range(ext_dp):
        for local_rep in range(internal_replicas):
            replica = ext * internal_replicas + local_rep
            ranks = topology.ranks(replica, stage)
            if not ranks:
                if ext == 0:
                    continue
                raise ModelingError(f"local_tp_external_replica_missing:{ext}:stage{stage}")
            for position, rank in enumerate(ranks):
                device = devices.get(int(rank))
                if device is None:
                    raise ModelingError(f"local_tp_rank_undeclared:{rank}")
                lane = (position // tp if int(parallel.pcp) > 1 else 0)
                tp_offset = (position % tp) % lane_stride if lane_stride > 1 else 0
                key = (ext, int(getattr(device, "node", 0)), lane, tp_offset)
                buckets.setdefault(key, []).append(int(rank))
    return buckets


def local_tp_groups(
    machine: Any, parallel: Any, *, stage: int, width: int, stride: int = 1,
) -> tuple[tuple[int, ...], ...]:
    """Partition actual external-DP/node/stage/PCP ranks into groups of width."""
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ModelingError(f"local_tp_width_invalid:{width}")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ModelingError(f"local_tp_stride_invalid:{stride}")
    groups: list[tuple[int, ...]] = []
    buckets = _stage_rank_buckets(machine, parallel, stage=stage, stride=stride)
    for (ext, node, lane, _tp_off), ranks in sorted(buckets.items()):
        del ext
        ordered = tuple(ranks)
        if width > len(ordered):
            raise ModelingError(
                f"local_tp_width_exceeds_node_lane:{width}>{len(ordered)}")
        if len(ordered) % width != 0:
            raise ModelingError(
                f"local_tp_width_not_divisor:{width}!|{len(ordered)}:node{node}:lane{lane}")
        for start in range(0, len(ordered), width):
            groups.append(ordered[start:start + width])
    if not groups:
        raise ModelingError(f"local_tp_stage_has_no_ranks:{stage}")
    return tuple(groups)


def _union_components(keys: Iterable[int], edges: Iterable[tuple[int, int]]) -> tuple[tuple[int, ...], ...]:
    parent = {int(k): int(k) for k in keys}

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            nxt = parent[node]
            parent[node] = root
            node = nxt
        return root

    for left, right in edges:
        if left in parent and right in parent:
            a, b = find(left), find(right)
            if a != b:
                parent[max(a, b)] = min(a, b)
    buckets: dict[int, list[int]] = {}
    for node in sorted(parent):
        buckets.setdefault(find(node), []).append(node)
    return tuple(tuple(sorted(items)) for _, items in sorted(buckets.items()))


def local_tp_cohorts(
    machine: Any, parallel: Any,
) -> tuple[tuple[int, ...], ...]:
    """Connected components of the union of EP and enabled local TP groups."""
    from mindie_sim.prediction.topology import Topology

    topology = Topology(machine, parallel)
    edges: list[tuple[int, int]] = []
    ext_dp = max(1, int(getattr(parallel, "external_dp", 1) or 1))
    replicas = max(1, ext_dp * int(parallel.dp))
    cp_spans = bool(getattr(parallel, "cp_spans_dp", False))
    tp = max(1, int(parallel.tp))
    for stage in range(max(1, int(parallel.pp))):
        for replica in range(replicas):
            for group in topology.groups("ep", replica, stage):
                edges.extend((group[0], rank) for rank in group[1:])
            if cp_spans:
                for group in topology.groups("cp", replica, stage):
                    edges.extend((group[0], rank) for rank in group[1:])
        for width in _non_lm_head_widths(parallel):
            for group in local_tp_groups(machine, parallel, stage=stage, width=width):
                edges.extend((group[0], rank) for rank in group[1:])
            if tp > 1 and width > tp and width % tp == 0:
                for group in local_tp_groups(
                        machine, parallel, stage=stage, width=width // tp, stride=tp):
                    edges.extend((group[0], rank) for rank in group[1:])
        lm_w = _lm_head_tp(parallel)
        if lm_w > 1:
            for group in local_tp_groups(
                    machine, parallel, stage=stage, width=lm_w, stride=tp):
                edges.extend((group[0], rank) for rank in group[1:])
    return _union_components(_rank_order(machine), edges)


def local_tp_replica_cohorts(
    machine: Any, parallel: Any,
) -> tuple[tuple[int, ...], ...]:
    """Projection of physical EP + DCP-DP + local-TP connected components onto DP replicas."""
    from mindie_sim.prediction.topology import Topology

    topology = Topology(machine, parallel)
    dp = max(1, int(parallel.dp))
    cp_spans = bool(getattr(parallel, "cp_spans_dp", False))
    tp = max(1, int(parallel.tp))
    edges: list[tuple[int, int]] = []
    for stage in range(max(1, int(parallel.pp))):
        owner = {int(rank): r for r in range(dp)
                 for rank in topology.ranks(r, stage)}
        for r in range(dp):
            for group in topology.groups("ep", r, stage):
                reps = [owner[int(rk)] for rk in group if int(rk) in owner]
                edges.extend((reps[0], rep) for rep in reps[1:])
            if cp_spans:
                for group in topology.groups("cp", r, stage):
                    reps = [owner[int(rk)] for rk in group if int(rk) in owner]
                    edges.extend((reps[0], rep) for rep in reps[1:])
        for width in _non_lm_head_widths(parallel):
            for group in local_tp_groups(machine, parallel, stage=stage, width=width):
                reps = [owner[int(rk)] for rk in group if int(rk) in owner]
                edges.extend((reps[0], rep) for rep in reps[1:])
            if tp > 1 and width > tp and width % tp == 0:
                for group in local_tp_groups(
                        machine, parallel, stage=stage, width=width // tp, stride=tp):
                    reps = [owner[int(rk)] for rk in group if int(rk) in owner]
                    edges.extend((reps[0], rep) for rep in reps[1:])
        lm_w = _lm_head_tp(parallel)
        if lm_w > 1:
            for group in local_tp_groups(
                    machine, parallel, stage=stage, width=lm_w, stride=tp):
                reps = [owner[int(rk)] for rk in group if int(rk) in owner]
                edges.extend((reps[0], rep) for rep in reps[1:])
    return _union_components(range(dp), edges)


def _validate_stage_groups(
    hardware: Any, parallel: Any, devices: Mapping[int, Any], expected: set[int],
    *, width: int, stride: int = 1,
) -> None:
    seen: set[int] = set()
    for stage in range(max(1, int(parallel.pp))):
        groups = local_tp_groups(hardware, parallel, stage=stage, width=width, stride=stride)
        stage_ranks = [rank for group in groups for rank in group]
        if len(stage_ranks) != len(set(stage_ranks)):
            raise ModelingError("local_tp_rank_in_multiple_groups")
        for group in groups:
            signatures = {devices[rank].resource_signature for rank in group}
            if len(signatures) > 1:
                raise ModelingError("local_tp_group_spans_heterogeneous_devices")
            nodes = {int(getattr(devices[rank], "node", 0)) for rank in group}
            if len(nodes) > 1:
                raise ModelingError("local_tp_group_spans_nodes")
        seen.update(stage_ranks)
    if not expected <= seen:
        raise ModelingError("local_tp_group_coverage_incomplete")


def validate_local_projection_tp(case: Any) -> None:
    """Validate the declared local widths against physical rank ownership."""
    parallel = case.runtime.parallel
    hardware = case.machine
    widths = _enabled_widths(parallel)
    if not widths:
        return
    model = getattr(case, "model", None)
    if model is not None and not _model_is_moe(model):
        raise ModelingError("local_tp_requires_moe_model")
    dp = max(1, int(parallel.dp))
    tp = max(1, int(parallel.tp))
    lane_plane = dp * tp
    ranks_per_node = max(1, int(getattr(hardware, "ranks_per_node", 1) or 1))
    hardware.validate_placement(parallel)
    devices = _device_map(hardware)
    declared = {int(device.rank) for device in hardware.dies}
    from mindie_sim.prediction.topology import Topology

    topology = Topology(hardware, parallel)
    ext_dp = max(1, int(getattr(parallel, "external_dp", 1) or 1))
    expected = {
        rank
        for replica in range(max(1, ext_dp * dp))
        for stage in range(max(1, int(parallel.pp)))
        for rank in topology.ranks(replica, stage)
    }
    if not expected <= declared:
        raise ModelingError("local_tp_physical_rank_missing")
    for width in widths:
        if width > ranks_per_node:
            raise ModelingError(
                f"local_tp_width_exceeds_ranks_per_node:{width}>{ranks_per_node}")
        if width > lane_plane:
            raise ModelingError(
                f"local_tp_width_exceeds_lane_plane:{width}>{lane_plane}")
        if dp % width != 0:
            raise ModelingError(f"local_tp_width_not_divisor_of_dp:{width}!|{dp}")
    for width in _non_lm_head_widths(parallel):
        if tp > 1 and width > tp and width % tp != 0:
            raise ModelingError(f"local_tp_width_not_multiple_of_tp:{width}%{tp}")
        _validate_stage_groups(hardware, parallel, devices, expected, width=width)
    lm_w = _lm_head_tp(parallel)
    if lm_w > 1:
        _validate_stage_groups(hardware, parallel, devices, expected, width=lm_w, stride=tp)


def local_tp_spans_dp(parallel: Any) -> bool:
    """True when an enabled local width or DP-cohort DCP crosses a DP boundary."""
    tp = max(1, int(getattr(parallel, "tp", 1) or 1))
    if int(parallel.dp) <= 1:
        return False
    if bool(getattr(parallel, "cp_spans_dp", False)):
        return True
    if _lm_head_tp(parallel) > 1:
        return True
    return any(width > tp for width in _non_lm_head_widths(parallel))


def rank_zero_graph_reuse(parallel: Any, replicas: int, has_device_group: bool) -> bool:
    """Legacy rank-zero graph reuse remains valid only without cross-DP groups."""
    ep_tp_w = max(1, int(parallel.tp) // max(1, int(getattr(parallel, "etp", 1) or 1)))
    ep_spans_pcp = int(parallel.pcp) > 1 and int(parallel.ep) > int(parallel.dp) * ep_tp_w
    ep_intra_w = ep_tp_w * int(parallel.pcp) if ep_spans_pcp else ep_tp_w
    ep_spans_dp = bool(parallel.ep_spans_dp) or (int(parallel.ep) > ep_intra_w)
    return (int(replicas) > 1 and not has_device_group
            and not ep_spans_dp
            and not bool(getattr(parallel, "cp_spans_dp", False))
            and not local_tp_spans_dp(parallel))


def expert_tp_groups(
    topology: Any, *, replica: int = 0, stage: int = 0,
) -> tuple[tuple[int, ...], ...]:
    """Construct expert-TP rank groups from the resolved ``parallel.etp`` width.

    When an explicit EP ``rank_table`` is declared, ranks inside each ``tp``
    slice are grouped by equal EP position so only ranks owning the same
    experts share an expert-TP group.
    """
    parallel = topology.parallel
    sources = tuple(int(rank) for rank in topology.ranks(replica, stage))
    if not sources:
        raise ModelingError(f"expert_tp_stage_has_no_ranks:{stage}")
    tp = max(1, int(parallel.tp))
    etp = max(1, int(parallel.etp))
    if tp % etp != 0:
        raise ModelingError(f"expert_tp_not_divisor_of_tp:{etp}!|{tp}")
    if etp <= 1:
        return tuple((rank,) for rank in sources)
    table = getattr(parallel, "rank_table", None)
    if table:
        ep_pos = {
            int(rank): pos
            for ep_group in table
            for pos, rank in enumerate(ep_group)
        }
        groups: list[tuple[int, ...]] = []
        for base in range(0, len(sources), tp):
            tp_slice = sources[base : base + tp]
            if all(rank in ep_pos for rank in tp_slice):
                by_pos: dict[int, list[int]] = {}
                for rank in tp_slice:
                    by_pos.setdefault(ep_pos[rank], []).append(rank)
                groups.extend(tuple(members) for members in by_pos.values() if members)
                continue
            stride = tp // etp
            groups.extend(
                group
                for offset in range(stride)
                if (group := tp_slice[offset::stride])
            )
        return tuple(groups)
    stride = tp // etp
    return tuple(
        group
        for base in range(0, len(sources), tp)
        for offset in range(stride)
        if (group := sources[base + offset : base + tp : stride])
    )


def collective_requires_cohort(op: Any, topology: Any) -> bool:
    """True when the op's actual group spans more than one DP replica."""
    metadata = dict(getattr(op, "metadata", ()) or ())
    parallel = topology.parallel
    stage = int(metadata.get("source_stage", getattr(op, "stage", 0)))
    ext_dp = max(1, int(getattr(parallel, "external_dp", 1) or 1))
    replicas = max(1, ext_dp * int(parallel.dp))
    owner: dict[int, int] = {}
    for replica in range(replicas):
        for rank in topology.ranks(replica, stage):
            owner[int(rank)] = replica
    axis = str(metadata.get("group_axis", "tp"))
    if axis not in {
        "cp", "dcp", "dcp_tp", "dcp_pcp", "dcp_dp", "dcp_a2a",
        "etp", "pcp", "ep", "tp", "dp", LOCAL_TP_AXIS,
    }:
        return False
    declared = metadata.get("group_ranks")
    rep = int(metadata.get("replica", 0))
    if declared:
        groups: Sequence[Sequence[int]] = (tuple(declared),)
    elif axis == "etp":
        groups = expert_tp_groups(topology, replica=rep, stage=stage)
    else:
        groups = tuple(topology.groups(axis, rep, stage, metadata=metadata))
    for group in groups:
        owners = {owner.get(int(rank)) for rank in group if int(rank) in owner}
        owners.discard(None)
        if len(owners) > 1:
            return True
    return False


ALIGN_ELEMENT = 16
FALLBACK_ACTIVATION = "bf16"
FALLBACK_OUTPUT = "bf16"
RETARGET_ADAPTERS: dict[str, Any] = {}
ROLE_COLUMN = "column"
ROLE_ROW = "row"
ROLE_GROUP_A = "group_pair_a"
ROLE_GROUP_B = "group_pair_b"
SDK_PLACEMENT = "local_tp"
_TRANSPORT_KEEP_META = frozenset({
    "layer_scope", "producer_scope", "draft_step", "structure_block",
    "structure_role", "graph_replay_scope", "graph_bucket", "fusion_boundary",
})


def register_retarget_adapter(implementation: str, adapter: Any) -> None:
    if not implementation or not callable(adapter):
        raise ModelingError("local_tp_retarget_adapter_invalid")
    RETARGET_ADAPTERS[str(implementation)] = adapter


def _width(runtime: Any, role: str) -> int:
    return role_width(role, runtime.parallel)


def _role(metadata: Mapping[str, Any]) -> str:
    explicit = str(metadata.get("projection_role", ""))
    if explicit:
        return explicit
    ffn_role = str(metadata.get("ffn_role", ""))
    ffn_cat = str(metadata.get("ffn_category", ""))
    if ffn_role == "gate_up" and ffn_cat == "dense_mlp":
        return ROLE_DENSE_MLP_UP
    if ffn_role == "down" and ffn_cat == "dense_mlp":
        return ROLE_DENSE_MLP_DOWN
    if ffn_role == "gate_up" and ffn_cat == "shared_expert":
        return ROLE_SHARED_EXPERT_UP
    if ffn_role == "down" and ffn_cat == "shared_expert":
        return ROLE_SHARED_EXPERT_DOWN
    return role_from_category(
        str(metadata.get("quant_category", metadata.get("catalog_category", ""))))


def _is_globally_replicated(
    op: OpSpec, role: str, metadata: Mapping[str, Any], runtime: Any, model: Any,
) -> bool:
    cat = str(metadata.get("quant_category", metadata.get("catalog_category", ""))).lower()
    if role == ROLE_ATTENTION_INPUT:
        return cat in {"mla.q_a", "mla.kv_a"} or op.op_id.endswith((".q_a", ".kv_a", ".wkv"))
    if role in {ROLE_SHARED_EXPERT_UP, ROLE_SHARED_EXPERT_DOWN}:
        return bool(dict(getattr(runtime, "metadata", ()) or ()).get("enable_shared_expert_dp", False))
    if role in {"logits", "draft_logits"} and model is not None:
        vocab = int(getattr(model, "vocab_size", 0) or 0)
        return vocab > 0 and len(op.local_shape) > 1 and int(op.local_shape[1]) >= vocab
    return False


def _effective_factor(
    op: OpSpec, role: str, metadata: Mapping[str, Any], runtime: Any, model: Any,
) -> tuple[int, int]:
    """Return ``(factor, stride)`` for one matrix op under global TP + Local-TP."""
    width = _width(runtime, role)
    if width <= 1:
        return 1, 1
    tp = max(1, int(getattr(runtime.parallel, "tp", 1) or 1))
    if role in {ROLE_LOGITS, ROLE_DRAFT_LOGITS}:
        return width, tp
    if tp <= 1 or _is_globally_replicated(op, role, metadata, runtime, model):
        return width, 1
    if width == tp:
        return 1, 1
    if width < tp or width % tp != 0:
        raise ModelingError(f"local_tp_width_not_multiple_of_tp:{width}%{tp}")
    return width // tp, tp


def _quant_for_op(model: Any, metadata: Mapping[str, Any]) -> Any:
    if model is None:
        return None
    if str(metadata.get("layer_scope", "")) == "draft":
        aux = getattr(model, "auxiliary", None)
        draft = getattr(aux, "mtp", None) or getattr(aux, "dspark", None)
        if draft is not None and getattr(draft, "quant", None) is not None:
            return draft.quant
    return getattr(model, "quant", None)


def _local_dimension(size: int, width: int, quant: Any = None) -> int:
    if width <= 1:
        return int(size)
    # A row-parallel split cuts the input axis, so the local hint is padded to
    # the weight-scale granule (512/TP8 with a 128-column block becomes 128).
    term = SimpleNamespace(quant=quant)
    return max(1, int(_padded_local(int(size), int(width), term)))


def _local_output_width(size: int, width: int) -> int:
    """Exact per-rank output features of one packed column split."""
    if width <= 1:
        return int(size)
    return max(1, (int(size) + int(width) - 1) // int(width))


def _slice_local(item: Mapping[str, Any], width: int, quant: Any = None) -> int:
    """Local output extent of one packed slice; replicated and KV-head slices stay whole."""
    shape = tuple(item.get("shape", ()))
    size = int(shape[-1]) if shape else 0
    if item.get("replicated"):
        return size
    if str(item.get("shard", "")) == "kv_out":
        units = int(item.get("units", 0) or 0)
        unit_width = int(item.get("unit_width", 0) or 0)
        if units > 0 and unit_width > 0:
            local_units = max(1, (units + int(width) - 1) // int(width))
            return local_units * unit_width
    return _local_output_width(size, width)


def _is_replicated_slice(item: Mapping[str, Any], width: int) -> bool:
    if item.get("replicated"):
        return True
    if str(item.get("shard", "")) == "kv_out" and width > 1:
        shape = tuple(item.get("shape", ()))
        size = int(shape[-1]) if shape else 0
        return size > 0 and _slice_local(item, width) == size
    return False


def _remote_exchange_width(
    metadata: Mapping[str, Any], width: int, local_n: int, full_n: int,
    quant: Any = None, *, is_logits: bool = False,
) -> int:
    """Distinct remote output columns each rank needs to restore its full row."""
    if is_logits:
        return max(0, (int(width) - 1) * int(local_n))
    slices = metadata.get("packed_projection_slices")
    if slices:
        remote = 0
        for item in slices:
            shape = tuple(item.get("shape", ()))
            size = int(shape[-1]) if shape else 0
            local_size = _slice_local(item, width, quant)
            remote += max(0, size - local_size)
        return remote
    return max(0, int(full_n) - int(local_n))


def _packed_local_width(
    metadata: Mapping[str, Any], width: int, total: int, quant: Any = None,
    *, role: str = "",
) -> int:
    slices = metadata.get("packed_projection_slices")
    if not slices:
        if role in {ROLE_LOGITS, ROLE_DRAFT_LOGITS}:
            explicit_pad = int(metadata.get("vocab_padding_size") or 0)
            width_i = max(1, int(width))
            if explicit_pad > 0:
                alignment = math.lcm(explicit_pad, width_i)
                return int(align_up(int(total), alignment)) // width_i
            return vocab_parallel_rows(total, width_i)
        if role in {
            ROLE_ATTENTION_INPUT, ROLE_LINEAR_INPUT,
            ROLE_DENSE_MLP_UP, ROLE_SHARED_EXPERT_UP,
        }:
            return _local_output_width(total, width)
        return _local_dimension(total, width, quant)
    return max(1, sum(_slice_local(s, width, quant) for s in slices))


def _localize_packed_slices(
    metadata: dict[str, Any], width: int, quant: Any = None, *, rows: int | None = None,
) -> None:
    slices = metadata.get("packed_projection_slices")
    if not slices:
        return
    # Every slice is a column view into the one localized packed output, so a
    # row advances by the parent buffer's full local width, not by the slice's
    # own width.  Using the slice width makes ``Q[1, 0]`` alias ``K[0, 0]``.
    local_sizes = [_slice_local(item, width, quant) for item in slices]
    packed_width = sum(local_sizes)
    offset, local_slices = 0, []
    for item, local_size in zip(slices, local_sizes):
        shape = tuple(item.get("shape", ()))
        out_rows = int(rows) if rows is not None else (int(shape[0]) if shape else 0)
        entry = {**dict(item), "shape": (out_rows, local_size),
                 "strides": (packed_width, 1), "offset_elements": offset}
        if _is_replicated_slice(item, width):
            entry["replicated"] = True
        local_slices.append(entry)
        offset += local_size
    metadata["packed_projection_slices"] = tuple(local_slices)


def _dtype(metadata: Mapping[str, Any], key: str, fallback: str) -> str:
    value = metadata.get(key)
    return str(value) if value else fallback


def _groups_for(
    machine: Any, runtime: Any, replica: int, stage: int, width: int,
    *, stride: int = 1,
) -> tuple[int, ...]:
    from mindie_sim.prediction.topology import Topology
    ranks = Topology(machine, runtime.parallel).ranks(replica, stage)
    if not ranks:
        raise ModelingError(f"local_tp_stage_has_no_ranks:{stage}")
    anchor = int(ranks[0])
    for group in local_tp_groups(
            machine, runtime.parallel, stage=stage, width=width, stride=stride):
        if anchor in group:
            return group
    raise ModelingError(f"local_tp_group_missing:{anchor}")


def _row_counts(cohort_views: Mapping[int, Sequence[Any]], group: Sequence[int],
                fallback: int, peer_rows: Mapping[int, int] | None = None) -> list[int]:
    counts: list[int] = []
    for rk in (int(rank) for rank in group):
        if peer_rows is not None and rk in peer_rows:
            counts.append(max(0, int(peer_rows[rk])))
        elif cohort_views and rk in cohort_views and cohort_views[rk] is not None:
            counts.append(sum(int(view.logical_q_len) for view in cohort_views[rk]))
        else:
            counts.append(int(fallback))
    return counts


def _transport_metadata(op: OpSpec, *, role: str, group: Sequence[int], width: int,
                        family: str, scope: str,
                        stride: int = 1) -> tuple[tuple[str, Any], ...]:
    src = dict(op.metadata)
    kept = {k: src[k] for k in _TRANSPORT_KEEP_META if k in src}
    kept.update({
        "collective": family, "group_axis": SDK_PLACEMENT,
        "comm_axis": "tp",
        "comm_group_size": int(len(group) or width),
        "group_ranks": tuple(int(rank) for rank in group),
        "local_tp_width": int(width), "projection_role": role,
        "placement_scope": str(src.get("placement_scope") or src.get("layer_scope") or scope or "target"),
        "local_tp_applied": True, "payload_dtype": str(src.get("payload_dtype", "")),
    })
    if stride > 1:
        kept["local_tp_stride"] = int(stride)
    return tuple(sorted(kept.items(), key=lambda pair: pair[0]))


def _collective(op: OpSpec, *, suffix: str, family: str, recipe: str, payload: float,
                shape: tuple[int, ...], role: str, group: Sequence[int], width: int,
                deps: tuple[str, ...], scope: str, dtype: str,
                public_id: str | None = None, stride: int = 1) -> OpSpec:
    built = comm_op(
        op_id=f"{op.op_id}:local_tp:{suffix}", deps=deps, stage=op.stage,
        layer=op.layer, requests=op.requests, family=family, phase=op.phase,
        recipe=recipe, link_bytes=payload, shape=shape,
        metadata=_transport_metadata(op, role=role, group=group, width=width,
                                     family=family, scope=scope, stride=stride),
        notes=(f"local_tp_{role}", f"local_tp_{suffix}"),
    )
    return replace(built, op_id=public_id or built.op_id, dtype=dtype)


def _scaled_matrix_flops(op: OpSpec, new_flops: float) -> tuple[tuple[str, float], ...]:
    if not op.matrix_flops:
        return ()
    total = sum(float(v) for _, v in op.matrix_flops)
    if total > 0.0:
        ratio = new_flops / total
        return tuple((dt, float(v) * ratio) for dt, v in op.matrix_flops)
    return tuple((dt, new_flops if idx == 0 else 0.0) for idx, (dt, _) in enumerate(op.matrix_flops))


def _rewrite_matrix(op: OpSpec, *, gemm_id: str, rows: int, cols: int, reduced: int,
                    role: str, width: int, dep_ids: tuple[str, ...],
                    weight_ratio: float, quant: Any, groups: int = 1,
                    extra_metadata: Mapping[str, Any] | None = None,
                    scale_ratio: float | None = None) -> OpSpec:
    orig_meta = dict(op.metadata)
    metadata = {**orig_meta, **dict(extra_metadata or ())}
    original_weight = float(metadata.get("packed_weight_bytes", 0.0) or 0.0)
    cat = str(metadata.get("quant_category", metadata.get("catalog_category", "")) or "")
    if quant is not None and cat and rows > 0:
        local_weight = float(quant.matrix_bytes(
            cols if groups <= 1 else (groups * cols), reduced, cat))
    else:
        local_weight = original_weight * float(weight_ratio) if rows > 0 else 0.0
    act = _dtype(metadata, "activation_format", op.dtype or FALLBACK_ACTIVATION)
    out = _dtype(metadata, "output_format", FALLBACK_OUTPUT)
    m, n, k = int(rows), int(cols), int(reduced)
    in_b, out_b = float(m * k) * dtype_bytes(act), float(m * n) * dtype_bytes(out)
    orig_m = max(1, int(op.local_shape[0]))
    orig_n = max(1, int(op.local_shape[1]))
    orig_k = max(1, int(op.local_shape[2]))
    orig_groups = max(1, int(orig_meta.get("matrix_groups", 1) or 1))
    new_groups = max(1, int(groups))
    weight_elem_ratio = (
        float(new_groups * n * k) / float(orig_groups * orig_n * orig_k)
    )
    ratio = float(m) / float(orig_m) if scale_ratio is None else float(scale_ratio)
    scale_shape, geometry_bytes = _local_scale_geometry(metadata, m, k)
    scale_b = (geometry_bytes if scale_shape else
               float(metadata.get("activation_scale_bytes", 0.0) or 0.0)
               * ratio)
    if scale_shape:
        metadata["activation_scale_shape"] = scale_shape
    epilogue = metadata.get("matrix_epilogue")
    w_fmt = str(metadata.get("weight_format", "") or "").strip().lower()
    comp_fmt = str(metadata.get("compute_format", op.dtype or "") or "").strip().lower()
    is_w4a16 = (
        metadata.get("dequant") == "groupwise_fused"
        or (w_fmt in W4A16_WEIGHT_FORMATS and comp_fmt != "int8")
    )
    if isinstance(epilogue, Mapping) and epilogue.get("kind") == "row_column_scale":
        ep_copy = dict(epilogue)
        ep_copy["row_scale_bytes"] = float(scale_b) if m > 0 else 0.0
        col_scale = float(ep_copy.get("column_scale_bytes", 0.0) or 0.0)
        ep_copy["column_scale_bytes"] = (
            col_scale * float(new_groups * n) / float(orig_groups * orig_n)
            if m > 0 else 0.0
        )
        metadata["matrix_epilogue"] = ep_copy
        ops_per_out = float(ep_copy.get("operations_per_output", 0) or 0)
        new_vector_ops = ops_per_out * float(m * n)
    elif is_w4a16:
        if m > 0:
            if float(op.vector_ops) > 0.0:
                new_vector_ops = float(op.vector_ops) * weight_elem_ratio
            else:
                ops_per_w = float(
                    metadata.get("dequant_ops_per_weight", W4A16_DEQUANT_OPS_PER_WEIGHT)
                    or W4A16_DEQUANT_OPS_PER_WEIGHT
                )
                base_elems = float(metadata.get("dequant_weight_elements", 0.0) or 0.0)
                new_elems = (
                    base_elems * weight_elem_ratio
                    if base_elems > 0.0 else float(new_groups * n * k)
                )
                new_vector_ops = ops_per_w * new_elems
            if "dequant_weight_elements" in metadata:
                metadata["dequant_weight_elements"] = (
                    float(metadata["dequant_weight_elements"]) * weight_elem_ratio
                )
            elif metadata.get("dequant") != "groupwise_fused":
                metadata["dequant"] = "groupwise_fused"
                metadata["dequant_ops_per_weight"] = float(W4A16_DEQUANT_OPS_PER_WEIGHT)
                metadata["dequant_weight_elements"] = float(new_groups * n * k)
        else:
            new_vector_ops = 0.0
    elif float(op.vector_ops) > 0.0:
        new_vector_ops = (
            float(op.vector_ops) * float(m * n) / float(orig_m * orig_n)
            if m > 0 else 0.0
        )
    else:
        new_vector_ops = 0.0
    metadata.update({
        "packed_weight_bytes": local_weight, "weight_bytes": local_weight,
        "activation_scale_bytes": scale_b, "placement_role": role,
        "local_tp_width": int(width), "local_tp_applied": True,
        "matrix_groups": max(1, int(groups)),
    })
    new_flops = float(2 * m * n * k)
    pad_m = align_up(m, ALIGN_ELEMENT) if m > 0 else 0
    return replace(
        op, op_id=gemm_id, local_shape=(m, n, k),
        padded_shape=(pad_m, align_up(n, ALIGN_ELEMENT), align_up(k, ALIGN_ELEMENT)),
        launch_shape=(m, n, k), flops=new_flops,
        matrix_flops=_scaled_matrix_flops(op, new_flops),
        vector_ops=new_vector_ops,
        hbm_bytes=in_b + local_weight + out_b + scale_b,
        mte2_bytes=in_b + local_weight + scale_b, mte3_bytes=out_b,
        deps=dep_ids, metadata=tuple(sorted(metadata.items(), key=lambda pair: pair[0])),
    )


def _partition_kind(role: str, metadata: Mapping[str, Any]) -> str:
    cat = str(metadata.get("quant_category", metadata.get("catalog_category", ""))).lower()
    if cat == "mla.o_a":
        return ROLE_GROUP_A
    if cat == "mla.o_b":
        return ROLE_GROUP_B
    if role in {"attention_output", "linear_output", ROLE_DENSE_MLP_DOWN, ROLE_SHARED_EXPERT_DOWN}:
        return ROLE_ROW
    return ROLE_COLUMN


def _empty_passthrough(op: OpSpec, metadata: Mapping[str, Any], width: int) -> OpSpec:
    meta = {
        **dict(metadata), "local_tp_applied": True, "local_tp_width": int(width),
        "packed_weight_bytes": 0.0, "weight_bytes": 0.0, "activation_scale_bytes": 0.0,
    }
    epilogue = meta.get("matrix_epilogue")
    if isinstance(epilogue, Mapping):
        ep_copy = dict(epilogue)
        if "row_scale_bytes" in ep_copy:
            ep_copy["row_scale_bytes"] = 0.0
        if "column_scale_bytes" in ep_copy:
            ep_copy["column_scale_bytes"] = 0.0
        meta["matrix_epilogue"] = ep_copy
    return replace(op, flops=0.0, matrix_flops=(), vector_ops=0.0,
                   hbm_bytes=0.0, mte2_bytes=0.0, mte3_bytes=0.0,
                   metadata=tuple(sorted(meta.items(), key=lambda pair: pair[0])))


def _local_scale_geometry(
    metadata: Mapping[str, Any], rows: int, local_k: int,
) -> tuple[tuple[int, ...], float]:
    """Received-row scale shape and bytes after the input axis is sharded."""
    shape = tuple(metadata.get("activation_scale_shape") or ())
    total = float(metadata.get("activation_scale_bytes", 0.0) or 0.0)
    group = int(metadata.get("activation_group_size") or 0)
    if not shape or total <= 0.0 or group <= 0:
        return (), 0.0
    per_row = total / float(max(1, int(shape[0])))
    old_tail = tuple(int(extent) for extent in shape[1:])
    if not old_tail:
        # One per-row scale: the received row count is the only change.
        return (int(rows),), per_row * float(rows)
    old_elements = 1
    for extent in old_tail:
        old_elements *= extent
    if len(old_tail) == 2:
        pack = max(1, old_tail[1])
        groups = (int(local_k) + group - 1) // group
        new_tail = ((groups + pack - 1) // pack, pack)
    else:
        new_tail = ((int(local_k) + group - 1) // group,)
    elements = 1
    for extent in new_tail:
        elements *= extent
    per_scale = per_row / float(max(1, old_elements))
    return (int(rows), *new_tail), per_scale * float(elements) * float(rows)


def _received_quantizer(op: OpSpec, *, rows: int, local_k: int, full_k: int,
                        exchange_id: str, width: int) -> OpSpec:
    """Move one pre-exchange activation quantizer onto the received local rows."""
    meta = dict(op.metadata)
    in_dtype = _dtype(meta, "input_format", FALLBACK_ACTIVATION)
    out_dtype = _dtype(meta, "output_format", op.dtype or in_dtype)
    old_rows = max(1, int(meta.get("local_tp_template_rows") or op.local_shape[0]))
    old_k = max(1, int(op.local_shape[1]) if len(op.local_shape) > 1 else full_k)
    ratio = float(rows) * float(local_k) / (float(old_rows) * float(old_k))
    scale_shape, scale_bytes = _local_scale_geometry(meta, rows, local_k)
    read = float(rows * local_k) * dtype_bytes(in_dtype)
    write = float(rows * local_k) * dtype_bytes(out_dtype) + scale_bytes
    meta.update({
        "local_tp_applied": True, "local_tp_width": int(width),
        "local_tp_quantization_scope": "received_expert_input_rows",
        "local_tp_pre_exchange_dtype": in_dtype,
        "source_rows": int(rows),
    })
    if scale_shape:
        meta["activation_scale_shape"] = scale_shape
        meta["activation_scale_bytes"] = scale_bytes
    pad_rows = align_up(rows, ALIGN_ELEMENT)
    return replace(
        op, deps=(exchange_id,),
        local_shape=(int(rows), int(local_k)), launch_shape=(int(rows), int(local_k)),
        padded_shape=(pad_rows, align_up(int(local_k), ALIGN_ELEMENT)),
        semantic_shape=(int(rows), int(local_k)),
        vector_ops=float(op.vector_ops) * ratio,
        mte2_bytes=read, mte3_bytes=write, hbm_bytes=read + write, flops=0.0,
        metadata=tuple(sorted(meta.items(), key=lambda pair: pair[0])),
    )


def _row_pre_quantizer(op: OpSpec, by_id: Mapping[str, OpSpec],
                       users: Mapping[str, list[str]]) -> OpSpec | None:
    """The single pre-GEMM activation quantizer this row matrix consumes."""
    if len(op.deps) != 1:
        return None
    producer = by_id.get(op.deps[0])
    if producer is None or producer.plan_required:
        return None
    if producer.family not in {"quantize", "rmsnorm_quant"} or len(producer.deps) != 1:
        return None
    if users.get(producer.op_id) != [op.op_id]:
        return None
    if str(dict(producer.metadata).get("quantization_scope", "")) == "moe_dispatch_expert_input":
        return None
    return producer


def _match_paired_ffn(
    op: OpSpec, role: str, by_id: Mapping[str, OpSpec],
    users: Mapping[str, list[str]],
) -> tuple[OpSpec, OpSpec | None, OpSpec] | None:
    """Match ``gate_up -> activation [-> quant] -> down`` for single-block Megatron Local-TP."""
    if role not in {ROLE_DENSE_MLP_UP, ROLE_SHARED_EXPERT_UP}:
        return None
    meta = dict(op.metadata)
    if meta.get("ffn_role") != "gate_up":
        return None
    cat = str(meta.get("ffn_category", ""))
    act_users = users.get(op.op_id) or []
    if len(act_users) != 1:
        return None
    act_op = by_id.get(act_users[0])
    if act_op is None or dict(act_op.metadata).get("ffn_role") != "activation":
        return None
    next_users = users.get(act_op.op_id) or []
    if len(next_users) != 1:
        return None
    nxt = by_id.get(next_users[0])
    if nxt is None:
        return None
    pre_quant: OpSpec | None = None
    down = nxt
    if dict(nxt.metadata).get("ffn_role") != "down":
        if nxt.family not in {"quantize", "rmsnorm_quant"}:
            return None
        down_users = users.get(nxt.op_id) or []
        if len(down_users) != 1:
            return None
        pre_quant = nxt
        down = by_id.get(down_users[0])
        if down is None or dict(down.metadata).get("ffn_role") != "down":
            return None
    if str(dict(down.metadata).get("ffn_category", "")) != cat:
        return None
    return act_op, pre_quant, down


def _scale_activation_op(
    act_op: OpSpec, *, rows: int, local_w: int, old_rows: int, old_w: int,
    dep_id: str, width: int,
) -> OpSpec:
    ratio = (float(rows) * float(local_w)) / float(max(1, old_rows * old_w))
    meta = {**dict(act_op.metadata), "local_tp_applied": True, "local_tp_width": int(width)}
    return replace(
        act_op, deps=(dep_id,),
        local_shape=(int(rows), int(local_w)),
        padded_shape=(align_up(int(rows), ALIGN_ELEMENT), align_up(int(local_w), ALIGN_ELEMENT)),
        launch_shape=(int(rows), int(local_w)),
        semantic_shape=(int(rows), int(local_w)),
        vector_ops=float(act_op.vector_ops) * ratio,
        hbm_bytes=float(act_op.hbm_bytes) * ratio,
        mte2_bytes=float(act_op.mte2_bytes) * ratio,
        mte3_bytes=float(act_op.mte3_bytes) * ratio,
        metadata=tuple(sorted(meta.items(), key=lambda pair: pair[0])),
    )


def _rewrite_paired_ffn(
    gate_up: OpSpec, act_op: OpSpec, pre_quant: OpSpec | None, down: OpSpec,
    *, role: str, down_role: str, factor: int, stride: int,
    group: Sequence[int], total_tok: int, pad_tok: int,
    quant: Any, op_scope: str,
) -> tuple[OpSpec, ...]:
    """Emit one Megatron-style FFN block: AllGather -> gate_up -> act -> [quant] -> down -> RS."""
    g_meta = dict(gate_up.metadata)
    d_meta = dict(down.metadata)
    m, _, hidden = (int(gate_up.local_shape[0]), int(gate_up.local_shape[1]),
                    int(gate_up.local_shape[2]))
    inter_w = int(down.local_shape[2])
    local_w = _local_output_width(inter_w, factor)
    act_dt = _dtype(g_meta, "activation_format", gate_up.dtype or FALLBACK_ACTIVATION)
    out_dt = _dtype(d_meta, "output_format", FALLBACK_OUTPUT)
    gathered_tok = factor * pad_tok
    gather = _collective(
        gate_up, suffix="gather", family="allgather", recipe="ring_allgather",
        payload=float((factor - 1) * pad_tok * hidden) * dtype_bytes(act_dt),
        shape=(gathered_tok, hidden), role=role, group=group, width=factor,
        deps=gate_up.deps, scope=op_scope, dtype=act_dt, stride=stride,
    )
    g_meta["local_tp_partition"] = ROLE_COLUMN
    gate_gemm = _rewrite_matrix(
        gate_up, gemm_id=f"{gate_up.op_id}:local_tp:gemm",
        rows=gathered_tok, cols=2 * local_w, reduced=hidden,
        role=role, width=factor, dep_ids=(gather.op_id,),
        weight_ratio=float(local_w) / float(max(1, inter_w)),
        quant=quant, extra_metadata=g_meta,
    )
    act_old_rows = max(1, int(dict(act_op.metadata).get("local_tp_template_rows") or m))
    scaled_act = _scale_activation_op(
        act_op, rows=gathered_tok, local_w=local_w, old_rows=act_old_rows, old_w=inter_w,
        dep_id=gate_gemm.op_id, width=factor,
    )
    emitted: list[OpSpec] = [gather, gate_gemm, scaled_act]
    down_dep = scaled_act.op_id
    scale_ratio: float | None = None
    if pre_quant is not None:
        requant = _received_quantizer(
            pre_quant, rows=gathered_tok, local_k=local_w, full_k=inter_w,
            exchange_id=scaled_act.op_id, width=factor,
        )
        emitted.append(requant)
        down_dep = requant.op_id
        pq_old_rows = max(1, int(dict(pre_quant.metadata).get("local_tp_template_rows") or m))
        scale_ratio = (float(gathered_tok) / float(pq_old_rows)
                       * float(local_w) / float(max(1, inter_w)))
    d_meta["local_tp_partition"] = ROLE_ROW
    down_gemm = _rewrite_matrix(
        down, gemm_id=f"{down.op_id}:local_tp:gemm",
        rows=gathered_tok, cols=hidden, reduced=local_w,
        role=down_role, width=factor, dep_ids=(down_dep,),
        weight_ratio=float(local_w) / float(max(1, inter_w)),
        quant=quant, extra_metadata=d_meta, scale_ratio=scale_ratio,
    )
    restore = _collective(
        down, suffix="reduce_scatter", family="reduce_scatter",
        recipe="ring_reduce_scatter",
        payload=float((factor - 1) * pad_tok * hidden) * dtype_bytes(out_dt),
        shape=(pad_tok, hidden), role=down_role, group=group, width=factor,
        deps=(down_gemm.op_id,), scope=op_scope, dtype=out_dt,
        public_id=down.op_id, stride=stride,
    )
    emitted.extend((down_gemm, restore))
    return tuple(emitted)


def _logits_tp_gather(
    op: OpSpec, by_id: Mapping[str, OpSpec], users: Mapping[str, list[str]],
) -> OpSpec | None:
    """Trailing TP ``allgather`` replaced when DP-axis LM-head TP owns full vocab."""
    for uid in users.get(op.op_id) or ():
        cand = by_id.get(uid)
        if (cand is not None and cand.family == "allgather"
                and cand.op_id == f"{op.op_id}_gather"
                and cand.deps == (op.op_id,)
                and dict(cand.metadata).get("group_axis") == "tp"):
            return cand
    return None


def _rewrite_single_projection(
    op: OpSpec, *, role: str, partition: str, factor: int, stride: int,
    group: Sequence[int], total_tok: int, pad_tok: int,
    quant: Any, op_scope: str, by_id: Mapping[str, OpSpec],
    users: Mapping[str, list[str]], metadata: dict[str, Any],
    model: Any = None,
) -> tuple[OpSpec, ...]:
    m, n, k = (int(op.local_shape[0]), int(op.local_shape[1]), int(op.local_shape[2]))
    act = _dtype(metadata, "activation_format", op.dtype or FALLBACK_ACTIVATION)
    out = _dtype(metadata, "output_format", FALLBACK_OUTPUT)
    metadata["local_tp_partition"] = partition
    gemm_id = f"{op.op_id}:local_tp:gemm"
    groups = max(1, int(metadata.get("matrix_groups", 1))) if partition == ROLE_GROUP_A else 1
    if partition == ROLE_GROUP_A:
        lg = groups // factor
        ex = _collective(op, suffix="exchange", family="alltoall", recipe="alltoall",
                         payload=float((factor - 1) * pad_tok * lg * k) * dtype_bytes(act),
                         shape=(total_tok * lg, k), role=role, group=group,
                         width=factor, deps=op.deps, scope=op_scope, dtype=act, stride=stride)
        gemm = _rewrite_matrix(op, gemm_id=op.op_id, rows=total_tok * lg, cols=n, reduced=k,
                               role=role, width=factor, dep_ids=(ex.op_id,),
                               weight_ratio=1 / float(factor), quant=quant, groups=lg, extra_metadata=metadata)
        return (ex, gemm)
    if partition == ROLE_GROUP_B:
        gemm = _rewrite_matrix(op, gemm_id=gemm_id, rows=total_tok, cols=n, reduced=k // factor,
                               role=role, width=factor, dep_ids=op.deps, weight_ratio=1 / float(factor),
                               quant=quant, groups=1, extra_metadata=metadata)
        restore = _collective(op, suffix="reduce_scatter", family="reduce_scatter", recipe="ring_reduce_scatter",
                              payload=float((factor - 1) * pad_tok * n) * dtype_bytes(out), shape=(m, n),
                              role=role, group=group, width=factor, deps=(gemm.op_id,), scope=op_scope,
                              dtype=out, public_id=op.op_id, stride=stride)
        return (gemm, restore)
    if partition == ROLE_COLUMN:
        full_n = n
        is_logits = role in {ROLE_LOGITS, ROLE_DRAFT_LOGITS}
        is_draft = str(metadata.get("layer_scope") or "") == "draft"
        if is_logits:
            if role == ROLE_LOGITS and not is_draft and model is not None and int(getattr(model, "vocab_size", 0) or 0) > 0:
                full_n = int(model.vocab_size)
            elif stride > 1:
                tp_gather = _logits_tp_gather(op, by_id, users)
                padded_v = int(metadata.get("vocab_padded_size") or 0)
                if padded_v > 0:
                    full_n = padded_v
                elif tp_gather is not None and len(tp_gather.local_shape) >= 2 and int(tp_gather.local_shape[1]) > n:
                    full_n = int(tp_gather.local_shape[1])
                elif not _is_globally_replicated(op, role, metadata, SimpleNamespace(metadata=()), model):
                    full_n = n * stride
        local_n = _packed_local_width(metadata, factor, full_n, quant, role=role)
        remote_n = _remote_exchange_width(
            metadata, factor, local_n, full_n, quant, is_logits=is_logits,
        )
        col_rows = (
            factor * pad_tok if is_logits
            else total_tok
        )
        restore_n = factor * local_n if is_logits else full_n
        if is_logits and (restore_n != full_n or "vocab_padded_size" in metadata):
            metadata["vocab_padded_size"] = restore_n
            metadata["vocab_cols_per_rank"] = local_n
        _localize_packed_slices(metadata, factor, quant, rows=col_rows)
        gather = _collective(op, suffix="gather", family="allgather", recipe="ring_allgather",
                             payload=float((factor - 1) * pad_tok * k) * dtype_bytes(act), shape=(col_rows, k),
                             role=role, group=group, width=factor, deps=op.deps, scope=op_scope, dtype=act,
                             stride=stride)
        gemm = _rewrite_matrix(op, gemm_id=gemm_id, rows=col_rows, cols=local_n, reduced=k,
                               role=role, width=factor, dep_ids=(gather.op_id,),
                               weight_ratio=float(local_n) / float(max(1, n)), quant=quant, extra_metadata=metadata)
        restore = _collective(op, suffix="exchange", family="alltoall", recipe="alltoall",
                              payload=float(pad_tok * remote_n) * dtype_bytes(out),
                              shape=(m, restore_n),
                              role=role, group=group, width=factor, deps=(gemm.op_id,), scope=op_scope,
                              dtype=out, public_id=op.op_id, stride=stride)
        return (gather, gemm, restore)
    local_k = _local_dimension(k, factor, quant)
    source = op.deps
    pre = _row_pre_quantizer(op, by_id, users)
    exchange_dtype = act
    if pre is not None:
        exchange_dtype = _dtype(dict(pre.metadata), "input_format", act)
        source = pre.deps
    ex = _collective(op, suffix="exchange", family="alltoall", recipe="alltoall",
                     payload=float((factor - 1) * pad_tok * local_k)
                             * dtype_bytes(exchange_dtype),
                     shape=(total_tok, local_k), role=role, group=group, width=factor,
                     deps=source, scope=op_scope, dtype=exchange_dtype, stride=stride)
    out_ops: list[OpSpec] = [ex]
    gemm_dep, scale_ratio = ex.op_id, None
    if pre is not None:
        requantized = _received_quantizer(
            pre, rows=total_tok, local_k=local_k, full_k=k,
            exchange_id=ex.op_id, width=factor)
        out_ops.append(requantized)
        gemm_dep = requantized.op_id
        scale_ratio = (float(total_tok) / float(max(1, m))
                       * float(local_k) / float(max(1, k)))
    gemm = _rewrite_matrix(op, gemm_id=gemm_id, rows=total_tok, cols=n, reduced=local_k,
                           role=role, width=factor, dep_ids=(gemm_dep,),
                           weight_ratio=float(local_k) / float(max(1, k)),
                           quant=quant, extra_metadata=metadata,
                           scale_ratio=scale_ratio)
    restore = _collective(op, suffix="reduce_scatter", family="reduce_scatter", recipe="ring_reduce_scatter",
                          payload=float((factor - 1) * pad_tok * n) * dtype_bytes(out), shape=(m, n),
                          role=role, group=group, width=factor, deps=(gemm.op_id,), scope=op_scope,
                          dtype=out, public_id=op.op_id, stride=stride)
    out_ops.extend((gemm, restore))
    return tuple(out_ops)


def lower_local_projection_tp(
    ops: Sequence[OpSpec], *, model: Any, runtime: Any, machine: Any,
    replica: int, cohort_views: Mapping[int, Sequence[Any]], scope: str,
    peer_rows: Mapping[tuple[int, str], Mapping[int, int]] | None = None,
) -> tuple[OpSpec, ...]:
    """Rewrite declared dense projection regions for node-local TP."""
    if not runtime.parallel.has_local_tp:
        return tuple(ops)

    validate_local_projection_tp(SimpleNamespace(model=model, runtime=runtime, machine=machine))
    by_id = {op.op_id: op for op in ops}
    users: dict[str, list[str]] = {}
    for consumer in ops:
        for dependency in consumer.deps:
            users.setdefault(dependency, []).append(consumer.op_id)
    paired_blocks: dict[str, tuple[OpSpec, OpSpec | None, OpSpec]] = {}
    paired_members: set[str] = set()
    retire: set[str] = set()
    dep_aliases: dict[str, str] = {}
    for candidate in ops:
        candidate_meta = dict(candidate.metadata)
        candidate_role = _role(candidate_meta)
        if not candidate_role or candidate_meta.get("local_tp_applied"):
            continue
        factor, stride = _effective_factor(candidate, candidate_role, candidate_meta, runtime, model)
        if factor <= 1:
            continue
        if candidate_role in {ROLE_LOGITS, ROLE_DRAFT_LOGITS} and stride > 1:
            tp_gather = _logits_tp_gather(candidate, by_id, users)
            if tp_gather is not None:
                retire.add(tp_gather.op_id)
                dep_aliases[tp_gather.op_id] = candidate.op_id
        matched = _match_paired_ffn(candidate, candidate_role, by_id, users)
        if matched is not None:
            paired_blocks[candidate.op_id] = matched
            act_op, pre_q, down_op = matched
            paired_members.update((act_op.op_id, down_op.op_id))
            if pre_q is not None:
                paired_members.add(pre_q.op_id)
            continue
        if (candidate.op_id not in paired_members
                and _partition_kind(candidate_role, candidate_meta) == ROLE_ROW):
            pre = _row_pre_quantizer(candidate, by_id, users)
            if pre is not None:
                retire.add(pre.op_id)
    rewritten: list[OpSpec] = []
    for op in ops:
        if op.op_id in retire or op.op_id in paired_members:
            continue
        if dep_aliases and any(d in dep_aliases for d in op.deps):
            op = replace(op, deps=tuple(dep_aliases.get(d, d) for d in op.deps))
        metadata = dict(op.metadata)
        role = _role(metadata)
        if not role or metadata.get("local_tp_applied"):
            rewritten.append(op)
            continue
        factor, stride = _effective_factor(op, role, metadata, runtime, model)
        if factor <= 1:
            rewritten.append(op)
            continue
        partition = _partition_kind(role, metadata)
        group = _groups_for(machine, runtime, replica, op.stage, factor, stride=stride)
        if op.source_boundary or metadata.get("implementation"):
            impl = str(metadata.get("implementation", "") or metadata.get("kernel", "source"))
            adapter = RETARGET_ADAPTERS.get(impl)
            if adapter is None:
                raise ModelingError(f"local_tp_source_placement_unsupported:{impl}")
            op = adapter(op, width=factor, role=role, group=group, partition=partition)
            metadata = dict(op.metadata)
        quant = _quant_for_op(model, metadata)
        m, k = int(op.local_shape[0]), int(op.local_shape[2])
        op_scope = str(metadata.get("layer_scope") or scope or "target")
        op_peers = (peer_rows or {}).get((int(op.stage), op.op_id))
        groups = max(1, int(metadata.get("matrix_groups", 1))) if partition == ROLE_GROUP_A else 1
        if (partition == ROLE_GROUP_A and groups % factor != 0) or (
                partition == ROLE_GROUP_B and k % factor != 0):
            raise ModelingError("local_tp_group_geometry_indivisible")
        counts = _row_counts(cohort_views, group, m // groups, op_peers)
        total_tok, pad_tok = sum(counts), max(counts, default=0)
        if total_tok <= 0:
            rewritten.append(_empty_passthrough(op, metadata, factor))
            if op.op_id in paired_blocks:
                act_op, pre_q, down_op = paired_blocks[op.op_id]
                rewritten.append(_empty_passthrough(down_op, dict(down_op.metadata), factor))
            continue
        if op.op_id in paired_blocks:
            act_op, pre_q, down_op = paired_blocks[op.op_id]
            down_role = (ROLE_DENSE_MLP_DOWN if role == ROLE_DENSE_MLP_UP
                         else ROLE_SHARED_EXPERT_DOWN)
            rewritten.extend(_rewrite_paired_ffn(
                op, act_op, pre_q, down_op, role=role, down_role=down_role,
                factor=factor, stride=stride, group=group,
                total_tok=total_tok, pad_tok=pad_tok, quant=quant,
                op_scope=op_scope,
            ))
            continue
        rewritten.extend(_rewrite_single_projection(
            op, role=role, partition=partition, factor=factor, stride=stride,
            group=group, total_tok=total_tok, pad_tok=pad_tok, quant=quant,
            op_scope=op_scope, by_id=by_id, users=users, metadata=metadata,
            model=model,
        ))
    return tuple(rewritten)


def _synth_idle_block(
    tmpl: OpSpec, paired: tuple[OpSpec, OpSpec | None, OpSpec] | None,
    dep_ids: tuple[str, ...],
) -> list[OpSpec]:
    """Build zero-row template ops for an idle peer, keeping paired FFN structure intact."""
    z_local = (0, int(tmpl.local_shape[1]), int(tmpl.local_shape[2]))
    z_pad = (0, int(tmpl.padded_shape[1]), int(tmpl.padded_shape[2]))
    z_launch = (0, int(tmpl.launch_shape[1]), int(tmpl.launch_shape[2]))
    gate = replace(tmpl, requests=(), deps=dep_ids, local_shape=z_local, padded_shape=z_pad,
                   launch_shape=z_launch, flops=0.0, matrix_flops=(), vector_ops=0.0,
                   hbm_bytes=0.0, mte2_bytes=0.0, mte3_bytes=0.0)
    if paired is None:
        return [gate]
    act_tmpl, pq_tmpl, down_tmpl = paired
    act_rows = max(1, int(act_tmpl.local_shape[0]))
    act_meta = tuple(sorted({
        **dict(act_tmpl.metadata), "local_tp_template_rows": act_rows,
    }.items(), key=lambda pair: pair[0]))
    act_shape = (0, *tuple(int(d) for d in act_tmpl.local_shape[1:]))
    synth_act = replace(act_tmpl, requests=(), deps=(gate.op_id,),
                        local_shape=act_shape, padded_shape=act_shape,
                        launch_shape=act_shape, semantic_shape=act_shape,
                        metadata=act_meta)
    chain: list[OpSpec] = [gate, synth_act]
    down_dep = synth_act.op_id
    if pq_tmpl is not None:
        pq_rows = max(1, int(pq_tmpl.local_shape[0]))
        pq_meta = tuple(sorted({
            **dict(pq_tmpl.metadata), "local_tp_template_rows": pq_rows,
        }.items(), key=lambda pair: pair[0]))
        pq_shape = (0, *tuple(int(d) for d in pq_tmpl.local_shape[1:]))
        synth_pq = replace(pq_tmpl, requests=(), deps=(synth_act.op_id,),
                           local_shape=pq_shape, padded_shape=pq_shape,
                           launch_shape=pq_shape, semantic_shape=pq_shape,
                           metadata=pq_meta)
        chain.append(synth_pq)
        down_dep = synth_pq.op_id
    dz_local = (0, int(down_tmpl.local_shape[1]), int(down_tmpl.local_shape[2]))
    dz_pad = (0, int(down_tmpl.padded_shape[1]), int(down_tmpl.padded_shape[2]))
    dz_launch = (0, int(down_tmpl.launch_shape[1]), int(down_tmpl.launch_shape[2]))
    synth_down = replace(down_tmpl, requests=(), deps=(down_dep,),
                         local_shape=dz_local, padded_shape=dz_pad, launch_shape=dz_launch,
                         flops=0.0, matrix_flops=(), vector_ops=0.0,
                         hbm_bytes=0.0, mte2_bytes=0.0, mte3_bytes=0.0)
    chain.append(synth_down)
    return chain


def lower_local_projection_tp_cohort(
    graphs: Mapping[int, Sequence[OpSpec]], *, model: Any, runtime: Any, machine: Any,
) -> dict[int, tuple[OpSpec, ...]]:
    """Cohort-level local-TP rewrite using actual per-rank operator rows and idle synthesis."""
    if not runtime.parallel.has_local_tp or not graphs:
        return {int(r): tuple(ops) for r, ops in graphs.items()}
    from mindie_sim.prediction.topology import Topology

    validate_local_projection_tp(SimpleNamespace(model=model, runtime=runtime, machine=machine))
    topo = Topology(machine, runtime.parallel)
    identifiers = [tuple(op.op_id for op in ops) for ops in graphs.values()]
    homogeneous = all(ids == identifiers[0] for ids in identifiers)
    ordered_occ: list[tuple[int, str]] = []
    template_by_occ: dict[tuple[int, str], tuple[OpSpec, frozenset[str], int, int, Any]] = {}
    paired_down_keys: set[tuple[int, str]] = set()
    peer_rows: dict[tuple[int, str], dict[int, int]] = {}
    op_maps = {int(r): {op.op_id: op for op in ops} for r, ops in graphs.items()}
    for r, ops in sorted(graphs.items()):
        seq = tuple(ops)
        r_by_id = op_maps[int(r)]
        r_users: dict[str, list[str]] = {}
        for consumer in seq:
            for dep in consumer.deps:
                r_users.setdefault(dep, []).append(consumer.op_id)
        for idx, op in enumerate(seq):
            meta = dict(op.metadata)
            role = _role(meta)
            if not role or meta.get("local_tp_applied"):
                continue
            factor, stride = _effective_factor(op, role, meta, runtime, model)
            if factor <= 1:
                continue
            key = (int(op.stage), op.op_id)
            if key not in template_by_occ:
                paired = _match_paired_ffn(op, role, r_by_id, r_users)
                if paired is not None:
                    paired_down_keys.add((int(op.stage), paired[2].op_id))
                ordered_occ.append(key)
                template_by_occ[key] = (
                    op, frozenset() if homogeneous else
                    frozenset(x.op_id for x in seq[idx + 1:]),
                    factor, stride, paired)
            grp_div = max(1, int(meta.get("matrix_groups", 1))) if _partition_kind(role, meta) == ROLE_GROUP_A else 1
            tok_m = int(op.local_shape[0]) // grp_div
            for rk in topo.ranks(int(r), int(op.stage)):
                peer_rows.setdefault(key, {})[int(rk)] = tok_m
    ext_roots = tuple(dict.fromkeys(d for r0, ops in sorted(graphs.items()) for item in ops
                                    for d in item.deps if d not in op_maps[int(r0)]))
    augmented: dict[int, list[OpSpec]] = {int(r): list(ops) for r, ops in graphs.items()}
    prev_synth: dict[tuple[int, int], str] = {}
    for stage, op_id in ordered_occ:
        if (stage, op_id) in paired_down_keys:
            continue
        tmpl, after_ids, factor, stride, paired = template_by_occ[(stage, op_id)]
        for r in sorted(augmented):
            if op_id in op_maps[r]:
                prev_synth[(r, stage)] = paired[2].op_id if paired is not None else op_id
                continue
            group = _groups_for(machine, runtime, r, stage, factor, stride=stride)
            for rk in topo.ranks(r, stage):
                peer_rows.setdefault((stage, op_id), {}).setdefault(int(rk), 0)
                if paired is not None:
                    peer_rows.setdefault((stage, paired[2].op_id), {}).setdefault(int(rk), 0)
            if sum(peer_rows[(stage, op_id)].get(int(rk), 0) for rk in group) <= 0:
                continue
            existing_ids = {item.op_id for item in augmented[r]}
            dep_ids = tuple(d for d in tmpl.deps if d in existing_ids) or (
                (prev_synth[(r, stage)],) if (r, stage) in prev_synth
                else ("iteration_input",) if "iteration_input" in existing_ids else ext_roots[:1])
            chain = _synth_idle_block(tmpl, paired, dep_ids)
            min_idx = max((i + 1 for i, x in enumerate(augmented[r]) if x.op_id in dep_ids), default=0)
            insert_at = next((i for i in range(min_idx, len(augmented[r]))
                              if augmented[r][i].op_id in after_ids), len(augmented[r]))
            for offset, synth in enumerate(chain):
                augmented[r].insert(insert_at + offset, synth)
                op_maps[r][synth.op_id] = synth
            prev_synth[(r, stage)] = chain[-1].op_id
    return {r: lower_local_projection_tp(ops, model=model, runtime=runtime, machine=machine,
                                         replica=r, cohort_views={}, scope="target", peer_rows=peer_rows)
            for r, ops in augmented.items()}
