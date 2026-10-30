"""Physical weight representation and sharding.

[Role] Convert logical matrix terms into per-rank resident bytes.
[Invariants] Packing, scales and expert ownership are conserved separately.
[Timeouts & Locks] Pure arithmetic.
[Side-Effect Policy] No I/O.
"""
from dataclasses import dataclass
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.catalog.types import QuantSpec


@dataclass(frozen=True)
class WeightTerm:
    name: str
    rows: int
    cols: int
    category: str
    shard: str = "replicated"
    count: int = 1
    groups: int = 0
    units: int = 0
    unit_width: int = 0
    shared_owner: bool = False
    source_owned: bool = True
    quant: QuantSpec | None = None
    runtime_dtype: str = ""
    runtime_quant: QuantSpec | None = None

    @property
    def logical_parameters(self) -> float:
        return float(self.rows) * float(self.cols) * float(self.count)


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _local_groups(groups: int, tp: int) -> tuple[int, bool]:
    if groups <= 0:
        return 0, False
    if groups % tp == 0:
        return groups // tp, True
    return groups, False


def _sharded_shape(
    term: WeightTerm, parallel: ParallelSpec, node_ranks: int,
) -> tuple[int, int, int]:
    rows, cols, count = int(term.rows), int(term.cols), int(term.count)
    tp, ep = max(1, parallel.tp), max(1, parallel.ep)
    etp = max(1, parallel.etp)
    if term.shard == "replicated":
        return rows, cols, count
    if term.shard == "out_tp":
        return _ceil_div(rows, tp), cols, count
    if term.shard == "in_tp":
        return rows, _ceil_div(cols, tp), count
    if term.shard == "kv_out":
        if term.units <= 0 or term.unit_width <= 0:
            raise ModelingError(f"catalog_kv_shard_granularity_missing:{term.name}")
        local_units = max(1, _ceil_div(term.units, tp))
        return local_units * term.unit_width, cols, count
    if term.shard == "expert_gate_up":
        return _ceil_div(rows, etp), cols, _ceil_div(count, ep)
    if term.shard == "expert_down":
        return rows, _ceil_div(cols, etp), _ceil_div(count, ep)
    if term.shard == "dense_up":
        return _ceil_div(rows, tp), cols, count
    if term.shard == "dense_down":
        return rows, _ceil_div(cols, tp), count
    if term.shard == "embedding":
        return (_ceil_div(rows, tp) if rows % tp == 0 else rows), cols, count
    if term.shard == "node":
        return _ceil_div(rows, max(1, node_ranks)), cols, count
    if term.shard == "group_out":
        local, _ = _local_groups(term.groups, tp)
        if term.groups <= 0:
            raise ModelingError("catalog_group_term_missing_groups")
        return max(1, rows * local // term.groups), cols, count
    if term.shard == "group_in":
        local, _ = _local_groups(term.groups, tp)
        if term.groups <= 0:
            raise ModelingError("catalog_group_term_missing_groups")
        return rows, max(1, cols * local // term.groups), count
    raise ModelingError(f"catalog_weight_shard_unsupported:{term.shard}")


def term_bytes(
    term: WeightTerm, quant: QuantSpec, parallel: ParallelSpec,
    node_ranks: int = 1, *, source: bool = False,
) -> float:
    if source and not term.source_owned:
        return 0.0
    rows, cols, count = _sharded_shape(term, parallel, node_ranks)
    if not source and term.runtime_dtype:
        return float(rows * cols * count) * dtype_bytes(term.runtime_dtype)
    representation = (term.runtime_quant if not source else None) or term.quant or quant
    per_matrix = representation.matrix_bytes(rows, cols, term.category, source=source)
    return float(per_matrix) * float(count)


def term_payload_bytes(
    term: WeightTerm, quant: QuantSpec, parallel: ParallelSpec,
    node_ranks: int = 1, *, source: bool = False,
) -> float:
    """Packed payload bytes of one term (quant scales/zero-points excluded)."""
    if source and not term.source_owned:
        return 0.0
    rows, cols, count = _sharded_shape(term, parallel, node_ranks)
    if not source and term.runtime_dtype:
        return float(rows * cols * count) * dtype_bytes(term.runtime_dtype)
    representation = (term.runtime_quant if not source else None) or term.quant or quant
    per_matrix = representation.matrix_payload_bytes(
        rows, cols, term.category, source=source,
    )
    return float(per_matrix) * float(count)


def _t(name: str, rows: int, cols: int, category: str, shard: str = "replicated",
       count: int = 1, groups: int = 0, units: int = 0,
       unit_width: int = 0) -> WeightTerm:
    if rows <= 0 or cols <= 0:
        raise ModelingError(f"catalog_weight_term_invalid:{name}:{rows}x{cols}")
    return WeightTerm(name, int(rows), int(cols), category, shard, int(count),
                      int(groups), int(units), int(unit_width))
