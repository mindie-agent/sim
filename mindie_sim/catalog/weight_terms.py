"""Physical weight representation and sharding.

[Role] Convert logical matrix terms into per-rank resident bytes.
[Invariants] Packing, scales and expert ownership are conserved separately.
[Timeouts & Locks] Pure arithmetic.
[Side-Effect Policy] No I/O.
"""
from dataclasses import dataclass
import math
from mindie_sim.schema.common import ModelingError, align_up, dtype_bytes
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.catalog.types import QuantSpec

ZERO = 0
ONE = 1
FIRST = 0
NODE_RANKS_DEFAULT = 1
#: Upstream vLLM / vLLM-Ascend ``DEFAULT_VOCAB_PADDING_SIZE`` for
#: ``VocabParallelEmbedding`` and ``ParallelLMHead``.
VOCAB_PADDING_SIZE = 64


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
    #: Declared dense-subgraph placement role and ownership scope.
    placement_role: str = ""
    placement_scope: str = "target"
    #: Already-resolved physical per-rank multiplicity (expert slots).  It is
    #: never divided by EP again.
    resident_count: int | None = None
    #: Runtime residency ownership.  A source-only checkpoint tensor is not a
    #: resident copy; a runtime-only prepared copy is not a logical parameter.
    runtime_owned: bool = True
    #: Concrete logical expert IDs resident on this rank (for mixed-quant
    #: per-EP-rank billing via ``QuantSpec.expert_subset_bytes``).
    expert_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.expert_ids and not isinstance(self.expert_ids, tuple):
            object.__setattr__(self, "expert_ids", tuple(int(eid) for eid in self.expert_ids))
        if (self.expert_ids and self.resident_count is not None
                and len(self.expert_ids) != int(self.resident_count)):
            raise ModelingError(
                f"catalog_expert_ids_count_mismatch:{len(self.expert_ids)}!={int(self.resident_count)}"
            )

    @property
    def logical_parameters(self) -> int:
        return int(self.rows) * int(self.cols) * int(self.count)


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def vocab_parallel_rows(rows: int, width: int, *, source: bool = False) -> int:
    """Per-rank vocabulary rows under vocab-parallel sharding.

    Resident shards (``source=False``, ``width > 1``) pad the vocabulary axis
    to ``lcm(VOCAB_PADDING_SIZE, width)`` before dividing by ``width`` so
    non-divisible vocabularies shard cleanly instead of falling back to full
    replication.  Unsharded (``width <= 1``) and checkpoint-source
    (``source=True``) counts never add runtime padding rows.
    """
    rows_i, width_i = int(rows), max(1, int(width))
    if width_i <= 1:
        return rows_i
    if source:
        return _ceil_div(rows_i, width_i)
    alignment = math.lcm(VOCAB_PADDING_SIZE, width_i)
    return int(align_up(rows_i, alignment)) // width_i


def _local_groups(groups: int, tp: int) -> tuple[int, bool]:
    if groups <= 0:
        return 0, False
    if groups % tp == 0:
        return groups // tp, True
    return groups, False


def _sharded_shape(
    term: WeightTerm, parallel: ParallelSpec, node_ranks: int, *,
    source: bool = False,
) -> tuple[int, int, int]:
    from mindie_sim.catalog.projection_placement import (
        SHARDED_ROLES, _term_role, placed_weight_shape, role_width,
    )
    role = _term_role(term)
    width = role_width(role, parallel) if role in SHARDED_ROLES else ONE
    if width > ONE or term.resident_count is not None or term.expert_ids:
        rows, cols, count = placed_weight_shape(
            term, parallel, node_ranks, source=source,
        )
        if term.expert_ids:
            count = len(term.expert_ids)
        elif term.resident_count is not None and term.shard not in {
                "expert_gate_up", "expert_down"}:
            count = int(term.resident_count)
        return rows, cols, count
    return _legacy_sharded_shape(term, parallel, node_ranks, source=source)


def _legacy_sharded_shape(
    term: WeightTerm, parallel: ParallelSpec, node_ranks: int, *,
    source: bool = False,
) -> tuple[int, int, int]:
    rows, cols, count = int(term.rows), int(term.cols), int(term.count)
    tp, ep = max(1, parallel.tp), max(1, parallel.ep)
    pcp = max(1, int(getattr(parallel, "pcp", 1) or 1))
    etp = max(1, parallel.etp)
    # KDA under PCP shards heads (and head-indexed weights) across tp * pcp.
    kda_tp = (tp * pcp) if (pcp > 1 and term.name.startswith("kda_")) else tp
    if term.shard == "replicated":
        return rows, cols, count
    if term.shard == "out_tp":
        if term.category == "lm_head" or term.placement_role in {"logits", "draft_logits"}:
            return vocab_parallel_rows(rows, tp, source=source), cols, count
        return _ceil_div(rows, kda_tp), cols, count
    if term.shard == "in_tp":
        return rows, _ceil_div(cols, kda_tp), count
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
        return vocab_parallel_rows(rows, tp, source=source), cols, count
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
    node_ranks: int = NODE_RANKS_DEFAULT, *, source: bool = False,
) -> float:
    if source and not term.source_owned:
        return 0.0
    if not source and not term.runtime_owned:
        return 0.0
    rows, cols, count = _sharded_shape(term, parallel, node_ranks, source=source)
    if not source and term.runtime_dtype:
        return float(rows * cols * count) * dtype_bytes(term.runtime_dtype)
    representation = (term.runtime_quant if not source else None) or term.quant or quant
    if term.expert_ids:
        return float(representation.expert_subset_bytes(
            rows, cols, term.category, term.expert_ids,
            source=source, include_scales=True,
        ))
    per_matrix = representation.matrix_bytes(rows, cols, term.category, source=source)
    return float(per_matrix) * float(count)


def term_payload_bytes(
    term: WeightTerm, quant: QuantSpec, parallel: ParallelSpec,
    node_ranks: int = NODE_RANKS_DEFAULT, *, source: bool = False,
) -> float:
    """Packed payload bytes of one term (quant scales/zero-points excluded)."""
    if source and not term.source_owned:
        return 0.0
    if not source and not term.runtime_owned:
        return 0.0
    rows, cols, count = _sharded_shape(term, parallel, node_ranks, source=source)
    if not source and term.runtime_dtype:
        return float(rows * cols * count) * dtype_bytes(term.runtime_dtype)
    representation = (term.runtime_quant if not source else None) or term.quant or quant
    if term.expert_ids:
        return float(representation.expert_subset_bytes(
            rows, cols, term.category, term.expert_ids,
            source=source, include_scales=False,
        ))
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
