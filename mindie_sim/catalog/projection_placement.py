"""Shared dense projection roles, packed slices and per-rank placed shapes.

[Role]
  One registry describes which semantic matrix subgraphs participate in
  node-local tensor parallelism and how their logical extents become physical
  per-rank extents.  Both the graph rewrite and the weight ledger consume it, so
  residency and communication cannot drift apart.

[Invariants]
  * Only declared matrix roles are sharded; recurrent state, convolution
    weights, norms, gates, embeddings, FFN weights and ordinary attention Q/K/V
    projections remain replicated.
  * Column roles shard output features, row roles shard input features and a
    grouped output pair shards whole groups.  The sharded dimension is padded to
    the local width and the declared quant granule.
  * ``weight_terms`` list the exact semantic terms represented by a placement;
    packed slices preserve semantic boundaries, byte offsets and quant groups.
  * A width of one reproduces every historical shape and byte count.

[Timeouts & Locks]
  Pure metadata arithmetic; no I/O or locks.

[Side-Effect Policy]
  Read-only immutable records.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from typing import Any, Iterable, Sequence

from mindie_sim.schema.common import ModelingError, align_up


ZERO = 0
ONE = 1
ZERO_FLOAT = 0.0
ROLE_ATTENTION_OUTPUT = "attention_output"
ROLE_LINEAR_INPUT = "linear_input"
ROLE_LINEAR_GATE = "linear_gate"
ROLE_LINEAR_OUTPUT = "linear_output"
ROLE_LOGITS = "logits"
ROLE_DRAFT_LOGITS = "draft_logits"
ROLE_TARGET_HEAD_ALIAS = "target_lm_head_alias"
PARTITION_COLUMN = "column"
PARTITION_ROW = "row"
PARTITION_GROUP_PAIR = "group_pair"
ATTENTION_ROLES = frozenset({ROLE_ATTENTION_OUTPUT})
LINEAR_ROLES = frozenset({ROLE_LINEAR_INPUT, ROLE_LINEAR_GATE, ROLE_LINEAR_OUTPUT})
LOGIT_ROLES = frozenset({ROLE_LOGITS, ROLE_DRAFT_LOGITS})
SHARDED_ROLES = ATTENTION_ROLES | LINEAR_ROLES | LOGIT_ROLES
_GROUP_PAIR_TERMS = frozenset({"wo_a", "wo_b", "draft_wo_a", "draft_wo_b"})
_ATTENTION_OUTPUT_TERMS = frozenset({"o_proj", "output", "draft_o"})
_LINEAR_INPUT_TERMS = frozenset({
    "kda_q", "kda_k", "kda_v", "kda_b", "kda_f_a", "kda_g_a", "kda_g",
    "gdn_qkvz", "gdn_ba", "linear_qkvbfg_a", "linear_q", "linear_k",
    "linear_v", "linear_beta", "linear_decay_a", "linear_gate_a",
    "linear_gate", "in_proj_qkv", "in_proj_qkvz", "kda_qkv", "kda_qkvz",
})
_LINEAR_GATE_TERMS = frozenset({
    "kda_f_b", "kda_g_b", "linear_decay_b", "linear_gate_b",
})
_LINEAR_OUTPUT_TERMS = frozenset({"kda_o", "gdn_o", "linear_output"})
_EMBEDDING_TERMS = frozenset({"embedding", "markov_embed"})
_PACKED_GRANULE_KEY = "packed_slices"
_PACKED_SLICES_KEY = "packed_slices"


@dataclass(frozen=True)
class ProjectionSlice:
    """One semantic slice of a packed or unfused projection matrix."""

    semantic_role: str
    byte_offset: int
    logical_extent: int
    physical_extent: int
    quant_category: str
    quant_group_boundaries: tuple[int, ...] = ()
    name: str = ""


@dataclass(frozen=True)
class ProjectionPlacement:
    """Physical placement of one semantic dense projection region."""

    role: str
    width: int
    partition: str
    logical_rows: int
    logical_cols: int
    physical_rows: int
    physical_cols: int
    groups: int
    weight_terms: tuple[Any, ...] = ()
    input_layout: str = ""
    output_layout: str = ""
    slices: tuple[ProjectionSlice, ...] = field(default=())


def _role_from_name_category(name: str, category: str) -> str:
    if name in _GROUP_PAIR_TERMS:
        return ROLE_ATTENTION_OUTPUT
    if name in _ATTENTION_OUTPUT_TERMS:
        return ROLE_ATTENTION_OUTPUT
    if name in _LINEAR_INPUT_TERMS:
        return ROLE_LINEAR_INPUT
    if name in _LINEAR_GATE_TERMS:
        return ROLE_LINEAR_GATE
    if name in _LINEAR_OUTPUT_TERMS:
        return ROLE_LINEAR_OUTPUT
    if name in {"lm_head", "vocab_logits"}:
        return ROLE_LOGITS
    if name in {"markov_head", "draft_markov_head"}:
        return ROLE_DRAFT_LOGITS
    if name == ROLE_TARGET_HEAD_ALIAS:
        return ROLE_TARGET_HEAD_ALIAS
    return role_from_category(category)


def role_from_category(category: str) -> str:
    """Map a declared quant category to its dense projection role."""
    name = str(category or "").strip().lower()
    if name in {"attn.o", "mla.o", "mla.o_a", "mla.o_b"}:
        return ROLE_ATTENTION_OUTPUT
    if name in {
            "kda.q", "kda.k", "kda.v", "kda.b", "kda.f_a", "kda.g_a", "kda.g",
            "gdn.qkvz", "gdn.beta_alpha", "linear.qkvbfg_a"}:
        return ROLE_LINEAR_INPUT
    if name in {"kda.f_b", "kda.g_b", "linear.decay_b", "linear.gate_b"}:
        return ROLE_LINEAR_GATE
    if name in {"kda.o", "gdn.o", "linear.output"}:
        return ROLE_LINEAR_OUTPUT
    if name in {"lm_head", "logits", "vocab_logits"}:
        return ROLE_LOGITS
    if name in {"markov_head", "draft_markov_head"}:
        return ROLE_DRAFT_LOGITS
    return ""


def projection_role(plan: Any, term_name: str, *, scope: str) -> str:
    """Return the declared placement role of one weight term, or ``""``."""
    del plan
    name = str(term_name)
    if scope and scope not in {"target", "mtp", "dspark", "draft"}:
        raise ModelingError(f"projection_scope_unknown:{scope}")
    if name == "markov_embed" or name in _EMBEDDING_TERMS:
        return ""
    return _role_from_name_category(name, "")


def role_width(role: str, parallel: Any) -> int:
    """Resolve the local width that owns ``role``."""
    if role in ATTENTION_ROLES:
        return int(parallel.o_proj_tp)
    if role in LINEAR_ROLES:
        return int(parallel.linear_proj_tp)
    if role in LOGIT_ROLES:
        return int(parallel.lm_head_tp)
    return ONE


def _padding_granule(width: int, term: Any) -> int:
    granule = max(ONE, int(width))
    quant = getattr(term, "quant", None) or getattr(term, "runtime_quant", None)
    block = getattr(quant, "weight_block", (ZERO, ZERO)) if quant is not None else (ZERO, ZERO)
    if block and len(block) > ONE:
        columns = max(ONE, int(block[ONE]))
        granule = int(granule * columns // gcd(granule, columns))
    return granule


def _padded_local(logical: int, width: int, term: Any) -> int:
    if width <= ONE:
        return int(logical)
    local = (int(logical) + width - ONE) // width
    return int(align_up(local, _padding_granule(width, term)))


def _term_role(term: Any) -> str:
    declared = str(getattr(term, "placement_role", "") or "")
    if declared:
        return declared
    return _role_from_name_category(str(term.name), str(term.category))


def placed_weight_shape(
    term: Any, parallel: Any, node_ranks: int,
) -> tuple[int, int, int]:
    """Per-rank (rows, cols, count) of one weight term under local placement."""
    del node_ranks
    role = _term_role(term)
    width = role_width(role, parallel) if role in SHARDED_ROLES else ONE
    rows, cols, count = int(term.rows), int(term.cols), int(term.count)
    resident = getattr(term, "resident_count", None)
    if resident is not None:
        count = int(resident)
    if width <= ONE:
        return _legacy_shape(term, parallel, count)
    if role == ROLE_ATTENTION_OUTPUT and (term.name in {"wo_a", "wo_b", "draft_wo_a", "draft_wo_b"}):
        groups = int(term.groups)
        if groups <= ZERO or groups % width != ZERO:
            raise ModelingError("local_tp_group_geometry_indivisible")
        if term.name.endswith("_a"):
            return rows // width, cols, count
        return rows, cols // width, count
    if role in {ROLE_ATTENTION_OUTPUT, ROLE_LINEAR_OUTPUT}:
        return rows, _padded_local(cols, width, term), count
    return _padded_local(rows, width, term), cols, count


def _legacy_shape(term: Any, parallel: Any, count: int) -> tuple[int, int, int]:
    """The pre-existing shard contract, with a resolved expert count."""
    from mindie_sim.catalog.weight_terms import _legacy_sharded_shape as legacy

    rows, cols, legacy_count = legacy(term, parallel, ONE)
    if getattr(term, "resident_count", None) is not None:
        if term.shard in {"expert_gate_up", "expert_down"}:
            return rows, cols, int(term.resident_count)
        return rows, cols, count
    return rows, cols, legacy_count


def _placement_for_terms(
    role: str, terms: Sequence[Any], parallel: Any,
) -> ProjectionPlacement:
    width = role_width(role, parallel)
    if not terms:
        raise ModelingError(f"projection_role_absent:{role}")
    if role in ATTENTION_ROLES and any(term.name in _GROUP_PAIR_TERMS for term in terms):
        representative = next(term for term in terms if term.name in {"wo_a", "draft_wo_a"})
        groups = int(representative.groups)
        rows = int(representative.rows)
        cols = int(representative.cols)
        if width > ONE:
            if groups <= ZERO or groups % width != ZERO:
                raise ModelingError("local_tp_group_geometry_indivisible")
            physical_rows = rows // width
            physical_cols = cols
        else:
            physical_rows, physical_cols = rows, cols
        return ProjectionPlacement(
            role, width, PARTITION_GROUP_PAIR, rows, cols, physical_rows,
            physical_cols, groups, tuple(terms), "activation_rows",
            "activation_rows")
    if role in {ROLE_ATTENTION_OUTPUT, ROLE_LINEAR_OUTPUT}:
        representative = terms[ZERO]
        rows, cols = int(representative.rows), int(representative.cols)
        physical_cols = _padded_local(cols, width, representative)
        return ProjectionPlacement(
            role, width, PARTITION_ROW, rows, cols, rows, physical_cols,
            max((int(term.groups) for term in terms), default=ZERO), tuple(terms),
            "input_feature_shard", "activation_rows")
    columns = {int(term.cols) for term in terms}
    if len(columns) != ONE:
        raise ModelingError("local_tp_packed_input_dimension_mismatch")
    logical_rows = sum(int(term.rows) for term in terms)
    cols = columns.pop()
    physical_rows = _padded_local(logical_rows, width, terms[ZERO])
    return ProjectionPlacement(
        role, width, PARTITION_COLUMN, logical_rows, cols, physical_rows,
        cols, sum(max(ZERO, int(term.groups)) for term in terms),
        tuple(terms), "activation_rows", "output_feature_shard")


def projection_placement(
    model: Any, plan: Any, parallel: Any, *, role: str, node_ranks: int,
    scope: str = "target",
) -> ProjectionPlacement:
    """Build the placement for one semantic role of a layer plan."""
    from mindie_sim.catalog.weights import layer_weight_terms

    if role not in SHARDED_ROLES:
        raise ModelingError(f"projection_role_unsupported:{role}")
    terms = tuple(
        term for term in layer_weight_terms(plan, model)
        if projection_role(plan, term.name, scope=scope) == role
    )
    return _placement_for_terms(role, terms, parallel)


def placed_term_bytes(
    term: Any, quant: Any, parallel: Any, node_ranks: int, *, source: bool = False,
) -> float:
    """Bytes of one term using the same placed shape as the graph rewrite."""
    from mindie_sim.schema.common import dtype_bytes

    if source and not term.source_owned:
        return ZERO_FLOAT
    rows, cols, count = placed_weight_shape(term, parallel, node_ranks)
    if not source and term.runtime_dtype:
        return float(rows * cols * count) * dtype_bytes(term.runtime_dtype)
    representation = (term.runtime_quant if not source else None) or term.quant or quant
    return float(representation.matrix_bytes(
        rows, cols, term.category, source=source)) * float(count)


def projection_slices(
    terms: Iterable[Any], placement: ProjectionPlacement, quant: Any,
) -> tuple[ProjectionSlice, ...]:
    """Split a packed column placement along its semantic output slices."""
    terms = tuple(terms)
    if placement.partition != PARTITION_COLUMN:
        return ()
    block = getattr(quant, "weight_block", (ZERO, ZERO)) or (ZERO, ZERO)
    granule = max(ONE, int(block[ONE]) if len(block) > ONE else ONE)
    slices: list[ProjectionSlice] = []
    byte_offset = ZERO
    extent_offset = ZERO
    width = max(ONE, placement.width)
    for term in terms:
        logical = int(term.rows)
        local = (logical + width - ONE) // width if width > ONE else logical
        padded = _padded_local(logical, width, term)
        boundaries = tuple(range(granule, local + ONE, granule)) if granule > ONE else ()
        bytes_per_row = float(quant.matrix_payload_bytes(ONE, int(term.cols), term.category))
        slices.append(ProjectionSlice(
            semantic_role=role_from_category(str(term.category)) or placement.role,
            byte_offset=int(byte_offset),
            logical_extent=logical,
            physical_extent=padded,
            quant_category=str(term.category),
            quant_group_boundaries=boundaries,
            name=str(term.name),
        ))
        byte_offset += int(round(bytes_per_row * local))
        extent_offset += local
    del extent_offset
    return tuple(slices)
