"""External shared-owner weight accounting.

[Role]
  Keep target-owned DSpark embedding/LM-head aliases separate from checkpoint
  bytes while retaining their consumer traffic and physical dependency.

[Invariants]
  * source_owned=False terms never become checkpoint parameter/weight bytes.
  * Runtime external-owner bytes are reported separately for union/dedup with
    the co-located target residency.
  * Lazy imports avoid import cycles with the weight geometry module.

[Timeouts & Locks]
  Pure arithmetic; no I/O.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from typing import Any

from .a3_types import ParallelSpec


def external_shared_weight_terms(model: Any) -> list[Any]:
    from .a3_catalog_weights import draft_terms

    if not model.is_standalone_draft or model.auxiliary is None:
        return []
    return [
        term for term in draft_terms(
            model.auxiliary.dspark, model, standalone_heads_only=True,
        ) if not term.source_owned
    ]


def external_shared_weight_bytes(
    model: Any, parallel: ParallelSpec, node_ranks: int = 1,
) -> float:
    from .a3_catalog_types import QuantSpec
    from .a3_catalog_weights import term_bytes

    quant = model.quant or QuantSpec("bf16")
    return float(sum(
        term_bytes(term, quant, parallel, node_ranks)
        for term in external_shared_weight_terms(model)
    ))


def body_parameter_count(model: Any) -> float:
    from .a3_catalog_weights import _body_terms

    return float(sum(
        term.logical_parameters for term in _body_terms(model)
        if term.source_owned
    ))


def body_weight_bytes(
    model: Any, parallel: ParallelSpec | None = None, node_ranks: int = 1,
) -> float:
    from .a3_catalog_types import QuantSpec
    from .a3_catalog_weights import _body_terms, term_bytes

    quant = model.quant or QuantSpec("bf16")
    parallel = parallel or ParallelSpec()
    return float(sum(
        term_bytes(term, quant, parallel, node_ranks)
        for term in _body_terms(model)
    ))


__all__ = [
    "body_parameter_count", "body_weight_bytes",
    "external_shared_weight_terms", "external_shared_weight_bytes",
]
