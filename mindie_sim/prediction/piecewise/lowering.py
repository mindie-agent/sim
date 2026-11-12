"""Post-compilation structural fusion attribution for piecewise operator models.

[Role]
  Attribute compiled per-operator piecewise costs (``OperatorRow`` / ``OperatorCost``)
  across a declared fused ``R1`` residual-attention boundary while strictly
  conserving per-layer and per-step operator time.

[Invariants]
  * Pure function of structural topology (``tp``, ``layers``) and compiled
    ``OperatorRow`` instances.
  * Never reads measured kernel traces, measured durations, or service timings.
  * Preserves all caller-supplied mechanism and subgraph keys and strictly conserves
    total per-layer operator time (``|sum(adj_per_layer.values()) - sum(per_layer.values())| <= 1e-12``),
    leaving ``body_ms``, ``decode_body_ms``, and ``prefill_body_ms`` invariant.
  * A standalone ``I0`` indexer row is its own compiled boundary.  No fraction of
    its duration is reassigned to a norm or projection clock because the
    compiler did not emit that fused launch.

[Timeouts & Locks]
  Pure in-memory arithmetic; no I/O, timers, subprocesses, or locks.

[Side-Effect Policy]
  Stateless and deterministic; mutates no caller mappings or global state.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

MECHANISM_ORDER: tuple[str, ...] = (
    "gemm",
    "attention",
    "elementwise_quant",
    "collective",
    "kv_cache",
    "activation",
    "norm_residual",
    "rope",
    "moe_router",
    "moe_grouped",
    "indexer",
    "gdn_kda",
    "hyperconnection",
)


def lower_layer_predictions(
    *,
    phase: str = "decode",
    prompt_tokens: int = 1024,
    concurrency: int = 1,
    max_tokens: int = 1,
    tp: int = 1,
    dp: int = 1,
    ep: int = 1,
    graph_buckets: Sequence[int] = (),
    graph_bucket: int = 0,
    layers: int = 1,
    rows: Sequence[Any] = (),
    per_layer: Mapping[str, float] = (),
    subgraphs: Mapping[str, float] = (),
    body_ms: float = 0.0,
    decode_body_ms: float = 0.0,
    prefill_body_ms: float = 0.0,
    **_extra: Any,
) -> tuple[dict[str, float], dict[str, float], float, float, float]:
    """Apply zero-sum structural ``R1`` and ``I0`` K-norm attribution."""
    del max_tokens, dp, ep, graph_buckets, graph_bucket, _extra
    layer_count = max(1, int(layers))
    raw_items = per_layer.items() if isinstance(per_layer, Mapping) else (per_layer or ())
    per_layer_in = {str(k): max(0.0, float(v)) for k, v in raw_items}
    subgraphs_in = {
        str(k): max(0.0, float(v))
        for k, v in (subgraphs.items() if isinstance(subgraphs, Mapping) else (subgraphs or ()))
    }
    has_r1 = any(
        getattr(r, "kind", "") == "R1" and getattr(r, "mechanism", "") == "norm_residual"
        for r in rows
    )
    has_attn = any(
        getattr(r, "kind", "") in {"A0", "A1", "A2", "A3", "A4"}
        or getattr(r, "mechanism", "") == "attention"
        for r in rows
    )
    i0_rows = [
        r
        for r in rows
        if getattr(r, "kind", "") == "I0" and getattr(r, "mechanism", "") == "indexer"
    ]
    has_i0_norm = (
        bool(i0_rows)
        and max(int(getattr(r, "count", 0) or 0) for r in i0_rows) >= layer_count
        and int(tp) == 1
    )
    if not rows or not ((has_r1 and has_attn) or has_i0_norm):
        return per_layer_in, subgraphs_in, float(body_ms), float(decode_body_ms), float(prefill_body_ms)

    per_layer_map = {m: per_layer_in.get(m, 0.0) for m in MECHANISM_ORDER}
    for key, val in per_layer_in.items():
        if key not in per_layer_map:
            per_layer_map[key] = val

    r1_raw = (
        sum(
            float(r.total_ms)
            for r in rows
            if getattr(r, "kind", "") == "R1" and getattr(r, "mechanism", "") == "norm_residual"
        )
        / layer_count
    )
    r1_ms = min(per_layer_map.get("norm_residual", 0.0), max(0.0, r1_raw)) if (has_r1 and has_attn) else 0.0

    adj_per_layer = dict(per_layer_map)
    if r1_ms > 0.0 and "norm_residual" in adj_per_layer:
        adj_per_layer["norm_residual"] = max(0.0, adj_per_layer["norm_residual"] - r1_ms)
        adj_per_layer["attention"] = adj_per_layer.get("attention", 0.0) + r1_ms
    # A standalone I0 row is its own compiled boundary.  Moving an assumed
    # fraction of the indexer or projection clock into the norm bucket would
    # invent a fused launch that the compiler did not emit, so no transfer is
    # applied here.  A genuinely fused K-projection norm must be represented by
    # the operator/compiler program itself.
    if subgraphs_in:
        adj_subgraphs = dict(subgraphs_in)
        nr_in = adj_subgraphs.get("norm_residual", 0.0)
        n0_ms = max(0.0, per_layer_map.get("norm_residual", 0.0) - r1_ms)
        r1_sg = min(nr_in, r1_ms, max(0.0, nr_in - n0_ms))
        if r1_sg > 0.0 and "norm_residual" in adj_subgraphs:
            adj_subgraphs["norm_residual"] = max(0.0, nr_in - r1_sg)
            adj_subgraphs["attention"] = adj_subgraphs.get("attention", 0.0) + r1_sg
    else:
        has_f2_coll = any(
            getattr(r, "kind", "") == "F2" and getattr(r, "mechanism", "") == "collective"
            for r in rows
        )
        moe_keys = (
            ("moe_grouped", "moe_router", "activation", "collective")
            if (has_f2_coll or int(tp) <= 1)
            else ("moe_grouped", "moe_router", "activation")
        )
        adj_subgraphs = {
            "attention": sum(
                adj_per_layer.get(k, 0.0)
                for k in ("attention", "rope", "kv_cache", "indexer", "gdn_kda")
            ),
            "norm_residual": adj_per_layer.get("norm_residual", 0.0),
            "hyperconnection": adj_per_layer.get("hyperconnection", 0.0),
            "moe": adj_per_layer.get("activation", 0.0)
            if adj_per_layer.get("moe_grouped", 0.0) <= 0.0
            else sum(adj_per_layer.get(k, 0.0) for k in moe_keys),
        }
    return (
        adj_per_layer,
        adj_subgraphs,
        float(body_ms),
        float(decode_body_ms),
        float(prefill_body_ms),
    )
