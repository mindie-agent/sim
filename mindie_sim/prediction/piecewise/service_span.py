"""L4 service-level TTFT wave scaling, host overhead, and kernel-attribution helpers.

[Role]
  Provide pure L4 framework scheduling / queueing / wave composition formulas
  and fused-kernel category attribution above L3 model body evaluation.
[Invariants]
  * Operator and subgraph durations (L1-L3) remain strictly free of L4 host
    queueing or multi-wave scheduling adjustments.
  * ``chunked_prefill_mean_ttft_ms`` models multi-chunk FCFS chunked-prefill
    scheduling across DP ranks.  When one rank executes ``N`` serial chunks, the
    arithmetic mean TTFT across its requests is ``T_total * (N + 1) / (2 * N)``.
  * ``chunked_prefill_request_ttft_ms`` exposes the same schedule as one finish
    per input request in original launch order; the mean wrapper reuses that
    computation and its historical rank-order sum.
  * ``ttft_wave_scale`` is the mean FCFS request finish in full-body units:
    requests are queued round-robin on DP ranks and each rank serves its queue
    serially.  A single request returns 1 (it must finish every one of its
    chunks); it is never the mean of chunk queue positions.  Unequal rank
    partitions use the exact 1+...+k finish sum, not ceil(c/dp).
  * ``service_ttft_overhead_ms`` returns a caller-declared host/first-step
    duration.  Tokenizer, IPC and scheduler time cannot be inferred from device
    topology, prompt length or TP/EP size, so this module does not synthesize it.
[Timeouts & Locks] Pure arithmetic; no I/O or locks.
[Side-Effect Policy] Stateless functions only.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence

from .lowering import lower_layer_predictions

__all__ = [
    "attribute_rope",
    "chunked_prefill_mean_ttft_ms",
    "chunked_prefill_request_ttft_ms",
    "lower_layer_predictions",
    "service_ttft_overhead_ms",
    "ttft_wave_scale",
]


def _body_takes_kv_context(body_one: Callable[..., float]) -> bool:
    """Whether a 1/2-argument callback expects ``(chunk, kv_context)``.

    The same arity probe is shared by the per-request and mean entries.  It
    inspects the callable's code/partial bindings only; it never calls the
    callback while deciding.
    """
    target_fn: Any = body_one
    bound_pos = 0
    bound_kw: set[str] = set()
    while hasattr(target_fn, "func"):
        bound_pos += len(getattr(target_fn, "args", ()) or ())
        bound_kw.update((getattr(target_fn, "keywords", None) or {}).keys())
        target_fn = target_fn.func
    if not hasattr(target_fn, "__code__") and hasattr(target_fn, "__call__"):
        target_fn = target_fn.__call__
    code = getattr(target_fn, "__code__", None)
    if getattr(target_fn, "__self__", None) is not None:
        bound_pos += 1
    rem_pos = (
        code.co_argcount - bound_pos
        - sum(1 for name in code.co_varnames[bound_pos:code.co_argcount]
              if name in bound_kw)
    ) if code is not None else 1
    return bool(code is not None and ((code.co_flags & 4) != 0 or rem_pos >= 2))


def _chunked_prefill_finishes(
    prompts: Sequence[int],
    *,
    dp: int,
    max_batched_tokens: int,
    body_one: Callable[..., float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return original-ordinal finishes and the FCFS rank-order view.

    The rank-order view preserves the exact public mean's floating-point sum
    order; the original-ordinal view is what launch consumers need.
    """
    budget = max(1, int(max_batched_tokens))
    prompts = tuple(prompts)
    ranks: list[list[tuple[int, int]]] = [[] for _ in range(max(1, int(dp)))]
    for index, prompt in enumerate(prompts):
        ranks[index % len(ranks)].append((index, max(1, int(prompt))))
    finishes = [0.0] * len(prompts)
    rank_order: list[float] = []
    two_arg = _body_takes_kv_context(body_one)
    for rank in ranks:
        clock = 0.0
        for index, prompt in rank:
            left = prompt
            done = 0
            while left > 0:
                take = min(budget, left)
                done += take
                step_ms = (float(body_one(take, done)) if two_arg
                           else float(body_one(take)))
                clock += step_ms
                left -= take
            finishes[index] = clock
            rank_order.append(clock)
    return tuple(finishes), tuple(rank_order)


def chunked_prefill_request_ttft_ms(
    prompts: Sequence[int],
    *,
    dp: int,
    max_batched_tokens: int = 8192,
    body_one: Callable[..., float],
) -> tuple[float, ...]:
    """Per-request TTFT under FCFS chunked prefill, in original launch order.

    Requests are placed round-robin onto ``dp`` ranks.  Each request stays on
    its rank and its chunks run serially, priced by
    ``body_one(chunk_tokens, kv_context_tokens)`` (or ``body_one(chunk_tokens)``
    for 1-argument callbacks).  The returned tuple has exactly one finite
    finish per input request, indexed by the input ordinal, not by rank order.
    This is the declared round-robin/FCFS scheduling model, not a claim about
    an unobserved physical scheduler.  Work is O(concurrent requests).
    """
    return _chunked_prefill_finishes(
        prompts, dp=dp, max_batched_tokens=max_batched_tokens,
        body_one=body_one)[0]


def chunked_prefill_mean_ttft_ms(
    prompts: Sequence[int],
    *,
    dp: int,
    max_batched_tokens: int = 8192,
    body_one: Callable[..., float],
) -> float:
    """Mean TTFT under FCFS chunked prefill across ``dp`` data-parallel ranks.

    Requests are placed round-robin onto ``dp`` ranks. Each request stays on its
    assigned rank, and chunks of up to ``max_batched_tokens`` on that rank run
    serially, priced by ``body_one(chunk_tokens, kv_context_tokens)`` (or
    ``body_one(chunk_tokens)`` for 1-argument callbacks).  This wrapper reuses
    the per-request ordinal computation and keeps the historical rank-order
    sum, so its value is exactly unchanged for uneven DP, multi-chunk and
    mixed-length inputs.
    """
    _, rank_order = _chunked_prefill_finishes(
        prompts, dp=dp, max_batched_tokens=max_batched_tokens,
        body_one=body_one)
    return sum(rank_order) / max(1, len(rank_order))


def ttft_wave_scale(
    *,
    phase: str,
    concurrency: int,
    dp: int,
    ep: int,
    layers: int,
    graph_buckets: tuple[int, ...],
    prompt_tokens: int = 0,
    max_batched_tokens: int | None = None,
) -> float:
    """Scale one already-batched body evaluation to mean L4 TTFT.

    ``body_ms`` in the public path is the body of the local batch
    (``ceil(concurrency/dp)`` requests), so every request in that batch
    finishes together and the scale is 1.0.  A single request must finish its
    whole body, including every chunk; it is never the average of chunk queue
    positions.  Real per-request FCFS/dispatch with unequal or chunked
    requests belongs to ``chunked_prefill_mean_ttft_ms``, which sums each
    request's actual chunk bodies and then means request finishes.
    """
    del layers, graph_buckets, ep, prompt_tokens, max_batched_tokens
    return 1.0


def service_ttft_overhead_ms(
    *,
    prompt_tokens: int = 0,
    tp: int = 1,
    ep: int = 1,
    host_overhead_ms: float = 0.0,
) -> float:
    """Return the caller-declared host/first-step overhead for one request.

    ``prompt_tokens``, ``tp`` and ``ep`` are accepted so deployment call sites
    keep one explicit request contract, but they do not synthesize overhead.
    Pass a measured/declared ``host_overhead_ms`` (zero when it is not modeled).
    """
    del prompt_tokens, tp, ep
    value = float(host_overhead_ms)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("host_overhead_ms_invalid")
    return value


def attribute_rope(kind: str, mechanism: str, subgraph: str) -> tuple[str, str]:
    """Map semantic rotary operator to its physical fused kernel bucket."""
    if mechanism != "rope":
        return mechanism, subgraph
    if kind == "A1":
        return "attention", "attention"
    if kind == "A0":
        return "norm_residual", "norm_residual"
    return mechanism, subgraph
