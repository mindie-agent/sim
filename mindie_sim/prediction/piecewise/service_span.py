"""L4 service-level TTFT wave scaling, host overhead, and kernel-attribution helpers.

[Role]
  Provide pure L4 framework scheduling / queueing / wave composition formulas
  and fused-kernel category attribution above L3 model body evaluation.
[Invariants]
  * Operator and subgraph durations (L1–L3) remain strictly free of L4 host
    queueing or multi-wave scheduling adjustments.
  * ``chunked_prefill_mean_ttft_ms`` models multi-chunk FCFS chunked-prefill
    scheduling across DP ranks.  When one rank executes ``N`` serial chunks, the
    arithmetic mean TTFT across its requests is ``T_total * (N + 1) / (2 * N)``.
  * ``ttft_wave_scale`` models only single-step concurrent-wave contention.  It
    is not the chunked-prefill queue formula: for prefill it scales with prompt
    length and concurrency, and it additionally honors ``max_batched_tokens``
    when a prompt exceeds one chunk.
[Timeouts & Locks] Pure arithmetic; no I/O or locks.
[Side-Effect Policy] Stateless functions only.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

from .lowering import lower_layer_predictions

__all__ = [
    "attribute_rope",
    "chunked_prefill_mean_ttft_ms",
    "lower_layer_predictions",
    "service_ttft_overhead_ms",
    "ttft_wave_scale",
]


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
    ``body_one(chunk_tokens)`` for 1-argument callbacks).
    """
    budget = max(1, int(max_batched_tokens))
    ranks: list[list[int]] = [[] for _ in range(max(1, int(dp)))]
    for index, prompt in enumerate(prompts):
        ranks[index % len(ranks)].append(max(1, int(prompt)))
    finishes: list[float] = []
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
    rem_pos = (code.co_argcount - bound_pos - sum(1 for n in code.co_varnames[bound_pos:code.co_argcount] if n in bound_kw)) if code is not None else 1
    two_arg = bool(code is not None and ((code.co_flags & 4) != 0 or rem_pos >= 2))
    for rank in ranks:
        clock = 0.0
        for prompt in rank:
            left = prompt
            done = 0
            while left > 0:
                take = min(budget, left)
                done += take
                step_ms = float(body_one(take, done)) if two_arg else float(body_one(take))
                clock += step_ms
                left -= take
            finishes.append(clock)
    return sum(finishes) / max(1, len(finishes))


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
    """Scale single-step prefill body to mean L4 TTFT under concurrent waves."""
    del layers, graph_buckets
    c, dp_n, ep_n = max(1, int(concurrency)), max(1, int(dp)), max(1, int(ep))
    if phase == "prefill":
        if max_batched_tokens is not None and int(max_batched_tokens) > 0:
            budget = int(max_batched_tokens)
            waves = max(1, math.ceil(c / dp_n) * math.ceil(max(1, int(prompt_tokens)) / budget))
            return (waves + 1.0) / (2.0 * waves)
        t_p = max(0.0, min(1.0, (float(prompt_tokens) - 2048.0) / 2048.0))
        t_c = max(0.0, min(1.0, (float(c) - 2.0) / 2.0))
        return 1.0 if ep_n < 16 else (1.0 + 0.045 * t_p * t_c)
    if ep_n == 1:
        at_dp = 1.0 + 0.01 * (dp_n - 1)
        return (1.0 + 0.01 * (c - 1)) if c <= dp_n else (at_dp + 0.65 * (c / dp_n - 1.0))
    if dp_n <= 4:
        return 1.0 + (1.15 / dp_n) * (min(c, 16) - 1) + 0.115 * max(0, c - 16)
    dp_t = max(0.0, min(1.0, (dp_n - 16.0) / 16.0))
    num = 0.81 * (1.0 + 0.295 * dp_t)
    slope = num / dp_n
    return 1.0 + slope * (c - 1)


def service_ttft_overhead_ms(
    *,
    prompt_tokens: int,
    tp: int,
    ep: int,
    short_ep_host_ms: float = 95.0,
) -> float:
    """Host tokenization, IPC, and first-step scheduler overhead for short prompts."""
    prompt = float(max(1, int(prompt_tokens)))
    if prompt >= 4096.0:
        return 0.0
    if prompt <= 1024.0:
        scale = 1.0
    elif prompt <= 2048.0:
        scale = 1.0 - 0.55 * ((prompt - 1024.0) / 1024.0)
    else:
        scale = 0.45 * (1.0 - (prompt - 2048.0) / 2048.0)
    if int(tp) >= 8:
        base = 145.0
    elif int(ep) > 1:
        base = float(short_ep_host_ms)
    else:
        base = 35.0
    return base * scale


def attribute_rope(kind: str, mechanism: str, subgraph: str) -> tuple[str, str]:
    """Map semantic rotary operator to its physical fused kernel bucket."""
    if mechanism != "rope":
        return mechanism, subgraph
    if kind == "A1":
        return "attention", "attention"
    if kind == "A0":
        return "norm_residual", "norm_residual"
    return mechanism, subgraph
