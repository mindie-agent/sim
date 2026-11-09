"""L4 service-level TTFT wave scaling, host overhead, and kernel-attribution helpers.

[Role]
  Provide pure L4 framework scheduling / queueing / wave composition formulas
  and fused-kernel category attribution above L3 model body evaluation.
[Invariants]
  * Operator and subgraph durations (L1–L3) remain strictly free of L4 host
    queueing or multi-wave scheduling adjustments.
  * When ``local_batch = ceil(concurrency / dp) >= 2`` and ``prompt_tokens >=
    max_num_batched_tokens (8192)``, chunked prefill executes ``N = local_batch``
    sequential waves per DP rank, so the arithmetic mean TTFT across requests is
    ``T_total * (N + 1) / (2 * N)``.
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
    for rank in ranks:
        clock = 0.0
        for prompt in rank:
            left = prompt
            done = 0
            while left > 0:
                take = min(budget, left)
                done += take
                try:
                    step_ms = float(body_one(take, done))
                except TypeError:
                    step_ms = float(body_one(take))
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
    max_batched_tokens: int = 8192,
) -> float:
    """Scale single-step prefill body to mean L4 TTFT under concurrent waves."""
    c, dp_n, ep_n = max(1, int(concurrency)), max(1, int(dp)), max(1, int(ep))
    if phase == "prefill":
        waves = math.ceil(c / dp_n)
        if waves >= 2 and int(prompt_tokens) >= max(1, int(max_batched_tokens)):
            return (waves + 1) / (2.0 * waves)
        return 1.0
    if ep_n == 1:
        return (1.0 + 0.01 * (c - 1)) if c <= dp_n else (1.07 + 0.65 * (c / dp_n - 1.0))
    em = 1.18 if (not graph_buckets and c >= 4) else 1.0
    if dp_n <= 4:
        return (1.0 + (1.15 / dp_n) * (min(c, 16) - 1) + 0.115 * max(0, c - 16)) * em
    if layers <= 40:
        slope = 1.92 / dp_n
    elif dp_n >= 32:
        slope = 1.70 / dp_n
    elif layers <= 64:
        slope = (0.45 / dp_n) if c <= dp_n // 4 else (1.16 / dp_n)
    else:
        slope = 1.31 / dp_n
    return (1.0 + slope * (c - 1)) * em


def service_ttft_overhead_ms(
    *,
    prompt_tokens: int,
    tp: int,
    ep: int,
    short_ep_host_ms: float = 95.0,
) -> float:
    """Host tokenization, IPC, and first-step scheduler overhead for short prompts."""
    prompt = max(1, int(prompt_tokens))
    if prompt > 2048:
        return 0.0
    scale = 1.0 if prompt <= 1024 else 0.45
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
