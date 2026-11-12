"""Open multi-request prefill scheduler for L4 TTFT, with declared conditions.

[Role]
  Model the V1 vLLM scheduler contract above the L3 body: one scheduler per DP
  rank, FCFS waiting admission, running requests first, a shared per-step token
  budget and a running-sequence cap, with the actual step mixture passed to a
  batch body primitive.
[Invariants]
  * The batch body is called once per non-empty rank step with per-request
    query tokens, KV-end tokens and kind; it is never multiplied or divided
    from a one-request scalar.
  * A request finishes only after the step that computes its last prompt token;
    finish >= declared arrival, steps are monotone and results keep original
    ordinal.
  * max_num_batched_tokens is a per-rank engine budget; max_num_seqs is a
    per-rank running cap. Neither is divided by DP and neither implies that an
    observed batch exists.
  * Route, arrival, prefix and background-decode inputs are declared model
    conditions. Missing physical route/batch observations stay unknown.
[Timeouts & Locks] Pure arithmetic; no I/O, device, filesystem or locks.
[Side-Effect Policy] Stateless; the only external call is the caller's
  ``body_batch`` once per scheduled step.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Callable, Sequence

__all__ = [
    "OpenBatchItem",
    "OpenPrefillFinishes",
    "OpenPrefillResult",
    "OpenRequestResult",
    "OpenStepResult",
    "chunked_prefill_open_scheduler_ttft_ms",
    "open_prefill_request_ttft_ms",
]


@dataclass(frozen=True)
class OpenBatchItem:
    """One request's contribution to one scheduled model batch.

    ``scheduled_tokens`` is the number of query/chunk tokens processed in this
    step; ``computed_before``/``computed_after`` are KV-end token counts.  The
    three quantities are intentionally separate: a chunked request's query is
    not its KV length, and a decode item's query is one token.
    """

    ordinal: int | None
    rank: int
    kind: str
    prompt_tokens: int
    scheduled_tokens: int
    computed_before: int
    computed_after: int
    is_last_prefill_chunk: bool
    decode_index: int | None


@dataclass(frozen=True)
class OpenRequestResult:
    ordinal: int
    rank: int
    finish_ms: float            # absolute lane finish
    prompt_tokens: int
    prefix_tokens: int          # declared/requested prefix
    computed_tokens: int
    steps: int
    batches: tuple[int, ...]
    arrival_ms: float = 0.0
    ttft_ms: float = 0.0        # launch-relative first-token TTFT
    effective_prefix_tokens: int = 0
    max_tokens: int = 1


@dataclass(frozen=True)
class OpenStepResult:
    rank: int
    step: int
    start_ms: float
    end_ms: float
    token_budget_used: int
    running_sequences: int
    items: tuple[OpenBatchItem, ...]


@dataclass(frozen=True)
class OpenPrefillResult:
    finishes_ms: tuple[float, ...]            # launch-relative TTFT
    absolute_finishes_ms: tuple[float, ...]   # absolute lane finishes
    requests: tuple[OpenRequestResult, ...]
    steps: tuple[OpenStepResult, ...]
    condition: dict[str, str]

    @property
    def mean_ttft_ms(self) -> float:
        return sum(self.finishes_ms) / max(1, len(self.finishes_ms))


class OpenPrefillFinishes(tuple):
    """Original-ordinal finishes with the scheduler trace as attributes.

    P14 consumes a tuple of finishes; keeping the condition, admission and
    per-request trace on the same tuple avoids a second scheduling pass.
    """

    def __new__(cls, values, *, metadata, steps, requests, condition):
        obj = super().__new__(cls, values)
        obj.metadata = dict(metadata)
        obj.steps = tuple(steps)
        obj.requests = tuple(requests)
        obj.condition = dict(condition)
        return obj

    @property
    def mean_ttft_ms(self) -> float:
        return sum(self) / max(1, len(self))

    @property
    def admission(self) -> dict[str, object]:
        return dict(self.metadata.get("admission", {}))

    @property
    def scheduler_condition(self) -> str:
        return str(self.metadata.get("scheduler_condition", ""))

    @property
    def actual_scheduler_observed(self) -> bool:
        return False


def _reject_bad_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name}_not_int:{value!r}")
    return value


def _validate_body_duration(value: object) -> float:
    duration = float(value)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("body_batch_duration_invalid")
    return duration


def _per_request_int(
    value: int | Sequence[int], name: str, count: int, minimum: int,
) -> tuple[int, ...]:
    if isinstance(value, bool):
        raise ValueError(f"{name}_not_int:{value!r}")
    if isinstance(value, int):
        values = (value,) * count
    else:
        values = tuple(_reject_bad_int(item, name) for item in value)
    if len(values) != count:
        raise ValueError(f"{name}_length_mismatch")
    if any(item < minimum for item in values):
        raise ValueError(f"{name}_invalid")
    return values


def open_prefill_request_ttft_ms(
    prompt_tokens: Sequence[int],
    *,
    dp: int,
    max_num_seqs: int = 32,
    max_num_batched_tokens: int = 8192,
    long_prefill_token_threshold: int = 0,
    body_batch: Callable[[tuple[OpenBatchItem, ...]], float],
    arrival_ms: Sequence[float] | None = None,
    prefix_tokens: Sequence[int] | None = None,
    rank_by_ordinal: Sequence[int] | None = None,
    arrival_order: Sequence[int] | None = None,
    max_tokens: int | Sequence[int] = 1,
    decode_sequences_by_rank: Sequence[int] | None = None,
    decode_steps: int = 0,
    decode_kv_tokens: int = 0,
) -> OpenPrefillResult:
    """Run the declared open V1-style prefill scheduler.

    The scheduler is deliberately one rank at a time because each DP rank owns
    an independent engine and token budget.  ``body_batch`` receives the whole
    rank-step mixture, including enrolled and background decode items.  This
    function does not claim that the declared route, arrivals, output lengths
    or batches are the observed physical ones; see ``result.condition``.
    """
    if not callable(body_batch):
        raise ValueError("body_batch_not_callable")
    prompts = tuple(_reject_bad_int(value, "prompt_tokens") for value in prompt_tokens)
    if any(value < 0 for value in prompts):
        raise ValueError("prompt_tokens_negative")
    count = len(prompts)
    ranks_count = max(1, int(dp))
    max_seqs = max(1, int(max_num_seqs))
    budget_limit = max(1, int(max_num_batched_tokens))
    threshold = max(0, int(long_prefill_token_threshold))
    output_limits = _per_request_int(max_tokens, "max_tokens", count, 1)

    if arrival_ms is None:
        arrivals = (0.0,) * count
    else:
        arrivals = tuple(float(value) for value in arrival_ms)
        if len(arrivals) != count:
            raise ValueError("arrival_ms_length_mismatch")
        if any(not math.isfinite(value) or value < 0.0 for value in arrivals):
            raise ValueError("arrival_ms_invalid")

    if prefix_tokens is None:
        requested_prefixes = (0,) * count
    else:
        requested_prefixes = tuple(_reject_bad_int(value, "prefix_tokens")
                                   for value in prefix_tokens)
        if len(requested_prefixes) != count:
            raise ValueError("prefix_tokens_length_mismatch")
    effective_prefixes: list[int] = []
    prefix_capped = False
    for index, prefix in enumerate(requested_prefixes):
        prompt = prompts[index]
        if prefix < 0 or prefix > prompt:
            raise ValueError(f"prefix_tokens_invalid:{index}")
        if prompt == 0:
            if prefix != 0:
                raise ValueError(f"prefix_tokens_invalid:{index}")
            effective_prefixes.append(0)
            continue
        effective = min(prefix, prompt - 1)
        if effective < prefix:
            prefix_capped = True
        effective_prefixes.append(effective)

    if rank_by_ordinal is None:
        ranks = tuple(index % ranks_count for index in range(count))
        route_condition = "round_robin_model_condition"
    else:
        ranks = tuple(_reject_bad_int(value, "rank_by_ordinal")
                      for value in rank_by_ordinal)
        if len(ranks) != count:
            raise ValueError("rank_by_ordinal_length_mismatch")
        if any(rank < 0 or rank >= ranks_count for rank in ranks):
            raise ValueError("rank_by_ordinal_out_of_range")
        route_condition = "declared_rank_by_ordinal"

    if arrival_order is None:
        arrival_priority = tuple(range(count))
        arrival_condition = "original_ordinal"
    else:
        arrival_priority = tuple(_reject_bad_int(value, "arrival_order")
                                 for value in arrival_order)
        if len(arrival_priority) != count:
            raise ValueError("arrival_order_length_mismatch")
        if sorted(arrival_priority) != list(range(count)):
            raise ValueError("arrival_order_not_permutation")
        priority_by_ordinal = [0] * count
        for position, ordinal in enumerate(arrival_priority):
            priority_by_ordinal[ordinal] = position
        arrival_priority = tuple(priority_by_ordinal)
        arrival_condition = "declared_arrival_order"

    if decode_sequences_by_rank is None:
        decode_by_rank = (0,) * ranks_count
    else:
        decode_by_rank = tuple(_reject_bad_int(value, "decode_sequences_by_rank")
                               for value in decode_sequences_by_rank)
        if len(decode_by_rank) != ranks_count:
            raise ValueError("decode_sequences_by_rank_length_mismatch")
        if any(value < 0 or value > max_seqs for value in decode_by_rank):
            raise ValueError("decode_sequences_by_rank_out_of_range")
    decode_step_limit = max(0, int(decode_steps))
    decode_kv = max(0, int(decode_kv_tokens))
    if any(decode_by_rank) and decode_step_limit <= 0:
        raise ValueError("decode_steps_required")

    computed = list(effective_prefixes)
    absolute_finishes = [0.0] * count
    ttfts = [0.0] * count
    request_batches: list[list[int]] = [[] for _ in range(count)]
    request_steps = [0] * count
    results: list[OpenRequestResult | None] = [None] * count
    steps: list[OpenStepResult] = []

    def record_first_token(
        ordinal: int, absolute_finish: float, step_count: int,
        batches: tuple[int, ...],
    ) -> None:
        ttft = max(0.0, float(absolute_finish) - arrivals[ordinal])
        absolute_finishes[ordinal] = float(absolute_finish)
        ttfts[ordinal] = ttft
        results[ordinal] = OpenRequestResult(
            ordinal=ordinal, rank=ranks[ordinal],
            finish_ms=float(absolute_finish),
            prompt_tokens=prompts[ordinal],
            prefix_tokens=requested_prefixes[ordinal],
            computed_tokens=computed[ordinal],
            steps=step_count, batches=batches,
            arrival_ms=arrivals[ordinal], ttft_ms=ttft,
            effective_prefix_tokens=effective_prefixes[ordinal],
            max_tokens=output_limits[ordinal],
        )

    # A zero-token request has no body work and is complete at its declared
    # arrival.  It does not occupy a running slot.
    queued_by_rank: list[list[int]] = [[] for _ in range(ranks_count)]
    for ordinal, prompt in enumerate(prompts):
        if prompt == 0:
            record_first_token(ordinal, arrivals[ordinal], 0, ())
        else:
            queued_by_rank[ranks[ordinal]].append(ordinal)

    for rank in range(ranks_count):
        queue = sorted(
            queued_by_rank[rank],
            key=lambda index: (arrivals[index], arrival_priority[index], index))
        running: list[int] = []
        # Enrolled requests have produced their first token and still owe
        # declared output tokens.  They occupy running slots and consume one
        # decode token before new prefill, as V1 does.
        enrolled_remaining: dict[int, int] = {}
        enrolled_kv: dict[int, int] = {}
        # Background decode is external declared load; keep per-sequence
        # remaining steps and KV so a small budget cannot erase later indices.
        background_remaining: dict[int, int] = {
            index: decode_step_limit for index in range(decode_by_rank[rank])
        }
        background_kv: dict[int, int] = {
            index: decode_kv for index in range(decode_by_rank[rank])
        }
        clock = 0.0
        rank_step = 0

        while queue or running or enrolled_remaining or background_remaining:
            # Once no prefill remains, future enrolled decodes cannot affect
            # any first token.  Do not require body calls for them.
            if not queue and not running:
                enrolled_remaining.clear()
                enrolled_kv.clear()

            if not running and not enrolled_remaining and not background_remaining:
                if not queue:
                    break
                next_arrival = arrivals[queue[0]]
                if next_arrival > clock:
                    clock = next_arrival

            step_budget = budget_limit
            items: list[OpenBatchItem] = []
            scheduled_prefill: list[tuple[int, int]] = []
            scheduled_enrolled: list[tuple[int, int]] = []
            scheduled_background: list[tuple[int, int]] = []

            # Running decode work is scheduled before new prefill.  Enrolled
            # requests are actual launch ordinals; background decode is an
            # explicitly declared external load (ordinal=None).
            for ordinal in sorted(enrolled_remaining):
                if step_budget <= 0:
                    break
                before = enrolled_kv[ordinal]
                after = before + 1
                items.append(OpenBatchItem(
                    ordinal=ordinal, rank=rank, kind="decode",
                    prompt_tokens=prompts[ordinal], scheduled_tokens=1,
                    computed_before=before, computed_after=after,
                    is_last_prefill_chunk=False, decode_index=None))
                step_budget -= 1
                scheduled_enrolled.append((ordinal, after))

            for decode_index in sorted(background_remaining):
                if step_budget <= 0:
                    break
                before = background_kv[decode_index]
                after = before + 1
                items.append(OpenBatchItem(
                    ordinal=None, rank=rank, kind="decode", prompt_tokens=0,
                    scheduled_tokens=1, computed_before=before,
                    computed_after=after, is_last_prefill_chunk=False,
                    decode_index=decode_index))
                step_budget -= 1
                scheduled_background.append((decode_index, after))

            next_running: list[int] = []
            for ordinal in running:
                if step_budget <= 0:
                    next_running.append(ordinal)
                    continue
                remaining = prompts[ordinal] - computed[ordinal]
                take = min(remaining, step_budget)
                if threshold > 0:
                    take = min(take, threshold)
                if take <= 0:
                    next_running.append(ordinal)
                    continue
                before = computed[ordinal]
                after = before + take
                last = after >= prompts[ordinal]
                items.append(OpenBatchItem(
                    ordinal=ordinal, rank=rank, kind="prefill",
                    prompt_tokens=prompts[ordinal], scheduled_tokens=take,
                    computed_before=before, computed_after=after,
                    is_last_prefill_chunk=last, decode_index=None))
                step_budget -= take
                next_running.append(ordinal)
                scheduled_prefill.append((ordinal, after))

            # Waiting requests are FCFS and only admissible once their declared
            # arrival is reached.  Running decodes/prefills occupy slots.
            active_slots = (
                len(running) + len(enrolled_remaining) + len(background_remaining))
            while queue and step_budget > 0 and active_slots < max_seqs:
                ordinal = queue[0]
                if arrivals[ordinal] > clock:
                    break
                queue.pop(0)
                remaining = prompts[ordinal] - computed[ordinal]
                take = min(remaining, step_budget)
                if threshold > 0:
                    take = min(take, threshold)
                if take <= 0:
                    # Should be unreachable: effective prefix is <= prompt-1
                    # for nonzero prompts, and zero prompts are pre-recorded.
                    record_first_token(
                        ordinal, clock, request_steps[ordinal],
                        tuple(request_batches[ordinal]))
                    continue
                before = computed[ordinal]
                after = before + take
                last = after >= prompts[ordinal]
                items.append(OpenBatchItem(
                    ordinal=ordinal, rank=rank, kind="prefill",
                    prompt_tokens=prompts[ordinal], scheduled_tokens=take,
                    computed_before=before, computed_after=after,
                    is_last_prefill_chunk=last, decode_index=None))
                step_budget -= take
                active_slots += 1
                next_running.append(ordinal)
                scheduled_prefill.append((ordinal, after))

            if not items:
                # No executable work at this instant: advance to the next
                # declared arrival; if none exists, the rank is drained.
                if queue and arrivals[queue[0]] > clock:
                    clock = arrivals[queue[0]]
                    continue
                if not running and not enrolled_remaining and not background_remaining:
                    break
                raise RuntimeError("open_prefill_scheduler_stalled")

            duration = _validate_body_duration(body_batch(tuple(items)))
            start = clock
            end = start + duration
            rank_step += 1
            steps.append(OpenStepResult(
                rank=rank, step=rank_step, start_ms=start, end_ms=end,
                token_budget_used=budget_limit - step_budget,
                running_sequences=len(items), items=tuple(items)))

            for ordinal, after in scheduled_prefill:
                computed[ordinal] = after
                request_batches[ordinal].append(rank_step)
                request_steps[ordinal] += 1
                if after >= prompts[ordinal]:
                    record_first_token(
                        ordinal, end, request_steps[ordinal],
                        tuple(request_batches[ordinal]))
                    remaining_outputs = output_limits[ordinal] - 1
                    if remaining_outputs > 0:
                        enrolled_remaining[ordinal] = remaining_outputs
                        enrolled_kv[ordinal] = after

            for ordinal, after in scheduled_enrolled:
                enrolled_kv[ordinal] = after
                left = enrolled_remaining[ordinal] - 1
                if left <= 0:
                    del enrolled_remaining[ordinal]
                    del enrolled_kv[ordinal]
                else:
                    enrolled_remaining[ordinal] = left

            for decode_index, after in scheduled_background:
                background_kv[decode_index] = after
                left = background_remaining[decode_index] - 1
                if left <= 0:
                    del background_remaining[decode_index]
                    del background_kv[decode_index]
                else:
                    background_remaining[decode_index] = left

            completed = {ordinal for ordinal, after in scheduled_prefill
                         if after >= prompts[ordinal]}
            running = [ordinal for ordinal in next_running
                       if ordinal not in completed]
            clock = end

    missing = [index for index, result in enumerate(results) if result is None]
    if missing:
        raise RuntimeError(f"open_prefill_scheduler_missing_result:{missing}")

    request_results = tuple(result for result in results if result is not None)
    if prefix_capped:
        prefix_condition = "declared_prefix_capped_prompt_minus_one"
    elif any(requested_prefixes):
        prefix_condition = "declared_prefix_tokens"
    else:
        prefix_condition = "not_observed_or_disabled"
    return OpenPrefillResult(
        finishes_ms=tuple(ttfts),
        absolute_finishes_ms=tuple(absolute_finishes),
        requests=request_results,
        steps=tuple(steps),
        condition={
            "scheduler": "vllm_v1_fcfs_per_rank_model_condition",
            "route_condition": route_condition,
            "arrival_condition": arrival_condition,
            "budget_scope": "per_rank_engine",
            "max_num_seqs_condition": "declared_per_rank_cap",
            "prefix_condition": prefix_condition,
            "prefix_bound_condition": ("prompt_minus_one" if prefix_capped
                                       else "declared"),
            "decode_condition": ("declared_background_decode"
                                 if any(decode_by_rank) else "none"),
            "termination_condition": (
                "declared_max_tokens_continuation_eos_unmodeled"
                if any(value > 1 for value in output_limits)
                else "declared_max_tokens_one_first_token_only"),
            "eos_condition": "unmodeled",
            "running_decode_order": "enrolled_then_background_then_prefill",
            "actual_scheduler_observed": "false",
            "source_vllm_version_observed": "0.30.0",
            "source_vllm_version_read": "0.27.1",
        })


def chunked_prefill_open_scheduler_ttft_ms(
    prompts: Sequence[int],
    *,
    dp: int,
    max_batched_tokens: int,
    max_num_seqs: int,
    arrival_order: Sequence[int] | None = None,
    body_batch: Callable[[Sequence[dict[str, object]]], float],
    scheduler: str = "vllm_v0_27_open_fcfs",
    max_tokens: int | Sequence[int] = 1,
) -> OpenPrefillFinishes:
    """P14-compatible open-scheduler entry point.

    ``body_batch`` receives the actual rank-step mixture as mappings with the
    P14/B13 BatchRequest keys ``ordinal, q_len, kv_end, phase, seq``.  The
    returned object is a tuple of original-ordinal finishes with ``metadata``,
    ``steps``, ``requests`` and ``condition`` attributes.
    """
    if scheduler not in {"vllm_v0_27_open_fcfs", "vllm_v1_open_fcfs"}:
        raise ValueError(f"scheduler_condition_unsupported:{scheduler}")
    if not callable(body_batch):
        raise ValueError("body_batch_not_callable")

    def adapted(items: tuple[OpenBatchItem, ...]) -> float:
        value = body_batch(tuple({
            "ordinal": item.ordinal,
            "q_len": item.scheduled_tokens,
            "kv_end": item.computed_after,
            "phase": item.kind,
            "seq": 1,
        } for item in items))
        # B13/predict_chunk_batch may return the canonical mapping; H3 advances
        # the clock only by its batch total, never by a per-request scalar.
        if isinstance(value, Mapping):
            value = value.get("total_ms")
        if value is None:
            raise ValueError("body_batch_total_ms_missing")
        return value

    result = open_prefill_request_ttft_ms(
        prompts, dp=dp, max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_batched_tokens, body_batch=adapted,
        arrival_order=arrival_order, max_tokens=max_tokens)
    metadata = dict(result.condition)
    metadata.update({
        "scheduler_condition": scheduler,
        "model_condition": True,
        "actual_scheduler_observed": False,
        "absolute_finishes_ms": tuple(result.absolute_finishes_ms),
        "admission": {
            "max_num_seqs": max(1, int(max_num_seqs)),
            "max_batched_tokens": max(1, int(max_batched_tokens)),
            "waiting_order": ("fcfs_declared_arrival_order" if arrival_order is not None
                              else "fcfs_original_ordinal"),
            "running_order": "running_first_then_waiting_fcfs",
        },
    })
    return OpenPrefillFinishes(
        result.finishes_ms, metadata=metadata, steps=result.steps,
        requests=result.requests, condition=result.condition)
