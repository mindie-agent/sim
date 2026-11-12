"""Open multi-request prefill scheduler: declared V1 conditions, no fits.

[Role]
  Verify the open scheduler prices a real rank-step batch mixture, preserves
  per-request query/KV/concurrency state and original ordinal, and labels all
  unobserved route/arrival/batch conditions instead of inferring them.
[Invariants]
  * Identical step budget co-batches admissible requests; a split budget
    serializes them.  The batch body is called once per non-empty rank step.
  * C2 and C8 use the same source limits: simultaneous C8 can co-batch two
    requests per DP lane, unlike a forced one-per-lane second wave.  A declared
    split arrival is what creates two clusters; max_num_seqs32 alone does not.
  * max_num_batched_tokens is per rank, not divided by DP.
  * Finish is causal (>= arrival), monotone, and original-ordinal; prefix,
    query tokens and KV-end tokens stay separate.
  * Background decode consumes slots/budget and is passed in the same batch.
  * Empty and validation edges are explicit.
[Timeouts & Locks] Tier 0 in-memory arithmetic; no I/O, device or locks.
[Side-Effect Policy] Stateless functions; body callback is the only external
  call and is recorded once per scheduler step.
"""
from __future__ import annotations

import math

import pytest

from mindie_sim.prediction.piecewise.open_prefill_scheduler import (
    OpenBatchItem,
    chunked_prefill_open_scheduler_ttft_ms,
    open_prefill_request_ttft_ms,
)


def _body(items: tuple[OpenBatchItem, ...]) -> float:
    # Deliberately a batch primitive: per-sequence query + KV context, plus a
    # declared batch interaction term.  Not a one-request scalar.
    return (0.001 * sum(item.scheduled_tokens for item in items)
            + 0.0001 * sum(item.computed_before for item in items)
            + 0.01 * len(items))


def test_identical_budget_cobatches_split_budget_serializes() -> None:
    calls: list[tuple[OpenBatchItem, ...]] = []

    def body(items):
        calls.append(items)
        return _body(items)

    co = open_prefill_request_ttft_ms(
        (4096, 4096), dp=1, max_num_seqs=32, max_num_batched_tokens=8192,
        body_batch=body)
    assert len(co.steps) == 1
    assert len(calls) == 1
    assert len(co.steps[0].items) == 2
    assert co.steps[0].token_budget_used == 8192
    assert co.finishes_ms[0] == co.finishes_ms[1]

    calls.clear()
    split = open_prefill_request_ttft_ms(
        (4096, 4096), dp=1, max_num_seqs=32, max_num_batched_tokens=4096,
        body_batch=body)
    assert len(split.steps) == 2
    assert len(calls) == 2
    assert [len(step.items) for step in split.steps] == [1, 1]
    assert split.finishes_ms[0] < split.finishes_ms[1]


def test_source_limits_c2_c8_open_batch_and_declared_split_arrival() -> None:
    prompts = (4096,) * 8
    calls: list[tuple[OpenBatchItem, ...]] = []

    def body(items):
        calls.append(items)
        return 123.0

    c2 = open_prefill_request_ttft_ms(
        prompts[:2], dp=4, max_num_seqs=32, max_num_batched_tokens=8192,
        body_batch=body)
    assert len(c2.steps) == 2
    assert all(len(step.items) == 1 for step in c2.steps)
    assert c2.finishes_ms == (123.0, 123.0)

    calls.clear()
    c8 = open_prefill_request_ttft_ms(
        prompts, dp=4, max_num_seqs=32, max_num_batched_tokens=8192,
        body_batch=body)
    # Each of the four source DP lanes can co-batch its two 4096-token requests;
    # there is no forced second wave just because concurrency exceeds DP.
    assert len(c8.steps) == 4
    assert all(len(step.items) == 2 for step in c8.steps)
    assert all(step.token_budget_used == 8192 for step in c8.steps)
    assert c8.finishes_ms == (123.0,) * 8
    assert len(calls) == 4  # one batch call per lane/step, not eight scalars
    assert c8.condition["route_condition"] == "round_robin_model_condition"
    assert c8.condition["actual_scheduler_observed"] == "false"

    # max_num_seqs32 is not the batch; a declared arrival split is what makes
    # two clusters.  The actual lane/arrival assignment remains unobserved.
    split = open_prefill_request_ttft_ms(
        prompts, dp=4, max_num_seqs=32, max_num_batched_tokens=8192,
        body_batch=body, arrival_ms=(0, 0, 0, 0, 100, 100, 100, 100))
    assert split.finishes_ms[:4] == (123.0,) * 4
    # TTFT is launch-relative: the second group's absolute finish 246 ms
    # minus its declared arrival 100 ms is 146 ms.
    assert split.finishes_ms[4:] == (146.0,) * 4
    assert split.absolute_finishes_ms[:4] == (123.0,) * 4
    assert split.absolute_finishes_ms[4:] == (246.0,) * 4
    assert len(split.steps) == 8
    assert split.condition["actual_scheduler_observed"] == "false"

    # max_num_seqs1 is a control that serializes even with simultaneous arrival.
    serial = open_prefill_request_ttft_ms(
        prompts[:2], dp=1, max_num_seqs=1, max_num_batched_tokens=8192,
        body_batch=body)
    assert serial.finishes_ms == (123.0, 246.0)


def test_per_rank_budget_is_not_divided_by_dp() -> None:
    prompts = (4096,) * 4
    result = open_prefill_request_ttft_ms(
        prompts, dp=2, max_num_seqs=32, max_num_batched_tokens=8192,
        body_batch=lambda items: 0.001 * sum(item.scheduled_tokens for item in items))
    assert len(result.steps) == 2  # rank0 and rank1, each one co-batch
    assert [len(step.items) for step in result.steps] == [2, 2]
    assert [step.token_budget_used for step in result.steps] == [8192, 8192]
    assert result.condition["budget_scope"] == "per_rank_engine"


def test_non_even_dp_mixed_q_original_ordinal_and_query_kv_separate() -> None:
    prompts = (4096, 8192, 4096)
    calls: list[tuple[OpenBatchItem, ...]] = []

    def body(items):
        calls.append(items)
        return 0.001 * sum(item.scheduled_tokens for item in items)

    result = open_prefill_request_ttft_ms(
        prompts, dp=2, max_num_seqs=32, max_num_batched_tokens=8192,
        body_batch=body)
    assert result.finishes_ms == (8.192, 8.192, 8.192)
    assert [request.rank for request in result.requests] == [0, 1, 0]
    # rank0 has two requests in one batch; rank1 has one.  No equal split.
    assert len(result.steps) == 2
    assert [(item.ordinal, item.scheduled_tokens, item.computed_before)
            for item in result.steps[0].items] == [(0, 4096, 0), (2, 4096, 0)]
    assert [(item.ordinal, item.scheduled_tokens, item.computed_before)
            for item in result.steps[1].items] == [(1, 8192, 0)]
    assert result.requests[0].computed_tokens == 4096
    assert result.requests[1].computed_tokens == 8192
    assert result.requests[2].finish_ms == result.requests[0].finish_ms

    declared = open_prefill_request_ttft_ms(
        prompts, dp=2, max_num_seqs=32, max_num_batched_tokens=8192,
        body_batch=body, rank_by_ordinal=(0, 0, 1))
    assert declared.condition["route_condition"] == "declared_rank_by_ordinal"
    assert [request.rank for request in declared.requests] == [0, 0, 1]
    assert len(declared.steps[0].items) == 2
    assert {item.ordinal for item in declared.steps[0].items} == {0, 1}


def test_route_condition_is_declared_not_measured() -> None:
    prompts = (4096, 2048, 4096)
    default = open_prefill_request_ttft_ms(
        prompts, dp=2, max_num_seqs=32, max_num_batched_tokens=8192,
        body_batch=_body)
    declared = open_prefill_request_ttft_ms(
        prompts, dp=2, max_num_seqs=32, max_num_batched_tokens=8192,
        body_batch=_body, rank_by_ordinal=(0, 1, 1))
    assert default.condition["route_condition"] == "round_robin_model_condition"
    assert declared.condition["route_condition"] == "declared_rank_by_ordinal"
    assert default.condition["actual_scheduler_observed"] == "false"
    assert declared.condition["actual_scheduler_observed"] == "false"
    assert default.finishes_ms != declared.finishes_ms
    with pytest.raises(ValueError, match="rank_by_ordinal_out_of_range"):
        open_prefill_request_ttft_ms(
            prompts, dp=2, body_batch=_body, rank_by_ordinal=(0, 2, 1))


def test_mixed_decode_uses_slots_budget_and_the_same_body_batch() -> None:
    calls: list[tuple[OpenBatchItem, ...]] = []

    def body(items):
        calls.append(items)
        return 0.1 * len(items) + 0.001 * sum(item.scheduled_tokens for item in items)

    without = open_prefill_request_ttft_ms(
        (4096,), dp=1, max_num_seqs=1, max_num_batched_tokens=8192,
        body_batch=body)
    calls.clear()
    mixed = open_prefill_request_ttft_ms(
        (4096,), dp=1, max_num_seqs=1, max_num_batched_tokens=8192,
        body_batch=body, decode_sequences_by_rank=(1,), decode_steps=2,
        decode_kv_tokens=1000)
    assert mixed.condition["decode_condition"] == "declared_background_decode"
    assert mixed.finishes_ms[0] > without.finishes_ms[0]
    assert [item.kind for item in mixed.steps[0].items] == ["decode"]
    assert [item.kind for item in mixed.steps[1].items] == ["decode"]
    assert [item.kind for item in mixed.steps[2].items] == ["prefill"]
    assert [step.items[0].computed_before for step in mixed.steps[:2]] == [1000, 1001]
    assert len(calls) == 3
    assert mixed.steps[2].items[0].scheduled_tokens == 4096


def test_prefix_arrival_causality_and_separate_tokens() -> None:
    result = open_prefill_request_ttft_ms(
        (4096,), dp=1, max_num_seqs=32, max_num_batched_tokens=8192,
        body_batch=lambda items: 0.001 * sum(item.scheduled_tokens for item in items),
        prefix_tokens=(1024,), arrival_ms=(100.0,))
    assert result.condition["prefix_condition"] == "declared_prefix_tokens"
    assert len(result.steps) == 1
    item = result.steps[0].items[0]
    assert (item.scheduled_tokens, item.computed_before, item.computed_after) == (
        3072, 1024, 4096)
    assert result.steps[0].start_ms == 100.0
    assert result.finishes_ms[0] == pytest.approx(3.0720000000000027, rel=1e-15)
    assert result.absolute_finishes_ms[0] == pytest.approx(103.0720000000000027, rel=1e-15)
    assert result.requests[0].ttft_ms == pytest.approx(3.0720000000000027, rel=1e-15)
    assert result.requests[0].prefix_tokens == 1024
    assert result.requests[0].computed_tokens == 4096
    assert result.requests[0].steps == 1


def test_p14_compatible_tuple_metadata_and_batchrequest_shape() -> None:
    calls: list[tuple[dict[str, object], ...]] = []

    def body(requests):
        calls.append(tuple(requests))
        return 10.0 + 2.0 * len(requests)

    result = chunked_prefill_open_scheduler_ttft_ms(
        (4096,) * 8, dp=4, max_batched_tokens=8192, max_num_seqs=32,
        body_batch=body)
    assert isinstance(result, tuple)
    assert len(result) == 8
    assert result == (14.0,) * 8  # four lanes, two requests each: 10+2*2
    assert result.metadata["scheduler_condition"] == "vllm_v0_27_open_fcfs"
    assert result.metadata["model_condition"] is True
    assert result.metadata["actual_scheduler_observed"] is False
    assert result.metadata["admission"]["max_num_seqs"] == 32
    assert result.metadata["admission"]["max_batched_tokens"] == 8192
    assert len(calls) == 4
    assert set(calls[0][0]) == {"ordinal", "q_len", "kv_end", "phase", "seq"}
    assert calls[0][0]["seq"] == 1
    assert calls[0][0]["q_len"] == 4096
    assert calls[0][0]["kv_end"] == 4096

    ordered = chunked_prefill_open_scheduler_ttft_ms(
        (4096, 4096), dp=1, max_batched_tokens=8192, max_num_seqs=32,
        arrival_order=(1, 0), body_batch=body)
    def batch_mapping(requests):
        return {"total_ms": 10.0 + 2.0 * len(requests),
                "per_request_ms": {},
                "rows": [], "scope": "single admitted batch"}

    mapped = chunked_prefill_open_scheduler_ttft_ms(
        (4096, 4096), dp=1, max_batched_tokens=8192, max_num_seqs=32,
        body_batch=batch_mapping)
    assert mapped == (14.0, 14.0)
    assert [item.ordinal for item in ordered.steps[0].items] == [1, 0]
    with pytest.raises(ValueError, match="arrival_order_not_permutation"):
        chunked_prefill_open_scheduler_ttft_ms(
            (4096, 4096), dp=1, max_batched_tokens=8192, max_num_seqs=32,
            arrival_order=(0, 0), body_batch=body)
    with pytest.raises(ValueError, match="scheduler_condition_unsupported"):
        chunked_prefill_open_scheduler_ttft_ms(
            (4096,), dp=1, max_batched_tokens=8192, max_num_seqs=32,
            scheduler="measured_scheduler", body_batch=body)


def test_empty_chunked_and_validation_edges() -> None:
    calls: list[tuple[OpenBatchItem, ...]] = []

    def body(items):
        calls.append(items)
        return 1.0

    empty = open_prefill_request_ttft_ms(
        (), dp=4, body_batch=body)
    assert empty.finishes_ms == ()
    assert empty.requests == ()
    assert empty.steps == ()
    assert empty.mean_ttft_ms == 0.0
    assert calls == []

    chunked = open_prefill_request_ttft_ms(
        (8192,), dp=1, max_num_seqs=32, max_num_batched_tokens=4096,
        body_batch=body)
    assert len(chunked.steps) == 2
    assert [item.scheduled_tokens for item in chunked.steps[0].items] == [4096]
    assert [item.scheduled_tokens for item in chunked.steps[1].items] == [4096]
    assert chunked.finishes_ms == (2.0,)

    capped = open_prefill_request_ttft_ms(
        (4096,), dp=1, max_num_seqs=32, max_num_batched_tokens=8192,
        long_prefill_token_threshold=1024, body_batch=body)
    assert [item.scheduled_tokens for item in capped.steps[0].items] == [1024]
    assert [item.scheduled_tokens for item in capped.steps[1].items] == [1024]
    assert len(capped.steps) == 4

    with pytest.raises(ValueError, match="prompt_tokens_negative"):
        open_prefill_request_ttft_ms((-1,), dp=1, body_batch=body)
    with pytest.raises(ValueError, match="prefix_tokens_invalid"):
        open_prefill_request_ttft_ms((10,), dp=1, body_batch=body,
                                     prefix_tokens=(11,))
    with pytest.raises(ValueError, match="arrival_ms_length_mismatch"):
        open_prefill_request_ttft_ms((10,), dp=1, body_batch=body,
                                     arrival_ms=(0, 1))
    with pytest.raises(ValueError, match="decode_steps_required"):
        open_prefill_request_ttft_ms((10,), dp=1, body_batch=body,
                                     decode_sequences_by_rank=(1,))
    with pytest.raises(ValueError, match="body_batch_duration_invalid"):
        open_prefill_request_ttft_ms(
            (10,), dp=1, body_batch=lambda _items: math.nan)


def test_source_c8_q8192_max128_enrolled_decode_delays_next_prefill() -> None:
    prompts = (8192,) * 8
    calls: list[tuple[OpenBatchItem, ...]] = []

    def body(items):
        calls.append(items)
        return 1.0

    result = open_prefill_request_ttft_ms(
        prompts, dp=4, max_num_seqs=128, max_num_batched_tokens=8192,
        max_tokens=128, body_batch=body)
    # First four first tokens at step 1; each rank's second request gets
    # 8191 tokens after the first request's one decode token, then the
    # remaining 1 token after a second decode token.
    assert result.finishes_ms[:4] == (1.0,) * 4
    assert result.finishes_ms[4:] == (3.0,) * 4
    assert result.absolute_finishes_ms == result.finishes_ms
    assert len(result.steps) == 12
    rank0 = [step for step in result.steps if step.rank == 0]
    assert len(rank0) == 3
    assert [(item.ordinal, item.kind, item.scheduled_tokens)
            for item in rank0[0].items] == [(0, "prefill", 8192)]
    assert [(item.ordinal, item.kind, item.scheduled_tokens)
            for item in rank0[1].items] == [(0, "decode", 1), (4, "prefill", 8191)]
    assert [(item.ordinal, item.kind, item.scheduled_tokens)
            for item in rank0[2].items] == [(0, "decode", 1), (4, "prefill", 1)]
    # First token is recorded once; later decodes do not overwrite it.
    assert result.requests[0].ttft_ms == 1.0
    assert result.requests[4].ttft_ms == 3.0
    assert result.requests[0].computed_tokens == 8192
    assert result.condition["termination_condition"].endswith("eos_unmodeled")
    assert len(calls) == 12  # no body calls after all eight first tokens

    calls.clear()
    one = open_prefill_request_ttft_ms(
        (8192, 8192), dp=1, max_num_seqs=128,
        max_num_batched_tokens=8192, max_tokens=1, body_batch=body)
    assert one.finishes_ms == (1.0, 2.0)
    assert len(calls) == 2
    assert len(one.steps) == 2

    calls.clear()
    one8 = open_prefill_request_ttft_ms(
        prompts, dp=4, max_num_seqs=128, max_num_batched_tokens=8192,
        max_tokens=1, body_batch=body)
    assert one8.finishes_ms[:4] == (1.0,) * 4
    assert one8.finishes_ms[4:] == (2.0,) * 4
    assert len(one8.steps) == 8
    assert len(calls) == 8


def test_arrival_shift_does_not_change_launch_relative_mean_ttft() -> None:
    def body(_items):
        return 1.0

    base = open_prefill_request_ttft_ms(
        (4096,), dp=1, max_num_seqs=32, max_num_batched_tokens=8192,
        body_batch=body)
    shifted = open_prefill_request_ttft_ms(
        (4096,), dp=1, max_num_seqs=32, max_num_batched_tokens=8192,
        body_batch=body, arrival_ms=(100.0,))
    assert base.finishes_ms == (1.0,)
    assert base.absolute_finishes_ms == (1.0,)
    assert shifted.finishes_ms == (1.0,)
    assert shifted.absolute_finishes_ms == (101.0,)
    assert shifted.mean_ttft_ms == base.mean_ttft_ms == 1.0
    assert shifted.requests[0].arrival_ms == 100.0
    assert shifted.requests[0].ttft_ms == 1.0
    assert shifted.requests[0].finish_ms == 101.0


def test_background_decode_each_sequence_progresses_under_budget_one() -> None:
    calls: list[tuple[OpenBatchItem, ...]] = []

    def body(items):
        calls.append(items)
        return 1.0

    result = open_prefill_request_ttft_ms(
        (), dp=1, max_num_seqs=3, max_num_batched_tokens=1,
        body_batch=body, decode_sequences_by_rank=(3,),
        decode_steps=2, decode_kv_tokens=10)
    assert result.finishes_ms == ()
    assert result.requests == ()
    assert len(result.steps) == 6
    progress = [step.items[0] for step in result.steps]
    assert [item.decode_index for item in progress] == [0, 0, 1, 1, 2, 2]
    assert [(item.computed_before, item.computed_after) for item in progress] == [
        (10, 11), (11, 12), (10, 11), (11, 12), (10, 11), (11, 12)]
    assert len(calls) == 6

    # The three active decodes occupy all three slots; the prefill is only
    # admitted after they finish, then completes under the same budget.
    calls.clear()
    admitted = open_prefill_request_ttft_ms(
        (1,), dp=1, max_num_seqs=3, max_num_batched_tokens=1,
        body_batch=body, decode_sequences_by_rank=(3,),
        decode_steps=2, decode_kv_tokens=10)
    assert len(admitted.steps) == 7
    assert admitted.steps[-1].items[0].kind == "prefill"
    assert admitted.finishes_ms == (7.0,)
    assert len(calls) == 7


def test_full_prefix_is_capped_at_prompt_minus_one() -> None:
    calls: list[tuple[OpenBatchItem, ...]] = []

    def body(items):
        calls.append(items)
        return 1.0

    full = open_prefill_request_ttft_ms(
        (16,), dp=1, max_num_seqs=32, max_num_batched_tokens=8192,
        prefix_tokens=(16,), body_batch=body)
    assert full.condition["prefix_condition"] == "declared_prefix_capped_prompt_minus_one"
    assert full.condition["prefix_bound_condition"] == "prompt_minus_one"
    assert len(full.steps) == 1
    item = full.steps[0].items[0]
    assert (item.scheduled_tokens, item.computed_before, item.computed_after) == (1, 15, 16)
    assert full.finishes_ms == (1.0,)
    assert len(calls) == 1

    exactly_left = open_prefill_request_ttft_ms(
        (16,), dp=1, max_num_seqs=32, max_num_batched_tokens=8192,
        prefix_tokens=(15,), body_batch=body)
    assert exactly_left.condition["prefix_condition"] == "declared_prefix_tokens"
    assert exactly_left.condition["prefix_bound_condition"] == "declared"
    assert exactly_left.steps[0].items[0].scheduled_tokens == 1

    empty = open_prefill_request_ttft_ms(
        (0,), dp=1, max_num_seqs=32, max_num_batched_tokens=8192,
        prefix_tokens=(0,), body_batch=body)
    assert empty.finishes_ms == (0.0,)
    assert empty.steps == ()
    with pytest.raises(ValueError, match="prefix_tokens_invalid"):
        open_prefill_request_ttft_ms(
            (16,), dp=1, body_batch=body, prefix_tokens=(17,))
    with pytest.raises(ValueError, match="prefix_tokens_invalid"):
        open_prefill_request_ttft_ms(
            (0,), dp=1, body_batch=body, prefix_tokens=(1,))


def test_mixed_output_endpoints_ragged_q_and_separate_kv() -> None:
    result = open_prefill_request_ttft_ms(
        (4096, 8192, 4096), dp=1, max_num_seqs=32,
        max_num_batched_tokens=8192, max_tokens=(128, 1, 128),
        body_batch=lambda _items: 1.0)
    assert result.finishes_ms == (1.0, 2.0, 3.0)
    assert result.requests[0].max_tokens == 128
    assert result.requests[1].max_tokens == 1
    assert result.requests[2].max_tokens == 128
    assert result.requests[0].computed_tokens == 4096
    assert result.requests[1].computed_tokens == 8192
    assert result.requests[2].computed_tokens == 4096
    # ord0 continues decoding while ord1's ragged prefill and ord2's one
    # remaining token are scheduled; each item keeps its own query/KV.
    assert [(item.ordinal, item.kind, item.scheduled_tokens,
             item.computed_before, item.computed_after)
            for item in result.steps[1].items] == [
                (0, "decode", 1, 4096, 4097),
                (1, "prefill", 4096, 4096, 8192),
                (2, "prefill", 4095, 0, 4095),
            ]
    assert [(item.ordinal, item.kind, item.scheduled_tokens,
             item.computed_before, item.computed_after)
            for item in result.steps[2].items] == [
                (0, "decode", 1, 4097, 4098),
                (2, "prefill", 1, 4095, 4096),
            ]
    # First-token results are not overwritten by later decodes.
    assert result.requests[0].ttft_ms == 1.0
    assert result.requests[2].ttft_ms == 3.0
