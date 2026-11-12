"""Per-request chunked-prefill TTFT in original launch order.

[Role]
  The L4 helper must expose one finish per input request, indexed by the
  original launch ordinal, while the public mean remains exactly the prior
  FCFS/DP mean even when rank order differs from launch order.
[Invariants]
  * ``chunked_prefill_request_ttft_ms`` returns one finite finish per prompt
    in input order; the mean wrapper reuses that computation.
  * The historical mean is bitwise unchanged for uneven DP, multi-chunk and
    the real mixed eight-request declaration.
  * One- and two-argument callbacks, bound callables and partials receive one
    call per chunk; callback introspection never calls the body.
  * Empty, single-request and clamped edge inputs keep their prior behavior.
[Timeouts & Locks] Tier 0 in-memory arithmetic; no I/O, device or locks.
[Side-Effect Policy] Stateless functions only.
"""
from __future__ import annotations

import functools
import inspect

import pytest

from mindie_sim.prediction.piecewise.service_span import (
    chunked_prefill_mean_ttft_ms,
    chunked_prefill_request_ttft_ms,
)


def _deterministic(take: int, done: int) -> float:
    return 0.001 * take + 0.0001 * done


def test_original_ordinal_counterexample_three_requests_two_dp():
    prompts = (8192, 4096, 16384)
    calls: list[tuple[int, int]] = []

    def body(take: int, done: int) -> float:
        calls.append((take, done))
        return _deterministic(take, done)

    finishes = chunked_prefill_request_ttft_ms(
        prompts, dp=2, max_batched_tokens=8192, body_one=body)
    # launch ordinal: req0 on rank0, req1 on rank1, req2 behind req0 on rank0.
    # A rank-order tuple would be (9.0112, 27.8528, 4.5056).
    assert finishes == pytest.approx((9.0112, 4.5056, 27.852800000000002), rel=1e-15)
    assert finishes[1] < finishes[0] < finishes[2]
    # Mutation control: rank order is observably different from launch order.
    assert finishes != pytest.approx((9.0112, 27.852800000000002, 4.5056), rel=1e-15)
    # One callback per chunk, executed in rank order (req0, req2, req2, req1).
    assert len(calls) == 4
    assert calls.count((8192, 8192)) == 2
    assert calls.count((8192, 16384)) == 1
    assert calls.count((4096, 4096)) == 1
    mean = chunked_prefill_mean_ttft_ms(
        prompts, dp=2, max_batched_tokens=8192, body_one=_deterministic)
    assert mean == 13.789866666666668


def test_mean_is_exactly_the_prior_values_for_real_mixed8_and_edges():
    mixed8 = (65536,) + (8192,) * 7
    assert chunked_prefill_mean_ttft_ms(
        mixed8, dp=4, max_batched_tokens=8192,
        body_one=lambda tokens: 0.1 * tokens) == 2662.4
    assert chunked_prefill_mean_ttft_ms(
        mixed8, dp=4, max_batched_tokens=8192,
        body_one=lambda tokens, kv: 0.05 * tokens + 0.01 * kv
    ) == 2170.8800000000006
    assert chunked_prefill_mean_ttft_ms(
        (8192, 8192, 8192, 8192), dp=1, max_batched_tokens=8192,
        body_one=lambda tokens: 0.1 * tokens) == 2048.0
    assert chunked_prefill_mean_ttft_ms(
        (8192, 8192), dp=2, max_batched_tokens=8192,
        body_one=lambda tokens: 0.1 * tokens) == 819.2
    assert chunked_prefill_mean_ttft_ms(
        (16384, 16384, 16384), dp=2, max_batched_tokens=8192,
        body_one=lambda _tokens, _done: 1.0) == 2.6666666666666665
    assert chunked_prefill_mean_ttft_ms(
        (), dp=2, max_batched_tokens=8192,
        body_one=lambda tokens: 0.1 * tokens) == 0.0
    assert chunked_prefill_mean_ttft_ms(
        (4096,), dp=2, max_batched_tokens=8192,
        body_one=lambda tokens: 0.1 * tokens) == 409.6
    # Per-request one request and empty preserve the same edge values.
    assert chunked_prefill_request_ttft_ms(
        (4096,), dp=2, max_batched_tokens=8192,
        body_one=lambda tokens: 0.1 * tokens) == (409.6,)
    assert chunked_prefill_request_ttft_ms(
        (), dp=2, max_batched_tokens=8192,
        body_one=lambda tokens: 0.1 * tokens) == ()


def test_mean_reuses_per_request_and_calls_body_once_per_chunk(monkeypatch):
    calls: list[int] = []

    def body(tokens: int) -> float:
        calls.append(tokens)
        return 0.1 * tokens

    result = chunked_prefill_mean_ttft_ms(
        (16384, 8192), dp=1, max_batched_tokens=8192, body_one=body)
    assert result == 2048.0  # finishes 1638.4 and 2457.6
    assert calls == [8192, 8192, 8192]

    calls.clear()
    finishes = chunked_prefill_request_ttft_ms(
        (16384, 8192), dp=1, max_batched_tokens=8192, body_one=body)
    assert finishes == pytest.approx((1638.4, 2457.6000000000004), rel=1e-15)
    assert calls == [8192, 8192, 8192]

    # No signature lock: introspection must not be used to decide arity.
    def fail(*_args, **_kwargs):
        raise AssertionError("inspect.signature must not be used")

    monkeypatch.setattr(inspect, "signature", fail)
    assert chunked_prefill_request_ttft_ms(
        (8192,), dp=1, max_batched_tokens=8192, body_one=body) == (819.2,)


def test_one_and_two_argument_callables_and_partials_preserve_ordinal_order():
    one_calls: list[int] = []
    two_calls: list[tuple[int, int]] = []

    class OneArg:
        def __call__(self, tokens: int) -> float:
            one_calls.append(tokens)
            return 0.1 * tokens

    class TwoArg:
        def __call__(self, tokens: int, done: int) -> float:
            two_calls.append((tokens, done))
            return _deterministic(tokens, done)

    def raw_two(scale: float, tokens: int) -> float:
        return scale * 0.1 * tokens

    def raw_three(scale: float, tokens: int, done: int) -> float:
        return scale * _deterministic(tokens, done)

    def kw_one(tokens: int, scale: float) -> float:
        return scale * 0.1 * tokens

    def kw_two(tokens: int, done: int, scale: float) -> float:
        return scale * _deterministic(tokens, done)

    assert chunked_prefill_request_ttft_ms(
        (8192, 4096), dp=1, max_batched_tokens=8192, body_one=OneArg()
    ) == pytest.approx((819.2, 1228.8), rel=1e-15)
    assert one_calls == [8192, 4096]
    assert chunked_prefill_request_ttft_ms(
        (8192, 4096), dp=1, max_batched_tokens=8192, body_one=TwoArg()
    ) == pytest.approx((9.0112, 9.0112 + 4.5056), rel=1e-15)
    assert two_calls == [(8192, 8192), (4096, 4096)]
    assert chunked_prefill_request_ttft_ms(
        (8192, 4096), dp=1, max_batched_tokens=8192,
        body_one=functools.partial(raw_two, 1.0)
    ) == pytest.approx((819.2, 1228.8), rel=1e-15)
    assert chunked_prefill_request_ttft_ms(
        (8192, 4096), dp=1, max_batched_tokens=8192,
        body_one=functools.partial(raw_three, 1.0)
    ) == pytest.approx((9.0112, 9.0112 + 4.5056), rel=1e-15)
    assert chunked_prefill_request_ttft_ms(
        (8192, 4096), dp=1, max_batched_tokens=8192,
        body_one=functools.partial(kw_one, scale=1.0)
    ) == pytest.approx((819.2, 1228.8), rel=1e-15)
    assert chunked_prefill_request_ttft_ms(
        (8192, 4096), dp=1, max_batched_tokens=8192,
        body_one=functools.partial(kw_two, scale=1.0)
    ) == pytest.approx((9.0112, 9.0112 + 4.5056), rel=1e-15)


def test_clamped_dp_budget_and_prompt_edges_preserve_old_semantics():
    assert chunked_prefill_request_ttft_ms(
        (0, -5), dp=0, max_batched_tokens=0,
        body_one=lambda tokens: float(tokens)) == (1.0, 2.0)
    assert chunked_prefill_mean_ttft_ms(
        (0, -5), dp=0, max_batched_tokens=0,
        body_one=lambda tokens: float(tokens)) == 1.5
