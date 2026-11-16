"""Explicit request/workload value contracts before hashing or coercion.

[Role]
  Validate one request or workload snapshot without reflection: exact scalar
  types, finite times, non-negative counts, media/prefix consistency, unique
  hashable ids and immutable sequence snapshots.

[Invariants]
  * ``prompt_tokens``/``output_tokens`` are at least one; the remaining counts
    are non-negative and bounded to finite binary64 arithmetic.
  * ``arrival_ms``/``gap_ms`` are finite non-negative numbers returned as
    floats; booleans and numeric strings never coerce.
  * ``prefix_tokens`` and ``media_tokens`` never exceed ``prompt_tokens``;
    ``text_prompt_tokens`` plus media must equal the prompt when declared.
  * Request ids are nonempty, hashable and unique inside one workload.
  * The returned request/metadata sequences are fresh tuples.

[Timeouts & Locks]
  Pure bounded in-memory validation; no I/O, locks or device work.

[Side-Effect Policy]
  Never mutates the supplied record; returns normalized immutable snapshots.
"""
from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

from mindie_sim.schema.common import ModelingError

_ZERO = 0
_ZERO_FLOAT = 0.0
_ONE = 1
_MAX_NUMBER = (1 << 1024) - (1 << 971)

_METADATA_PAIR_SIZE = 2


def _request_int(name, value, *, minimum=_ZERO, reason="workload"):
    """Require one exact bounded integer and return it without coercion."""
    if type(value) is not int or value < minimum or abs(value) > _MAX_NUMBER:
        raise ModelingError(
            f"{reason}_{name}_invalid:type={type(value).__name__}")
    return value


def _request_number(name, value, *, minimum=_ZERO_FLOAT, reason="workload"):
    """Require one finite non-negative number, returned as a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelingError(
            f"{reason}_{name}_invalid:type={type(value).__name__}")
    try:
        number = float(value)
    except OverflowError as error:
        raise ModelingError(f"{reason}_{name}_invalid:overflow") from error
    if not math.isfinite(number) or number < minimum:
        raise ModelingError(f"{reason}_{name}_invalid:type={type(value).__name__}")
    return number


def _validate_request(value):
    """One explicit request contract for construction and raw consumers."""
    request_id = getattr(value, "request_id", None)
    if not isinstance(request_id, str):
        raise ModelingError(
            f"workload_request_id_invalid:type={type(request_id).__name__}")
    if not request_id.strip():
        raise ModelingError("workload.request_id_invalid:empty")
    try:
        hash(request_id)
    except TypeError as error:
        raise ModelingError("workload.request_id_invalid:unhashable") from error
    for name, minimum in (
            ("prompt_tokens", _ONE), ("output_tokens", _ONE),
            ("prefix_tokens", _ZERO), ("priority", _ZERO),
            ("text_prompt_tokens", _ZERO), ("media_tokens", _ZERO),
            ("chain_id", _ZERO)):
        _request_int(name, getattr(value, name, None), minimum=minimum)
    media_encoded = getattr(value, "media_encoded", None)
    if type(media_encoded) is not bool:
        raise ModelingError(
            f"workload_media_encoded_invalid:type={type(media_encoded).__name__}")
    prefix_cache_key = getattr(value, "prefix_cache_key", None)
    if type(prefix_cache_key) is not str:
        raise ModelingError(
            f"workload_prefix_cache_key_invalid:type={type(prefix_cache_key).__name__}")
    times = tuple(
        _request_number(name, getattr(value, name, None), reason="workload")
        for name in ("arrival_ms", "gap_ms"))
    if value.prefix_tokens > value.prompt_tokens:
        raise ModelingError("workload.prefix_exceeds_prompt")
    if value.media_tokens > value.prompt_tokens:
        raise ModelingError("workload.media_exceeds_prompt")
    if (value.text_prompt_tokens > _ZERO
            and value.text_prompt_tokens + value.media_tokens != value.prompt_tokens):
        raise ModelingError("workload.text_media_prompt_mismatch")
    return times


def _validate_workload(value):
    """Validate before hashing ids; snapshot public sequence inputs immutably."""
    from mindie_sim.schema.model import RequestSpec
    requests = getattr(value, "requests", None)
    if not isinstance(requests, Sequence) or isinstance(requests, (str, bytes, bytearray)):
        raise ModelingError("workload.requests_invalid")
    requests = tuple(requests)
    if not requests or any(not isinstance(item, RequestSpec) for item in requests):
        raise ModelingError("workload.requests_invalid")
    for request in requests:
        _validate_request(request)
    if not isinstance(getattr(value, "synthetic", None), bool):
        raise ModelingError("workload.synthetic_invalid")
    ids = [request.request_id for request in requests]
    try:
        unique = set(ids)
    except TypeError as error:
        raise ModelingError("workload.request_id_invalid:unhashable") from error
    if len(ids) != len(unique):
        raise ModelingError("workload.request_id_duplicate")
    metadata = getattr(value, "metadata", None)
    if not isinstance(metadata, Sequence) or isinstance(metadata, (str, bytes, bytearray)):
        raise ModelingError("workload.metadata_invalid")
    pairs = []
    for pair in metadata:
        if (not isinstance(pair, Sequence) or isinstance(pair, (str, bytes, bytearray))
                or len(pair) != _METADATA_PAIR_SIZE
                or any(not isinstance(item, str) for item in pair)):
            raise ModelingError("workload.metadata_invalid")
        pairs.append(tuple(pair))
    return requests, tuple(pairs)


__all__ = ["_request_int", "_validate_request", "_validate_workload"]
