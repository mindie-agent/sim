"""Maintained request/workload construction and bypass-resistance contracts.

[Role]
  Preserve the NR-S7 FIX8/FIX9 workload regressions after retired sidecars:
  request field types, media/prefix boundaries, immutable snapshots, unique
  ids, metadata shape and giant-value rejection.

[Invariants]
  Values are rejected by the canonical ``RequestSpec``/``WorkloadSpec``
  constructors and by a forged record supplied directly to ``WorkloadSpec``;
  no test calls a retired analytical reporter.
"""
from __future__ import annotations

import dataclasses as dc

import pytest

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import RequestSpec, WorkloadSpec


def _request() -> RequestSpec:
    return RequestSpec("ok", 0, 128, 1)


def _forge(record, **changes):
    """Deliberately skip constructors so consumer checks are tested separately."""
    forged = object.__new__(type(record))
    for field in dc.fields(record):
        object.__setattr__(
            forged, field.name,
            changes.get(field.name, getattr(record, field.name)))
    return forged


def _bad_request_fields():
    integers = (-1, True, False, 1.0, 1.5, "1", None, [], {},
                float("nan"), float("inf"), -float("inf"), 10**400, -10**400)
    groups = {name: integers for name in (
        "prompt_tokens", "output_tokens", "prefix_tokens", "priority",
        "text_prompt_tokens", "media_tokens", "chain_id")}
    for name in ("prompt_tokens", "output_tokens"):
        groups[name] += (0,)
    groups.update(
        request_id=([], {}, None, 1, True, "", " \t\n", b"ok"),
        media_encoded=(0, 1, "true", None, [], {}),
    )
    for name in ("arrival_ms", "gap_ms"):
        groups[name] = (-1, -0.1, True, False, "0", None, [], {},
                        float("nan"), float("inf"), -float("inf"),
                        10**400, -10**400)
    return [(name, bad) for name, values in groups.items() for bad in values] + [
        ("prefix_tokens", 129), ("media_tokens", 129),
        ("text_prompt_tokens", 127), ("text_prompt_tokens", 129),
    ]


@pytest.mark.parametrize("field", tuple(RequestSpec.__dataclass_fields__))
def test_request_constructor_replace_mapping_and_workload(field):
    request = _request()
    for name, bad in _bad_request_fields():
        if name != field:
            continue
        changes = {name: bad}
        with pytest.raises(ModelingError):
            RequestSpec(**{**dc.asdict(request), **changes})
        with pytest.raises(ModelingError):
            dc.replace(request, **changes)
        with pytest.raises(ModelingError):
            RequestSpec.from_mapping({**request.to_dict(), **changes}, 0)
        forged = _forge(request, **changes)
        with pytest.raises(ModelingError):
            WorkloadSpec((forged,))


def _bad_metadata():
    return (None, 1, True, {}, set(), "ab", b"ab", bytearray(b"ab"),
            (None,), (1,), ((),), (("key",),), (("a", "b", "c"),),
            ("ab",), (b"ab",), ({"a": "b"},), ((1, "b"),),
            (("a", None),), (("a", []),), (("ok", "v"), ("bad", False)))


def test_metadata_constructor_replace_and_immutable_snapshot():
    from collections import UserList, deque
    request = _request()
    workload = WorkloadSpec((request,))
    for bad in _bad_metadata():
        with pytest.raises(ModelingError):
            WorkloadSpec((request,), metadata=bad)
        with pytest.raises(ModelingError):
            dc.replace(workload, metadata=bad)
    for container in (list, tuple, UserList, deque):
        pair = ["source", "fixture"]
        supplied = container([pair])
        normalized = dc.replace(workload, metadata=supplied)
        pair[1] = "changed"
        assert normalized.metadata == (("source", "fixture"),)
        assert type(normalized.metadata) is tuple
        assert type(normalized.metadata[0]) is tuple


def test_valid_request_shorthand_media_boundaries_and_roundtrip():
    class RequestId(str):
        pass
    requests = [
        RequestSpec("ok", 0, 128, 1),
        RequestSpec("media", 1, 128, 1, prefix_tokens=128,
                    text_prompt_tokens=96, media_tokens=32, priority=1,
                    chain_id=1, gap_ms=2, media_encoded=True),
        RequestSpec("media_only", 0, 128, 1, media_tokens=128),
        RequestSpec(RequestId("string_subclass"), 0, 128, 1),
    ]
    for request in requests:
        assert type(request.arrival_ms) is type(request.gap_ms) is float
        assert dc.replace(request) == request
        roundtrip = RequestSpec.from_mapping(request.to_dict(), 0)
        assert roundtrip.to_dict() == request.to_dict()
        WorkloadSpec([request], metadata=[["source", "fixture"]])
    with pytest.raises(ModelingError):
        RequestSpec("bad", 0, 128, 1, text_prompt_tokens=100, media_tokens=32)
    with pytest.raises(ModelingError):
        RequestSpec("bad", 0, 128, 1, prefix_tokens=129)


def test_workload_rejects_unconsumed_metadata_stream():
    events = []

    def pairs():
        events.append("consumed")
        yield ("source", "fixture")

    stream = pairs()
    with pytest.raises(ModelingError):
        WorkloadSpec((_request(),), metadata=stream)
    assert events == []
    stream.close()


_BAD_REQUESTS = ((), [], {}, set(), "", b"", None, (123,), [None])


@pytest.mark.parametrize("bad", _BAD_REQUESTS)
def test_workload_rejects_invalid_request_containers(bad):
    workload = WorkloadSpec((_request(),))
    with pytest.raises(ModelingError, match=r"^workload\.requests_invalid$"):
        WorkloadSpec(requests=bad)
    with pytest.raises(ModelingError, match=r"^workload\.requests_invalid$"):
        dc.replace(workload, requests=bad)


@pytest.mark.parametrize("container", (tuple, list))
@pytest.mark.parametrize("synthetic", (False, True))
def test_valid_sequences_normalize_without_aliasing(container, synthetic):
    from collections import UserList
    base = WorkloadSpec((_request(),))
    req = base.requests[0]
    for expected in ((req,), (req, dc.replace(req, request_id=req.request_id + "_other"))):
        supplied = container(expected)
        workloads = (
            WorkloadSpec(requests=supplied, synthetic=synthetic),
            dc.replace(base, requests=supplied, synthetic=synthetic),
        )
        if isinstance(supplied, (list, UserList)):
            supplied.clear()
        for workload in workloads:
            assert type(workload.requests) is tuple
            assert workload.requests == expected
            assert workload.synthetic is synthetic
            roundtripped = WorkloadSpec.from_mapping({
                "requests": [r.to_dict() for r in expected],
                "synthetic": synthetic}).requests
            assert [r.to_dict() for r in roundtripped] == [
                r.to_dict() for r in expected]


@pytest.mark.parametrize("bad", (1, "yes", 0, None, (), []))
def test_synthetic_requires_bool(bad):
    workload = WorkloadSpec((_request(),))
    with pytest.raises(ModelingError, match=r"^workload\.synthetic_invalid$"):
        WorkloadSpec(requests=workload.requests, synthetic=bad)
    with pytest.raises(ModelingError, match=r"^workload\.synthetic_invalid$"):
        dc.replace(workload, synthetic=bad)


def test_duplicate_ids_reject_at_all_construction_paths():
    workload = WorkloadSpec((_request(),))
    req = workload.requests[0]
    requests = (
        req,
        dc.replace(req, request_id=req.request_id + "_other"),
        dc.replace(req, arrival_ms=req.arrival_ms + 1),
    )
    for call in (
        lambda: WorkloadSpec(requests=requests),
        lambda: dc.replace(workload, requests=requests),
        lambda: WorkloadSpec.from_mapping(
            {"requests": [r.to_dict() for r in requests]}),
    ):
        with pytest.raises(ModelingError, match=r"^workload\.request_id_duplicate$"):
            call()


def test_giant_values_cannot_break_error_formatting():
    request = _request()
    giant = 10**5000
    for name in RequestSpec.__dataclass_fields__:
        for bad in (giant, [giant], {"bad": giant}):
            changes = {name: bad}
            with pytest.raises(ModelingError):
                RequestSpec(**{**dc.asdict(request), **changes})
            with pytest.raises(ModelingError):
                dc.replace(request, **changes)
            with pytest.raises(ModelingError):
                RequestSpec.from_mapping({**request.to_dict(), **changes}, 0)
            with pytest.raises(ModelingError):
                WorkloadSpec((_forge(request, **changes),))
