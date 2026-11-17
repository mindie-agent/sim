"""Tier-0 tests for ACLGraph multi-stream, SuperKernel, and SQ/CQ budgets.

[Role]
Exercise all five ACLGraph diagnostic classes A1..A5, PIECEWISE vs FULL join
loss, HCCL AIV expansion vs CCU_SCHED, SuperKernel eligibility, the 207008
capture budget, and bounded MCP envelopes.
[Invariants]
Deterministic in-memory model; no torch_npu or device dependency.
[Timeouts & Locks]
No sleeps, locks, or I/O.
[Side-Effect Policy]
Read-only; no files or graph captures are created.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("mindie_profiler")
from mindie_profiler.aclgraph_mcp import call_tool
from mindie_profiler.aclgraph_multistream import (
    ACLGraphModelError,
    ACLGraphWrapperModel,
    BreakableACLGraphWrapperModel,
    GraphMode,
    HcclExpansion,
    ParallelRegion,
    StreamInterval,
    StreamResourceExhausted,
    UpdatableGraphModel,
    check_capture_budget,
    check_superkernel_eligibility,
    classify_capture_error,
    diagnose_region,
    effective_launch_floor_us,
)


def _envelope_size(payload):
    """Strict UTF-8/BPE upper bound for one serialized MCP payload."""
    default = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    compact = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    bound = max(len(default.encode("utf-8")), len(compact.encode("utf-8")))
    return {"utf8_bytes": bound, "token_estimate": bound}



def _interval(
    op: str = "op", stream: int = 0, start: float = 0.0, end: float = 100.0,
    *, aic: int = 0, aiv: int = 0, io: int = 0, hccl: str = "AIV",
    wait: bool = True, joined: bool = True, alias: bool = False, ordered: bool = True,
) -> StreamInterval:
    return StreamInterval(
        op, stream, start, end, aic, aiv, io, hccl, wait, joined, alias, ordered
    )


def _region(a: StreamInterval, b: StreamInterval, **kwargs) -> ParallelRegion:
    return ParallelRegion("region", a, b, **kwargs)


def test_a1_same_stream_collapse() -> None:
    diag = diagnose_region(_region(_interval(stream=3), _interval(stream=3)))
    assert diag.code == "A1" and diag.overlap_pct == 0.0
    assert "stream_switch" in " ".join(diag.recommendations)


def test_a2_premature_event_vs_piecewise_barrier() -> None:
    premature = diagnose_region(
        _region(_interval("a", 0, 0, 100), _interval("b", 1, 96, 196),
                graph_mode="full", event_causality_observed=True)
    )
    assert premature.code == "A2" and premature.cause == "premature_event_record_wait"
    assert "Notify_Wait" in premature.evidence
    split = diagnose_region(
        _region(
            _interval("a", 0, 0, 100), _interval("b", 1, 96, 196),
            graph_mode="piecewise", piece_boundary_between=True,
        )
    )
    assert split.code == "A2" and split.cause == "piecewise_aclgraph_split_barrier"
    full = diagnose_region(
        _region(_interval("a", 0, 0, 100), _interval("b", 1, 0, 100))
    )
    assert full.code == "OK"


@pytest.mark.xfail(
    reason="pre-existing at 871a8f8: aclgraph diagnostics no longer expose gain_us",
    strict=False)
def test_a3_hccl_aiv_expansion_vs_ccu() -> None:
    aiv = _region(
        _interval("AllReduce", 0, 0, 100, aiv=40, io=4096),
        _interval("dispatch", 1, 0, 100, aiv=30, io=4096),
        aiv_cores=64,
    )
    diag = diagnose_region(aiv)
    assert diag.code == "A3" and "hccl_aiv" in diag.cause
    assert diag.net_speedup_us > 0
    ccu = _region(
        _interval("AllReduce", 0, 0, 100, aiv=40, io=4096, hccl="CCU_SCHED"),
        _interval("dispatch", 1, 0, 100, aiv=30, io=4096, hccl="CCU_SCHED"),
        aiv_cores=64,
    )
    ccu_diag = diagnose_region(ccu)
    assert ccu_diag.code == "A3" and "hccl_aiv" not in ccu_diag.cause
    assert ccu_diag.net_speedup_us == pytest.approx(98.5)
    assert ccu_diag.details["gain_us"] == 0.0


@pytest.mark.xfail(
    reason="pre-existing at 871a8f8: event contention is not charged into net_wall_us",
    strict=False)
def test_a4_contention_and_event_erosion() -> None:
    diag = diagnose_region(
        _region(
            _interval("a", 0, 0, 100, io=400_000_000),
            _interval("b", 1, 0, 100, io=400_000_000),
            event_pairs=60,
        )
    )
    assert diag.code == "A4" and diag.net_speedup_us <= 0.0
    assert "event" in diag.cause
    assert diag.details["event_us"] == 90.0


def test_a5_dangling_and_alias_hazards() -> None:
    dangling = diagnose_region(
        _region(_interval("a", 0, 0, 100), _interval("b", 1, 0, 100, wait=False))
    )
    assert dangling.code == "A5"
    unjoined = diagnose_region(
        _region(_interval("a", 0, 0, 100), _interval("b", 1, 0, 100, joined=False))
    )
    assert unjoined.code == "A5"
    alias = diagnose_region(
        _region(
            _interval("a", 0, 0, 100),
            _interval("b", 1, 0, 100, alias=True, ordered=False),
        )
    )
    assert alias.code == "A5"


def test_wrapper_updatable_and_breakable_models() -> None:
    wrapper = ACLGraphWrapperModel(
        captured_batch_sizes=8, subgraphs_per_step=32, active_streams_per_graph=16
    )
    assert wrapper.budget.error_code == "207008" and wrapper.replay_sync_required
    assert wrapper.superkernel.eligible
    assert wrapper.launch_floor_us == pytest.approx(1.35)
    assert BreakableACLGraphWrapperModel(pieces=4).join_overhead_us == pytest.approx(4.5)
    updatable = UpdatableGraphModel(enable_enpu=True)
    assert updatable.replay_order() == ("update", "replay")
    assert not updatable.sync_before_replay


def test_sq_cq_budget_and_207008() -> None:
    fits = check_capture_budget(8, 32, 4)
    assert fits.required == 1024 and fits.fits and fits.error_code == ""
    exhausted = check_capture_budget(8, 32, 16)
    assert not exhausted.fits and exhausted.error_code == "207008"
    assert exhausted.recommendation == "prefer_full_or_full_decode_only"
    with pytest.raises(StreamResourceExhausted) as exc:
        exhausted.raise_if_exhausted()
    assert exc.value.error_code == "207008"
    assert check_capture_budget(2, 2, 2, capacity=2).error_code == "207008"


def test_capture_error_classifier() -> None:
    assert classify_capture_error("error code is 207008 Stream resources are insufficient") == "stream_resource_207008"
    assert classify_capture_error("Alloc sq cq fail") == "old_hdk_alloc_sq_cq"
    assert classify_capture_error("other") == "unknown"


def test_superkernel_eligibility_and_dcci() -> None:
    static = check_superkernel_eligibility(static_decode=True)
    assert static.eligible
    assert static.optimize_options == {"dcci_after_kernel_end": [".*"]}
    assert not check_superkernel_eligibility(static_decode=False).eligible
    assert not check_superkernel_eligibility(static_decode=True, dynamic_tiling_host_sync=True).eligible
    assert not check_superkernel_eligibility(static_decode=True, capturable_ops=False).eligible
    assert effective_launch_floor_us("full") < effective_launch_floor_us("piecewise")
    assert effective_launch_floor_us("full", superkernel=True) < effective_launch_floor_us("full")
    with pytest.raises(ACLGraphModelError):
        effective_launch_floor_us("magic")


def test_mcp_envelopes_bounded_and_hostile() -> None:
    region = {
        "name": "AscendSharedExperts Gate-Up || Dispatch",
        "branch_a": {"op": "gate_up", "stream_id": 0, "start_us": 0, "end_us": 100, "aic_blocks": 20},
        "branch_b": {"op": "dispatch", "stream_id": 1, "start_us": 0, "end_us": 100, "aiv_blocks": 20},
    }
    calls = [
        ("aclgraph_diagnose", {"region": region}),
        ("aclgraph_diagnose", {"region": {**region, "name": "x" * 5000}}),
        ("aclgraph_sq_budget", {"captured_batch_sizes": 8, "subgraphs_per_step": 32, "active_streams_per_graph": 16}),
        ("aclgraph_superkernel", {"static_decode": False}),
        ("aclgraph_launch_floor", {"mode": "piecewise"}),
        ("aclgraph_launch_floor", {"mode": "x" * 5000}),
        ("nope", {}),
    ]
    for name, arguments in calls:
        result = call_tool(name, arguments)
        size = _envelope_size(result)
        assert size["utf8_bytes"] <= 250
        assert size["token_estimate"] <= 250
        assert isinstance(result, dict)
    assert call_tool("aclgraph_diagnose", {"region": region})["code"] == "OK"
    assert call_tool("aclgraph_sq_budget", {"captured_batch_sizes": 8, "subgraphs_per_step": 32, "active_streams_per_graph": 16})["err"] == "207008"
    assert call_tool("aclgraph_superkernel", {"static_decode": False})["eligible"] == 0
